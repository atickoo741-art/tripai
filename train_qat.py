import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, Conv1D
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from data_preparation import load_and_preprocess_wikitext2
from torch.utils.tensorboard import SummaryWriter
import time
from typing import List

# Help CUDA memory fragmentation in long runs (must be set before first CUDA alloc)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
print("train_qat.py script has started execution!", flush=True) # <--- Add this line

# Define the FakeQuantize module with STE
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, num_bits, symmetric=0):
        """Per-channel fake quantization with straight-through estimator.
        
        Args:
            x: Input tensor
            scale: Scale tensor, shape [C] or [1] for per-tensor
            zero_point: Zero point tensor, shape [C] or [1] for per-tensor
            num_bits: Number of bits for quantization
            symmetric: If 1, use symmetric quantization (range will be [-2^(n-1), 2^(n-1)-1])
        """
        ctx.save_for_backward(x, scale, zero_point)
        ctx.num_bits = num_bits
        ctx.symmetric = bool(symmetric)
        
        if ctx.symmetric:
            qmin = -2 ** (num_bits - 1)
            qmax = 2 ** (num_bits - 1) - 1
        else:
            qmin = 0
            qmax = 2 ** num_bits - 1
            
        # Ensure scale and zero_point have the right shape for broadcasting
        if scale.dim() < x.dim():
            # Add singleton dimensions to match x's dimensions starting from the last dimension
            for _ in range(x.dim() - scale.dim()):
                scale = scale.unsqueeze(-1)
                zero_point = zero_point.unsqueeze(-1)
        
        # Save these for backward
        ctx.scale = scale
        ctx.zero_point = zero_point
        
        x_quant = torch.round(x / scale + zero_point)
        x_quant = torch.clamp(x_quant, qmin, qmax)
        x_dequant = (x_quant - zero_point) * scale
        
        return x_dequant
    
    @staticmethod
    def backward(ctx, grad_output):
        """Straight-through estimator for the backward pass."""
        x, scale, zero_point = ctx.saved_tensors
        
        # For STE, we only pass gradients through for the input (x)
        # Scale and zero_point gradients are None because they're computed
        # elsewhere in the QATW4A4Linear class
        grad_input = grad_output.clone()
        
        # Return None for all other gradients (scale, zero_point, num_bits, symmetric)
        return grad_input, None, None, None, None


class QATW4A4Linear(nn.Module):
    def __init__(self, original_linear_layer, num_bits=4, init_clip_val=2.0):
        super().__init__()
        self.original_linear = original_linear_layer
        self.num_bits = num_bits
        self.fake_quantize = FakeQuantize.apply
        # Support both nn.Linear (weight shape [out, in]) and GPT-2 Conv1D (weight shape [in, out])
        if hasattr(original_linear_layer, 'in_features') and hasattr(original_linear_layer, 'out_features'):
            # nn.Linear-like
            self.in_features = original_linear_layer.in_features
            self.out_features = original_linear_layer.out_features
            self.weight_is_out_in = True
        else:
            # Conv1D-like (GPT-2)
            weight_shape = original_linear_layer.weight.shape
            # Conv1D stores weight as [in_features, out_features]
            self.in_features = weight_shape[0]
            self.out_features = weight_shape[1]
            self.weight_is_out_in = False
        
        # Get device from original linear layer
        self.device = next(original_linear_layer.parameters()).device
        
        # PACT activation clipping
        # self.activation_clip = nn.Parameter(
        #     torch.tensor(init_clip_val, dtype=torch.float32, device=self.device)
        # )

        self.activation_clip = nn.Parameter(
            torch.ones(self.in_features, device=self.device) * init_clip_val
        )
        
        # SmoothQuant parameters
        # self.raw_alpha = nn.Parameter(
        #     torch.tensor(-1.0, device=self.device)  # Will be passed through sigmoid
        # )

        # Initialize per-channel smooth factors to 1.0
        self.smooth_factor = nn.Parameter(torch.ones(self.in_features, device=self.device))

        if hasattr(self.original_linear, 'weight'):
            self.original_linear.weight.requires_grad_(True)
        self.activation_clip.requires_grad_(True)
        # self.raw_alpha.requires_grad_(True)
        self.smooth_factor.requires_grad_(True)
        
        # Track per-channel input statistics (min/max for asymmetric quantization)
        self.register_buffer('input_min', 
            torch.zeros(self.in_features, dtype=torch.float32, device=self.device)
        )
        self.register_buffer('input_max', 
            torch.zeros(self.in_features, dtype=torch.float32, device=self.device)
        )
        self.momentum = 0.1  # For EMA of input statistics
        
        # Buffers for quantization parameters
        # Per-channel for weights ([out_features])
        self.register_buffer('weight_scale', 
            torch.ones(self.out_features, dtype=torch.float32, device=self.device)
        )
        self.register_buffer('weight_zero_point', 
            torch.zeros(self.out_features, dtype=torch.float32, device=self.device)
        )
        # Per-channel for activations ([in_features])
        self.register_buffer('act_scale', 
            torch.ones(self.in_features, dtype=torch.float32, device=self.device)
        )
        self.register_buffer('act_zero_point', 
            torch.zeros(self.in_features, dtype=torch.float32, device=self.device)
        )

        W_base = self.original_linear.weight
        W_cpu = W_base.detach().to(torch.float32).cpu()

        self.min_base = W_cpu.min(dim=1).values    
        self.max_base = W_cpu.max(dim=1).values    

        self.min_base = self.min_base.to(device=self.device)
        self.max_base = self.max_base.to(device=self.device)

        

    @property 
    def constrained_clip_val(self):
        return torch.clamp(torch.abs(self.activation_clip), min=0.5, max=6.0)


    @property
    def weight(self):
        """Return the merged weights for PEFT compatibility"""
        return self._get_merged_weights()

    @weight.setter  
    def weight(self, value):
        """Set the original linear weight for PEFT compatibility"""
        self.original_linear.weight = nn.Parameter(value.detach().clone())
        
    def _calculate_quant_params(self, min_vals, max_vals, num_bits, symmetric=False):
        """Calculate scale and zero-point for quantization, supporting per-channel.
        
        Args:
            min_vals: Tensor of minimum values, shape [C] or [1] for per-tensor
            max_vals: Tensor of maximum values, shape [C] or [1] for per-tensor
            num_bits: Number of bits for quantization
            symmetric: If True, use symmetric quantization (range will be [-2^(n-1), 2^(n-1)-1])
        """
        # Ensure min_vals and max_vals are tensors and have proper device
        if not isinstance(min_vals, torch.Tensor):
            min_vals = torch.tensor(min_vals, device=self.device, dtype=torch.float32)
        if not isinstance(max_vals, torch.Tensor):
            max_vals = torch.tensor(max_vals, device=self.device, dtype=torch.float32)
            
        # Ensure both are on the same device and dtype
        min_vals = min_vals.to(device=self.device, dtype=torch.float32)
        max_vals = max_vals.to(device=self.device, dtype=torch.float32)
        
        if symmetric:
            # For symmetric quantization, range is [-2^(n-1), 2^(n-1)-1]
            qmin = -2 ** (num_bits - 1)
            qmax = 2 ** (num_bits - 1) - 1
            
            # For symmetric quantization, we use the maximum absolute value
            abs_max = torch.max(torch.abs(min_vals), torch.abs(max_vals))
            # Add small epsilon to avoid zero scale
            abs_max = torch.clamp(abs_max, min=1e-8)
            
            scale = abs_max / (qmax)  # qmax for symmetric since range is -qmax to qmax
            zero_point = torch.zeros_like(scale, dtype=torch.float32)
        else:
            # Asymmetric quantization
            qmin = 0
            qmax = 2 ** num_bits - 1
                
            # min_vals = torch.min(min_vals, torch.zeros_like(min_vals))  # clip to <= 0
            # max_vals = torch.max(max_vals, torch.zeros_like(max_vals))  # clip to >= 0
            min_vals = torch.minimum(min_vals, max_vals)

            # Ensure we have a non-zero range
            range_vals = max_vals - min_vals
            range_vals = torch.clamp(range_vals, min=1e-8)
            
            scale = range_vals / (qmax - qmin)
            zero_point = qmin - min_vals / scale
            zero_point = torch.clamp(zero_point, qmin, qmax)
        
        return scale, zero_point

    def _get_merged_weights(self):
        """Merge base weights with LoRA deltas if they exist."""
        # Cache during eval mode
        if not self.training and hasattr(self, '_cached_merged_weights'):
            return self._cached_merged_weights
            
        base_weight = self.original_linear.weight
        # Normalize to [out_features, in_features] for internal computation
        if not self.weight_is_out_in:
            base_weight = base_weight.t()
        
        if hasattr(self.original_linear, 'lora_A') and hasattr(self.original_linear, 'lora_B'):
            lora_A = self.original_linear.lora_A.default.weight
            lora_B = self.original_linear.lora_B.default.weight
            lora_scaling = self.original_linear.scaling['default']
            
            lora_delta = lora_B @ lora_A * lora_scaling
            
            if lora_delta.shape != base_weight.shape:
                delta = torch.zeros_like(base_weight)
                delta[:lora_delta.shape[0], :lora_delta.shape[1]] = lora_delta
                lora_delta = delta
                
            merged = base_weight + lora_delta
        else:
            merged = base_weight
        
        # Cache for eval mode
        if not self.training:
            self._cached_merged_weights = merged
            
        return merged

        
    # @property
    # def alpha(self):
    #     # Constrain alpha between 0 and 1 using sigmoid
    #     return torch.sigmoid(self.raw_alpha)

    def update_input_stats(self, x):
        # x shape: [batch_size, seq_len, in_features] or [batch_size, in_features]
        # Flatten all dimensions except the last one (features)
        x_flat = x.reshape(-1, x.size(-1))  # [N, in_features]
        
        # Calculate min and max across batch and sequence dimensions
        current_min = x_flat.min(dim=0).values  # [in_features]
        current_max = x_flat.max(dim=0).values  # [in_features]
        
        # Update running min/max
        if not hasattr(self, 'input_min'):
            # First update, initialize with current values
            self.input_min = current_min
            self.input_max = current_max
        else:
            # Track the most extreme values seen so far
            self.input_min = torch.minimum(self.input_min, current_min)
            self.input_max = torch.maximum(self.input_max, current_max)
        
        return current_min, current_max

    def forward(self, input):
        # input shape: [batch_size, seq_len, in_features] or [batch_size, in_features]
        # Ensure input_absmax is the right shape [in_features]
        if input.dim() == 3:
            in_features = input.size(-1)
            if self.input_min.dim() != 1 or self.input_min.size(0) != in_features:
                self.input_min = torch.zeros(in_features, device=input.device, dtype=torch.float32)
                self.input_max = torch.zeros(in_features, device=input.device, dtype=torch.float32)
        
        # Update input statistics during training
        if self.training:
            current_min, current_max = self.update_input_stats(input)
        
        # Get merged weights [out_features, in_features] - COMPUTE ONLY ONCE
        # merged_weights = self._get_merged_weights()
        
        # Calculate scaling factor using SmoothQuant formula - COMPUTE ONLY ONCE
        # if not self.training and hasattr(self, '_cached_scaling_factor'):
        #     scaling_factor = self._cached_scaling_factor
        # else:
        #     # Calculate max abs value for each input channel (column-wise max)
        #     weight_absmax_per_input_channel = merged_weights.abs().amax(dim=0, keepdim=False)
            
        #     # s_j = (|X_j|^alpha) * (|W_j|^(1-alpha))
        #     scaling_factor = (self.input_absmax ** self.alpha) * \
        #                     (weight_absmax_per_input_channel ** (1 - self.alpha))
        #     scaling_factor = scaling_factor.clamp(min=1e-8, max=1e3)
        
        #     # Cache for eval mode
        #     if not self.training:
        #         self._cached_scaling_factor = scaling_factor
        
        # Reshape smooth_factor for broadcasting: [in_features] -> [1, 1, in_features] or [1, in_features]
        if input.dim() == 3:  # [batch, seq_len, in_features]
            smooth_factor = self.smooth_factor.view(1, 1, -1)
        else:  # [batch, in_features]
            smooth_factor = self.smooth_factor.view(1, -1)
            
        # Apply per-channel scaling to input: x = x / s_j
        scaled_input = input / smooth_factor
        
        # Get merged weights and apply per-channel scaling: W = W * s_j
        merged_weights = self._get_merged_weights()  # [out_features, in_features]
        scaled_weights = merged_weights * self.smooth_factor.unsqueeze(0)  # [out_features, in_features]
        
        # PACT: per-channel clipping for activations
        clip_vals = self.constrained_clip_val  # [in_features]
        # Reshape for broadcasting: [in_features] -> [1, 1, in_features] or [1, in_features]
        if scaled_input.dim() == 3:  # [batch, seq_len, in_features]
            clip_vals = clip_vals.view(1, 1, -1)
        else:  # [batch, in_features]
            clip_vals = clip_vals.view(1, -1)
        
        # Apply symmetric per-channel clipping to [-a, a]
        clipped_input = torch.clamp(scaled_input, -clip_vals, clip_vals)
        
        if not self.training and hasattr(self, '_cached_quant_params'):
            weight_scale, weight_zero_point, act_scale, act_zero_point = self._cached_quant_params
        else:
            # Quantization parameters for weights (per-channel, symmetric)
            weight_mins = scaled_weights.min(dim=1, keepdim=True)[0]
            weight_maxs = scaled_weights.max(dim=1, keepdim=True)[0]
            weight_scale, weight_zero_point = self._calculate_quant_params(
                weight_mins, weight_maxs, self.num_bits, symmetric=True
            )
            
            # Quantization parameters for activations (per-channel, symmetric)
            # Use the learnable symmetric clip as bounds for both train and eval
            act_min = -self.constrained_clip_val
            act_max = self.constrained_clip_val
            act_scale, act_zero_point = self._calculate_quant_params(
                act_min, act_max, self.num_bits, symmetric=True
            )
            
            # Cache for eval mode
            if not self.training:
                self._cached_quant_params = (weight_scale, weight_zero_point, act_scale, act_zero_point)
        
        # Store scales for potential use
        # Ensure correct shapes: [out_features] for weights, [1] for activations
        # Ensure no NaN or zero values in scales before storing
        weight_scale_safe = torch.clamp(weight_scale.detach().flatten(), min=1e-8)
        weight_zero_point_safe = weight_zero_point.detach().flatten()
        act_scale_safe = torch.clamp(act_scale.detach().flatten(), min=1e-8)
        act_zero_point_safe = act_zero_point.detach().flatten()
        
        # Check for NaN values and replace with safe defaults
        if torch.isnan(weight_scale_safe).any():
            print(f"Warning: NaN detected in weight_scale, replacing with safe values")
            weight_scale_safe = torch.where(torch.isnan(weight_scale_safe), 
                                           torch.tensor(1e-4, device=self.device), 
                                           weight_scale_safe)
        
        # Using copy_() to maintain buffer status and proper device placement
        self.weight_scale.copy_(weight_scale_safe)
        self.weight_zero_point.copy_(weight_zero_point_safe)
        self.act_scale.copy_(act_scale_safe)
        self.act_zero_point.copy_(act_zero_point_safe)
        
        # Quantize
        # Using positional arguments (can't use keyword args with Function.apply)
        # Reshape activation quant params for broadcasting along the last (channel) dim
        if clipped_input.dim() == 3:  # [batch, seq_len, in_features]
            act_scale_b = act_scale.view(1, 1, -1)
            act_zero_point_b = act_zero_point.view(1, 1, -1)
        else:  # [batch, in_features]
            act_scale_b = act_scale.view(1, -1)
            act_zero_point_b = act_zero_point.view(1, -1)

        quant_input = self.fake_quantize(
            clipped_input, act_scale_b, act_zero_point_b, self.num_bits, 1  # 1 for symmetric
        )
        quant_weight = self.fake_quantize(
            scaled_weights, weight_scale, weight_zero_point, self.num_bits, 1  # 1 for symmetric
        )

        # Perform the linear operation with quantized input and weights
        # Note: Bias is not quantized, it's usually kept in full precision.
        # IMPORTANT: GPT-2 uses Conv1D, where weights are (in_features, out_features).
        # F.linear expects weights as (out_features, in_features) and applies transpose internally (input @ weight.T).
        # So, we need to transpose quant_weight before passing it to F.linear to match Conv1D behavior.
        output = F.linear(quant_input, quant_weight, self.original_linear.bias)
        return output
    def eval(self):
        """Override eval to pre-compute cached values"""
        super().eval()
        if hasattr(self, '_cached_scaling_factor'):
            del self._cached_scaling_factor
        if hasattr(self, '_cached_merged_weights'):
            del self._cached_merged_weights  
        if hasattr(self, '_cached_quant_params'):
            del self._cached_quant_params
        
        # Pre-compute all cached values with a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, self.in_features, device=next(self.parameters()).device)
            _ = self.forward(dummy_input)
        
        return self

    def train(self, mode=True):
        """Override train to clear caches"""
        super().train(mode)
        if hasattr(self, '_cached_scaling_factor'):
            del self._cached_scaling_factor
        if hasattr(self, '_cached_merged_weights'):
            del self._cached_merged_weights
        if hasattr(self, '_cached_quant_params'):
            del self._cached_quant_params
        return self

    def check_gradients(self):
        """Debug method to check gradient flow"""
        print(f"activation_clip grad: {self.activation_clip.grad}")
        if hasattr(self, 'smooth_factor'):
            grad = self.smooth_factor.grad
            grad_info = 'None' if grad is None else f"mean={grad.mean().item():.6f}"
            print(f"smooth_factor grad: {grad_info}")
        if hasattr(self.original_linear, 'lora_A'):
            print(f"lora_A grad: {self.original_linear.lora_A.default.weight.grad is not None}")
            print(f"lora_B grad: {self.original_linear.lora_B.default.weight.grad is not None}")

# Custom Quantization-Aware GPT2Block
class QuantizedGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        # Replace original Linear layers with QATW4A4Linear
        self.attn.c_attn = QATW4A4Linear(self.attn.c_attn)
        self.attn.c_proj = QATW4A4Linear(self.attn.c_proj)
        self.mlp.c_fc = QATW4A4Linear(self.mlp.c_fc)
        self.mlp.c_proj = QATW4A4Linear(self.mlp.c_proj)


    def forward(self, *args, **kwargs):
        """
        Robust forward method that handles both positional and keyword arguments
        to avoid conflicts with multiple argument passing styles.
        """
        # Extract arguments in order, with fallbacks to kwargs
        hidden_states = args[0] if len(args) > 0 else kwargs.get('hidden_states')
        layer_past = args[1] if len(args) > 1 else kwargs.get('layer_past', None)
        attention_mask = args[2] if len(args) > 2 else kwargs.get('attention_mask', None)
        head_mask = args[3] if len(args) > 3 else kwargs.get('head_mask', None)
        encoder_hidden_states = args[4] if len(args) > 4 else kwargs.get('encoder_hidden_states', None)
        encoder_attention_mask = args[5] if len(args) > 5 else kwargs.get('encoder_attention_mask', None)
        use_cache = args[6] if len(args) > 6 else kwargs.get('use_cache', False)
        output_attentions = args[7] if len(args) > 7 else kwargs.get('output_attentions', False)
        
        # Remove the extracted arguments from kwargs to avoid passing them twice
        kwargs_clean = {k: v for k, v in kwargs.items() if k not in [
            'hidden_states', 'layer_past', 'attention_mask', 'head_mask',
            'encoder_hidden_states', 'encoder_attention_mask', 'use_cache', 'output_attentions'
        ]}
        
        # Attention part
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            **kwargs_clean,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        # Residual connection and LayerNorm
        hidden_states = attn_output + hidden_states
        hidden_states = self.ln_1(hidden_states)

        # MLP part
        mlp_output = self.mlp(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            **kwargs_clean,
        )

        # Residual connection and LayerNorm
        hidden_states = mlp_output + hidden_states
        hidden_states = self.ln_2(hidden_states)

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs




# 1. Load the GPT-2 small model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set up device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.set_default_dtype(torch.float32)  # MPS works best with float32
    print("Using MPS device (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_dtype(torch.float32)  # Change from bfloat16 to float32
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    torch.set_default_dtype(torch.float32)
    print("Using CPU")

# Load model with appropriate settings
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Change from bfloat16 to float32
    device_map="auto" if device.type == "cuda" else None,
)
if device.type != "cuda":  # For MPS and CPU, explicitly move to device
    model = model.to(device)

# Add a pad token to the tokenizer if it doesn't have one
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

print(f"Model '{model_name}' loaded successfully for QAT.")
print(f"Model device: {model.device}")
print(f"Number of parameters: {model.num_parameters()}")

def apply_quantization_to_model(model):
    """Apply QATW4A4Linear to all linear layers in the model"""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, Conv1D)):
            # Skip if already quantized
            if hasattr(module, 'original_linear'):
                continue
                
            # Get parent module and attribute name
            parent = model
            attrs = name.split('.')
            for attr in attrs[:-1]:
                parent = getattr(parent, attr)
            
            # Replace with quantized version
            setattr(parent, attrs[-1], QATW4A4Linear(module))


def apply_quantization_to_peft_model(peft_model):
    """Apply QATW4A4Linear to all linear layers within the peft_model's base_model."""
    base_model_to_quantize = peft_model.base_model.model

    for name, module in base_model_to_quantize.named_modules():
        if isinstance(module, (nn.Linear, Conv1D)):
            # This check is crucial to ensure we don't re-wrap if somehow already done
            if hasattr(module, 'original_linear'): # This is how your QATW4A4Linear signals it's a wrapper
                continue # Skip if it's already your custom QAT module

            print(f"Applying quantization to {name}")

            parent = base_model_to_quantize
            attrs = name.split('.')
            for attr in attrs[:-1]:
                parent = getattr(parent, attr)
            
            # Replace the original (now LoRA-enabled) linear layer with your QATW4A4Linear wrapper
            setattr(parent, attrs[-1], QATW4A4Linear(module))

# print("\n--- Applying quantization to model ---")

# apply_quantization_to_model(model)
# 3. Prepare the model for LoRA fine-tuning
print("\n--- Applying LoRA Adapters to quantized model ---")
lora_config = LoraConfig(
    r=8,  # Rank of the update matrices (lower = less parameters)
    lora_alpha=32,  # Alpha parameter for LoRA scaling
    target_modules=["c_attn", "c_proj", "c_fc"],  # Target attention and feed-forward layers
    lora_dropout=0.05,
    bias="none",  # Don't train bias parameters
    task_type=TaskType.CAUSAL_LM,
)

# 4. Apply LoRA adapters to the quantized model
# print("\n--- Applying LoRA Adapters to quantized model ---")
peft_model = get_peft_model(model, lora_config)

for param in peft_model.base_model.parameters():
    if hasattr(param, 'requires_grad'):
        param.requires_grad = True
# print("\n--- LoRA Adapters Applied ---") 
# peft_model.print_trainable_parameters()
apply_quantization_to_peft_model(peft_model)

# First apply quantization to the base model
# print("\n--- Applying W4A4 quantization to base model ---")

# apply_quantization_to_peft_model(peft_model)
# print("\n--- W4A4 Quantization Applied to LoRA-enabled model ---")





# 5. Configure Trainable Parameters
# Ensure only LoRA adapters and PACT thresholds are trainable.
print("\n--- Configuring Trainable Parameters (LoRA, PACT Clip_Val) ---")
for name, param in peft_model.named_parameters():
    # LoRA parameters are automatically set to requires_grad=True by peft.
    # We only need to ensure the PACT clip_val are trainable,
    # and other base model parameters are frozen.

    # Enable gradients for LoRA, PACT, and SmoothQuant parameters
    if "activation_clip" in name or "smooth_factor" in name:
        param.requires_grad = True
    elif "lora" in name: # LoRA parameters
        param.requires_grad = True # PEFT sets this, but explicit for clarity
    else:
        param.requires_grad = False

# Verify trainable parameters
print("Trainable parameters after configuration:")
for name, param in peft_model.named_parameters():
    if param.requires_grad:
        print(f"- {name}")

# The final step will be setting up the training loop.

# --- Start of Training Loop and Evaluation (Step 6, 7, 8) ---
def find_available_dir(base_dir):
    """Find the next available directory with an index to prevent overwrites."""
    index = 1
    while True:
        dir_name = f"{base_dir}_{index}" if index > 1 else base_dir
        if not os.path.exists(dir_name):
            return dir_name
        index += 1

def setup_tensorboard_logging(output_dir, model):
    """Setup TensorBoard logging with model architecture and quantization info"""
    log_dir = os.path.join(output_dir, "tensorboard_logs")
    writer = SummaryWriter(log_dir=log_dir)
    
    # Log model architecture
    dummy_input = torch.randint(0, 1000, (1, 512), device=next(model.parameters()).device)
    try:
        writer.add_graph(model, dummy_input)
    except Exception as e:
        print(f"Could not log model graph: {e}")
    
    # Count quantized layers
    qat_layers = 0
    lora_params = 0
    total_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, QATW4A4Linear):
            qat_layers += 1
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'lora' in name.lower():
            lora_params += param.numel()
    
    # Log model info as text
    model_info = f"""
    Model Configuration:
    - Base Model: GPT-2
    - Quantized Layers: {qat_layers}
    - Total Parameters: {total_params:,}
    - LoRA Parameters: {lora_params:,}
    - Quantization: W4A4 (4-bit weights, 4-bit activations)
    - LoRA Rank: 8
    - LoRA Alpha: 32
    """
    writer.add_text("Model/Configuration", model_info, 0)
    
    return writer


class TensorBoardLoggingCallback:
    """Custom callback for detailed TensorBoard logging during training"""
    
    def __init__(self, writer, model):
        self.writer = writer
        self.model = model
        self.step = 0
        
    def on_log(self, logs):
        """Called when trainer logs metrics"""
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"Training/{key}", value, self.step)
        self.step += 1
    
    def on_evaluate(self, eval_logs):
        """Called after evaluation"""
        for key, value in eval_logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"Evaluation/{key}", value, self.step)
        
        # Calculate and log perplexity
        if 'eval_loss' in eval_logs:
            perplexity = torch.exp(torch.tensor(eval_logs['eval_loss']))
            self.writer.add_scalar("Evaluation/perplexity", perplexity, self.step)
    
    # REPLACE the log_quantization_params method in TensorBoardLoggingCallback with:
    def log_quantization_params(self, epoch):
        """Log quantization parameters for analysis and debugging."""
        for name, module in self.model.named_modules():
            if isinstance(module, QATW4A4Linear):
                # Log PACT activation clipping values
                clip_val = module.constrained_clip_val.item()
                self.writer.add_scalar(f"Quantization/{name}/activation_clip", clip_val, epoch)
                
                # Log SmoothQuant scaling factor statistics
                if hasattr(module, 'smooth_factor'):
                    self.writer.add_histogram(f"Quantization/{name}/smooth_factor", 
                                              module.smooth_factor.detach().cpu(), epoch)
                    self.writer.add_scalar(f"Quantization/{name}/smooth_factor_mean", 
                                           module.smooth_factor.detach().mean().item(), epoch)
                
                # Log input statistics
                if hasattr(module, 'input_min') and hasattr(module, 'input_max'):
                    self.writer.add_histogram(f"Quantization/{name}/input_min", 
                                              module.input_min.detach().cpu(), epoch)
                    self.writer.add_histogram(f"Quantization/{name}/input_max", 
                                              module.input_max.detach().cpu(), epoch)
                
                # Log quantization scales and zero points
                if hasattr(module, 'weight_scale') and module.weight_scale.numel() > 0:
                    self.writer.add_histogram(f"Quantization/{name}/weight_scales", 
                                            module.weight_scale, epoch)
                    self.writer.add_scalar(f"Quantization/{name}/weight_scale_mean", 
                                        module.weight_scale.mean().item(), epoch)
                    
                if hasattr(module, 'weight_zero_point') and module.weight_zero_point.numel() > 0:
                    self.writer.add_histogram(f"Quantization/{name}/weight_zero_points", 
                                            module.weight_zero_point, epoch)
                    
                if hasattr(module, 'act_scale') and module.act_scale.numel() > 0:
                    self.writer.add_histogram(f"Quantization/{name}/act_scale", 
                                              module.act_scale.detach().cpu(), epoch)
                    self.writer.add_scalar(f"Quantization/{name}/act_scale_mean", 
                                           module.act_scale.detach().mean().item(), epoch)
                    
                if hasattr(module, 'act_zero_point') and module.act_zero_point.numel() > 0:
                    self.writer.add_histogram(f"Quantization/{name}/act_zero_point", 
                                              module.act_zero_point.detach().cpu(), epoch)


if __name__ == "__main__":
    # 6. Setup and Run Training Loop
    print("\n--- Setting up Training Loop for W4A4 QAT ---")
    
    # Create unique output directories
    base_output_dir = "./output_qat_lora"
    base_model_dir = "./qat_lora_model"
    
    output_dir = find_available_dir(base_output_dir)
    model_dir = find_available_dir(base_model_dir)
    
    print(f"Using output directory: {output_dir}")
    print(f"Model will be saved to: {model_dir}")

    # Load and preprocess the WikiText-2 dataset
    datasets, _ = load_and_preprocess_wikitext2(model_name="gpt2") # Use _ for tokenizer as it\'s already loaded
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]

    # Set up Training Arguments
    use_fp16 = torch.cuda.is_available() and not torch.backends.mps.is_available()
    
    # Choose memory-friendly settings based on device
    is_cuda = torch.cuda.is_available() and not torch.backends.mps.is_available()
    per_device_bs = 4 if is_cuda else 8
    grad_accum = 16 if is_cuda else 8

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        num_train_epochs=10,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs * 2,
        gradient_accumulation_steps=grad_accum,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_dir=os.path.join(output_dir, "tensorboard_logs"),
        logging_steps=20,
        learning_rate=5e-5,  # Reduced learning rate for stability
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=is_cuda,
        bf16=False,
        tf32=False,
        report_to="tensorboard",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )

    # Define Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Build optimizer with parameter groups: LoRA vs PACT clip vs SmoothQuant factors
    lora_params: List[torch.nn.Parameter] = []
    clip_params: List[torch.nn.Parameter] = []
    smooth_params: List[torch.nn.Parameter] = []

    for name, param in peft_model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if "lora" in lname:
            lora_params.append(param)
        elif "activation_clip" in lname:
            clip_params.append(param)
        elif "smooth_factor" in lname:
            smooth_params.append(param)

    param_groups = []
    # Set learning rates: LoRA uses the TrainingArguments LR; quant params use a higher LR
    lora_lr = training_args.learning_rate
    quant_lr = 1e-3
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lora_lr, "weight_decay": 0.0})
    if clip_params:
        param_groups.append({"params": clip_params, "lr": quant_lr, "weight_decay": 0.0})
    if smooth_params:
        param_groups.append({"params": smooth_params, "lr": quant_lr, "weight_decay": 0.0})

    # Fallback: if for some reason groups are empty, include all trainable params
    if not param_groups:
        all_trainable = [p for p in peft_model.parameters() if p.requires_grad]
        param_groups = [{"params": all_trainable, "lr": lora_lr, "weight_decay": 0.0}]

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    # Initialize Trainer with custom optimizer (Trainer will create scheduler)
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None),
    )
    
    # Save quantization params inside each checkpoint and enable seamless resume
    class SaveQuantParamsCallback:
        def on_save(self, args, state, control, model=None, **kwargs):
            try:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                if os.path.isdir(ckpt_dir):
                    quant_params = {}
                    for name, module in model.named_modules():
                        if isinstance(module, QATW4A4Linear):
                            prefix = f"{name}."
                            quant_params[f"{prefix}activation_clip"] = module.activation_clip.data.detach().cpu()
                            if hasattr(module, 'smooth_factor'):
                                quant_params[f"{prefix}smooth_factor"] = module.smooth_factor.data.detach().cpu()
                            if hasattr(module, 'input_min') and hasattr(module, 'input_max'):
                                quant_params[f"{prefix}input_min"] = module.input_min.data.detach().cpu()
                                quant_params[f"{prefix}input_max"] = module.input_max.data.detach().cpu()
                            quant_params[f"{prefix}weight_scale"] = module.weight_scale.data.detach().cpu()
                            quant_params[f"{prefix}weight_zero_point"] = module.weight_zero_point.data.detach().cpu()
                            quant_params[f"{prefix}act_scale"] = module.act_scale.data.detach().cpu()
                            quant_params[f"{prefix}act_zero_point"] = module.act_zero_point.data.detach().cpu()
                    torch.save(quant_params, os.path.join(ckpt_dir, "quantization_params.pth"))
            except Exception as e:
                print(f"Warning: failed to save quant params at checkpoint: {e}")

    trainer.add_callback(SaveQuantParamsCallback())
    
    # Ensure model on device and enable input grads for all devices to allow LoRA/SQ to learn
    peft_model = peft_model.to(device)
    peft_model.enable_input_require_grads()
    # Disable KV cache to avoid conflicts with potential checkpointing and ensure full backprop
    if hasattr(peft_model, 'config'):
        peft_model.config.use_cache = False

    # 7. Train the model (fresh run). Checkpoints will save quant params via callback.
    print("\n--- Starting W4A4 QAT with LoRA Fine-tuning ---")
    trainer.train()

    # 8. Evaluate and Save
    print("\n--- Evaluating W4A4 QAT LoRA Model ---")
    eval_results = trainer.evaluate()
    print(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss']))}")

    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model, tokenizer, and quantization parameters
    print(f"\n--- Saving W4A4 QAT LoRA model to {model_dir} ---")
    
    # Save quantization parameters
    def save_quantization_parameters(model, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        quant_params = {}
        i = 0
        for name, module in model.named_modules():
            i += 1
            print(i)
            if isinstance(module, QATW4A4Linear):
                # Save all quantization parameters
                param_prefix = f"{name}."
                quant_params[f"{param_prefix}activation_clip"] = module.activation_clip.data
                quant_params[f"{param_prefix}smooth_factor"] = module.smooth_factor.data
                quant_params[f"{param_prefix}input_min"] = module.input_min.data
                quant_params[f"{param_prefix}input_max"] = module.input_max.data
                quant_params[f"{param_prefix}weight_scale"] = module.weight_scale.data
                quant_params[f"{param_prefix}weight_zero_point"] = module.weight_zero_point.data
                quant_params[f"{param_prefix}act_scale"] = module.act_scale.data
                quant_params[f"{param_prefix}act_zero_point"] = module.act_zero_point.data
        
        # Save quantization parameters
        torch.save(quant_params, os.path.join(save_dir, "quantization_params.pth"))
        print(f"Saved {len(quant_params)} quantization parameters")
    
    # Save quantization parameters after training
    save_quantization_parameters(peft_model, model_dir)
    
    # Save the LoRA adapters
    peft_model.save_pretrained(model_dir)
    print(f"--- Saving W4A4 QAT LoRA adapters to {model_dir} ---")
    
    # Save the tokenizer
    tokenizer.save_pretrained(model_dir)
    
    # Save the model config
    peft_model.config.to_json_file(os.path.join(model_dir, "config.json"))
    
    # Save quantization parameters
    quant_config = {
        "quantization_config": {
            "bits": 4,
            "smoothquant_alpha": 0.5,
            "activation_clip_init": 6.0,
            "momentum": 0.1
        }
    }
    with open(os.path.join(model_dir, "quant_config.json"), 'w') as f:
        json.dump(quant_config, f, indent=2)
    
    print(f"\n--- Model, tokenizer, and quantization config saved to {model_dir} ---")
    print("To load this model for inference, use:")
    print(f"model = AutoModelForCausalLM.from_pretrained('{model_name}').to('{device}')")
    print(f"model = PeftModel.from_pretrained(model, '{model_dir}')")
    print("# Make sure to apply the same quantization setup when loading")