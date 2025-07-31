import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from data_preparation import load_and_preprocess_wikitext2
from torch.utils.tensorboard import SummaryWriter
import time

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
        """Straight-through estimator for the backward pass.
        
        We use the straight-through estimator, which means we pass the gradient through
        the quantization operation as if it was the identity function.
        """
        x, scale, zero_point = ctx.saved_tensors
        
        # For the input gradient, we use the straight-through estimator
        grad_input = grad_output.clone()
        
        # For scale and zero_point, we don't compute gradients as they are learned separately
        # in the QATW4A4Linear class through the _calculate_quant_params method
        grad_scale = None
        grad_zero_point = None
        grad_num_bits = None
        grad_symmetric = None
        
        return grad_input, grad_scale, grad_zero_point, grad_num_bits, grad_symmetric


class QATW4A4Linear(nn.Module):
    def __init__(self, original_linear_layer, num_bits=4, init_clip_val=2.0):
        super().__init__()
        self.original_linear = original_linear_layer
        self.num_bits = num_bits
        self.fake_quantize = FakeQuantize.apply
        self.in_features = original_linear_layer.in_features
        self.out_features = original_linear_layer.out_features
        
        # Get device from original linear layer
        self.device = next(original_linear_layer.parameters()).device
        
        # PACT activation clipping
        self.activation_clip = nn.Parameter(
            torch.tensor(init_clip_val, dtype=torch.float32, device=self.device)
        )
        
        # SmoothQuant parameters
        self.raw_alpha = nn.Parameter(
            torch.tensor(-1.0, device=self.device)  # Will be passed through sigmoid
        )
        # Track per-channel input statistics
        self.register_buffer('input_absmax', 
            torch.ones(self.in_features, dtype=torch.float32, device=self.device)
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
        # Per-tensor for activations
        self.register_buffer('act_scale', 
            torch.ones(1, dtype=torch.float32, device=self.device)
        )
        self.register_buffer('act_zero_point', 
            torch.zeros(1, dtype=torch.float32, device=self.device)
        )

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
        if symmetric:
            # For symmetric quantization, range is [-2^(n-1), 2^(n-1)-1]
            qmin = -2 ** (num_bits - 1)
            qmax = 2 ** (num_bits - 1) - 1
            
            # For symmetric quantization, we use the maximum absolute value
            max_vals = torch.max(torch.abs(min_vals), torch.abs(max_vals))
            min_vals = -max_vals
            
            # Ensure min_vals and max_vals have at least 2 dimensions for broadcasting
            if min_vals.dim() == 1:
                min_vals = min_vals.unsqueeze(-1)
            if max_vals.dim() == 1:
                max_vals = max_vals.unsqueeze(-1)
                
            scale = (max_vals - min_vals) / (qmax - qmin)
            scale = torch.max(scale, torch.tensor(1e-8, device=scale.device))  # avoid division by zero
            
            # For symmetric quantization, zero point is always 0
            zero_point = torch.zeros_like(scale, dtype=torch.int32)
        else:
            # Original asymmetric quantization
            qmin = 0
            qmax = 2 ** num_bits - 1

            # Ensure min_vals and max_vals are tensors
            if not isinstance(min_vals, torch.Tensor):
                min_vals = torch.tensor(min_vals, device=self.device)
            if not isinstance(max_vals, torch.Tensor):
                max_vals = torch.tensor(max_vals, device=self.device)
                
            min_vals = torch.min(min_vals, torch.zeros_like(min_vals))  # clip to <= 0
            max_vals = torch.max(max_vals, torch.zeros_like(max_vals))  # clip to >= 0
            
            # Ensure min_vals and max_vals have at least 2 dimensions for broadcasting
            if min_vals.dim() == 1:
                min_vals = min_vals.unsqueeze(-1)
            if max_vals.dim() == 1:
                max_vals = max_vals.unsqueeze(-1)
            
            scale = (max_vals - min_vals) / (qmax - qmin)
            scale = torch.max(scale, torch.tensor(1e-8, device=scale.device))  # avoid division by zero
            
            zero_point = qmin - min_vals / scale
            zero_point = torch.clamp(zero_point, qmin, qmax).round()
        
        return scale, zero_point

    def _get_merged_weights(self):
        """Merge base weights with LoRA deltas if they exist."""
        # Cache during eval mode
        if not self.training and hasattr(self, '_cached_merged_weights'):
            return self._cached_merged_weights
            
        base_weight = self.original_linear.weight
        
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
        
    @property
    def alpha(self):
        # Constrain alpha between 0 and 1 using sigmoid
        return torch.sigmoid(self.raw_alpha)

    def update_input_stats(self, x):
        # x shape: [batch_size, seq_len, in_features] or [batch_size, in_features]
        # Calculate per-channel absmax
        if x.dim() == 3:  # [batch, seq_len, in_features]
            current_absmax = x.abs().amax(dim=(0, 1))  # [in_features]
        else:  # [batch, in_features]
            current_absmax = x.abs().amax(dim=0)  # [in_features]
        
        # Update running statistics with EMA
        self.input_absmax.mul_(1 - self.momentum).add_(current_absmax * self.momentum)
        return current_absmax

    def forward(self, input):
        
        if not self.training and hasattr(self, '_cached_scaling_factor'):
            scaling_factor = self._cached_scaling_factor
        else:
            merged_weights = self._get_merged_weights()
            weight_absmax_per_input_channel = merged_weights.abs().amax(dim=0, keepdim=False)
            scaling_factor = (self.input_absmax.detach() ** (self.alpha)) * \
                            (weight_absmax_per_input_channel.detach() ** (1 - self.alpha))
            scaling_factor = scaling_factor.clamp(min=1e-8, max = 1e3)
        
        # Cache for eval mode
        if not self.training:
            self._cached_scaling_factor = scaling_factor

        # input shape: [batch_size, seq_len, in_features] or [batch_size, in_features]
        # Ensure input_absmax is the right shape [in_features]
        if input.dim() == 3:
            in_features = input.size(-1)
            if self.input_absmax.dim() != 1 or self.input_absmax.size(0) != in_features:
                self.input_absmax = torch.ones(in_features, device=input.device, dtype=torch.float32)
        
        # Update input statistics during training
        if self.training:
            self.update_input_stats(input)
        
        # Get merged weights [out_features, in_features]
        merged_weights = self._get_merged_weights()
        
        # Calculate max abs value for each input channel (column-wise max)
        # Shape: [in_features]
        weight_absmax_per_input_channel = merged_weights.abs().amax(dim=0, keepdim=False)
        
        # Calculate per-input-channel scaling factor using SmoothQuant formula:
        # s_j = (|X_j|^(1-alpha)) * (|W_j|^alpha)
        # where |X_j| is input_absmax[j] and |W_j| is weight_absmax_per_input_channel[j]
        scaling_factor = (self.input_absmax.detach() ** (self.alpha)) * \
                        (weight_absmax_per_input_channel.detach() ** (1 - self.alpha))
        scaling_factor = scaling_factor.clamp(min=1e-8, max = 1e3)  # [in_features]
        
        # Apply scaling:
        # Input: x = x / s_j (broadcast over batch and sequence dims)
        scaled_input = input / scaling_factor
        
        # Weights: W = W * s_j (broadcast over output features)
        scaled_weights = merged_weights * scaling_factor.unsqueeze(0)  # [out_features, in_features]
        
        # PACT: symmetric clipping for activations
        # clip_val = torch.abs(self.activation_clip)
        clip_val = self.constrained_clip_val
        clipped_input = torch.clamp(scaled_input, torch.tensor(0, device=scaled_input.device), clip_val)
        # clipped_input = torch.clamp(scaled_input, 0, clip_val)
        
        if not self.training and hasattr(self, '_cached_quant_params'):
            weight_scale, weight_zero_point, act_scale, act_zero_point = self._cached_quant_params
        else:
            # Quantization parameters for weights (per-channel, symmetric)
            weight_mins = scaled_weights.min(dim=1, keepdim=True)[0]
            weight_maxs = scaled_weights.max(dim=1, keepdim=True)[0]
            weight_scale, weight_zero_point = self._calculate_quant_params(
                weight_mins, weight_maxs, self.num_bits, symmetric=True
            )
            
            # Quantization parameters for activations (per-tensor)
            clip_val = self.constrained_clip_val
            act_scale, act_zero_point = self._calculate_quant_params(
                0, clip_val, self.num_bits, symmetric=False
            )
            
            # Cache for eval mode
            if not self.training:
                self._cached_quant_params = (weight_scale, weight_zero_point, act_scale, act_zero_point)
        
        # Store scales for potential use
        # Ensure correct shapes: [out_features] for weights, [1] for activations
        # Using copy_() to maintain buffer status and proper device placement
        self.weight_scale.copy_(weight_scale.detach().flatten())
        self.weight_zero_point.copy_(weight_zero_point.detach().flatten())
        self.act_scale.copy_(act_scale.detach().flatten())
        self.act_zero_point.copy_(act_zero_point.detach().flatten())
        
        # Quantize
        # Using positional arguments (can't use keyword args with Function.apply)
        quant_input = self.fake_quantize(
            clipped_input, act_scale, act_zero_point, self.num_bits, 0  # 0 for asymmetric
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
        if isinstance(module, (nn.Linear, torch.nn.modules.conv.Conv1d)):
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
        if isinstance(module, (nn.Linear, torch.nn.modules.conv.Conv1d)):
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
    if "activation_clip" in name or "raw_alpha" in name:
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
        """Log ALL quantization parameters that are saved in quantization_params.pth"""
        for name, module in self.model.named_modules():
            if isinstance(module, QATW4A4Linear):
                # Log PACT activation clipping values
                clip_val = module.constrained_clip_val.item()
                self.writer.add_scalar(f"Quantization/{name}/activation_clip", clip_val, epoch)
                
                # Log SmoothQuant raw_alpha and computed alpha values
                raw_alpha = module.raw_alpha.item()
                alpha = module.alpha.item()
                self.writer.add_scalar(f"Quantization/{name}/raw_alpha", raw_alpha, epoch)
                self.writer.add_scalar(f"Quantization/{name}/smoothquant_alpha", alpha, epoch)
                
                # Log input statistics (SmoothQuant component)
                if hasattr(module, 'input_absmax') and module.input_absmax.numel() > 0:
                    self.writer.add_histogram(f"Quantization/{name}/input_absmax", 
                                            module.input_absmax, epoch)
                    self.writer.add_scalar(f"Quantization/{name}/input_absmax_mean", 
                                        module.input_absmax.mean().item(), epoch)
                
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
                    self.writer.add_scalar(f"Quantization/{name}/act_scale", 
                                        module.act_scale.item(), epoch)
                    
                if hasattr(module, 'act_zero_point') and module.act_zero_point.numel() > 0:
                    self.writer.add_scalar(f"Quantization/{name}/act_zero_point", 
                                        module.act_zero_point.item(), epoch)


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
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        num_train_epochs=10,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        max_grad_norm=0.5,
        gradient_checkpointing=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_dir=os.path.join(output_dir, "tensorboard_logs"),  # Updated this line
        logging_steps=20,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=False,
        tf32=False,
        report_to="tensorboard",  # Changed from "none" to "tensorboard"
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )

    # Define Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize Trainer with MPS-specific optimizations
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # For MPS, we need to ensure the model is on the correct device
    if torch.backends.mps.is_available():
        peft_model = peft_model.to(device)
        # Add gradient checkpointing for MPS to save memory
        peft_model.enable_input_require_grads()
        peft_model.config.use_cache = False  # Gradient checkpointing requires this

    # 7. Train the model
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
                quant_params[f"{param_prefix}raw_alpha"] = module.raw_alpha.data
                quant_params[f"{param_prefix}input_absmax"] = module.input_absmax.data
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