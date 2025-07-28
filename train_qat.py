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

# Define the FakeQuantize module with STE
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, num_bits):
        """Per-channel fake quantization with straight-through estimator.
        
        Args:
            x: Input tensor
            scale: Scale tensor, shape [C] or [1] for per-tensor
            zero_point: Zero point tensor, shape [C] or [1] for per-tensor
            num_bits: Number of bits for quantization
        """
        qmin = 0
        qmax = 2 ** num_bits - 1
        
        # Ensure scale and zero_point have the right shape for broadcasting
        if scale.dim() < x.dim():
            # Add singleton dimensions to match x's dimensions starting from the last dimension
            for _ in range(x.dim() - scale.dim()):
                scale = scale.unsqueeze(-1)
                zero_point = zero_point.unsqueeze(-1)
        
        x_quant = torch.round(x / scale + zero_point)
        x_quant = torch.clamp(x_quant, qmin, qmax)
        x_dequant = (x_quant - zero_point) * scale
        
        return x_dequant

    @staticmethod
    def backward(ctx, grad_output):
        # STE: simply pass the gradients through
        return grad_output, None, None, None

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
        
    def _calculate_quant_params(self, min_vals, max_vals, num_bits):
        """Calculate scale and zero-point for quantization, supporting per-channel.
        
        Args:
            min_vals: Tensor of minimum values, shape [C] or [1] for per-tensor
            max_vals: Tensor of maximum values, shape [C] or [1] for per-tensor
            num_bits: Number of bits for quantization
        """
        qmin = 0
        qmax = 2 ** num_bits - 1
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
                self.input_absmax = torch.ones(in_features, device=input.device, dtype=torch.bfloat16)
        
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
        clipped_input = torch.clamp(scaled_input, -clip_val, clip_val)
        # clipped_input = torch.clamp(scaled_input, 0, clip_val)
        
        if not self.training and hasattr(self, '_cached_quant_params'):
            weight_scale, weight_zero_point, act_scale, act_zero_point = self._cached_quant_params
        else:
            # Quantization parameters for weights (per-channel)
            weight_mins = scaled_weights.min(dim=1, keepdim=True)[0]
            weight_maxs = scaled_weights.max(dim=1, keepdim=True)[0]
            weight_scale, weight_zero_point = self._calculate_quant_params(
                weight_mins, weight_maxs, self.num_bits
            )
            
            # Quantization parameters for activations (per-tensor)
            clip_val = self.constrained_clip_val
            act_scale, act_zero_point = self._calculate_quant_params(
                -clip_val, clip_val, self.num_bits
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
        quant_input = self.fake_quantize(
            clipped_input, act_scale, act_zero_point, self.num_bits
        )
        quant_weight = self.fake_quantize(
            scaled_weights, weight_scale, weight_zero_point, self.num_bits
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
    torch.set_default_dtype(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32)
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    torch.set_default_dtype(torch.float32)
    print("Using CPU")

# Load model with appropriate settings
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32 if device.type == "mps" else torch.bfloat16,
    device_map="auto" if device.type != "mps" else None,  # device_map not supported on MPS
).to(device)

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

# First apply quantization to the base model
print("\n--- Applying W4A4 quantization to base model ---")
apply_quantization_to_model(model)

# 3. Prepare the model for LoRA fine-tuning
lora_config = LoraConfig(
    r=8,  # Rank of the update matrices (lower = less parameters)
    lora_alpha=32,  # Alpha parameter for LoRA scaling
    target_modules=["c_attn", "c_proj", "c_fc"],  # Target attention and feed-forward layers
    lora_dropout=0.05,
    bias="none",  # Don't train bias parameters
    task_type=TaskType.CAUSAL_LM,
)

# 4. Apply LoRA adapters to the quantized model
print("\n--- Applying LoRA Adapters to quantized model ---")
peft_model = get_peft_model(model, lora_config)
print("\n--- LoRA Adapters Applied ---")
peft_model.print_trainable_parameters()

# 5. Configure Trainable Parameters
# Ensure only LoRA adapters and PACT thresholds are trainable.
print("\n--- Configuring Trainable Parameters (LoRA, PACT Clip_Val) ---")
for name, param in peft_model.named_parameters():
    # LoRA parameters are automatically set to requires_grad=True by peft.
    # We only need to ensure the PACT clip_val are trainable,
    # and other base model parameters are frozen.

    # Enable gradients for LoRA, PACT, and SmoothQuant parameters
    if "lora" in name or "activation_clip" in name or "alpha" in name:
        param.requires_grad = True
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
        output_dir=output_dir, # Output directory for checkpoints and logs
        overwrite_output_dir=False,  # Disable overwrite since we're using unique directories
        num_train_epochs=1, # Train for one epoch as per plan
        per_device_train_batch_size=1 if torch.backends.mps.is_available() else 4, # Smaller batch for MPS
        per_device_eval_batch_size=2 if torch.backends.mps.is_available() else 8,
        gradient_accumulation_steps=64 if torch.backends.mps.is_available() else 16, # Increase for MPS
        max_grad_norm=0.5,  # Gradient clipping with max norm of 1.0
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_dir="./logs_qat_lora",
        logging_steps=20, # More frequent logging for MPS
        learning_rate=1e-5 if torch.backends.mps.is_available() else 5e-5, # Lower lr for MPS
        weight_decay=0.01,
        warmup_ratio = 0.3,
        lr_scheduler_type="cosine",
        fp16=use_fp16, # Only enable for CUDA
        bf16=not use_fp16 and torch.cuda.is_bf16_supported(),
        tf32=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8, # For Ampere+
        report_to="none", # Disable reporting
        remove_unused_columns=False, # Important for DataCollatorForLanguageModeling
        load_best_model_at_end=True, # Load the best model based on evaluation loss
        metric_for_best_model="eval_loss",
        # MPS-specific optimizations
        dataloader_pin_memory=not torch.backends.mps.is_available(), # Disable pin_memory for MPS
        dataloader_num_workers=0 if torch.backends.mps.is_available() else 4, # No workers for MPS
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
        for name, module in model.named_modules():
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