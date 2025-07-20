from transformers import AutoModelForCausalLM, AutoTokenizer
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

class W4A4Linear(nn.Module):
    def __init__(self, original_linear_layer, num_bits=4, init_clip_val=6.0):
        super().__init__()
        self.original_linear = original_linear_layer
        self.num_bits = num_bits
        self.fake_quantize = FakeQuantize.apply
        
        # For activations: use PACT-style learnable clipping (symmetric around zero)
        self.activation_clip = nn.Parameter(torch.tensor(init_clip_val, dtype=torch.bfloat16))
        
        # Buffers for scale and zero point (computed in forward)
        self.register_buffer('weight_scale', torch.ones(1, dtype=torch.bfloat16))
        self.register_buffer('weight_zero_point', torch.zeros(1, dtype=torch.bfloat16))
        self.register_buffer('act_scale', torch.ones(1, dtype=torch.bfloat16))
        self.register_buffer('act_zero_point', torch.zeros(1, dtype=torch.bfloat16))

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
        # Get base weights
        base_weight = self.original_linear.weight
        
        # Check if this is a LoRA layer
        if hasattr(self.original_linear, 'lora_A') and hasattr(self.original_linear, 'lora_B'):
            # Calculate LoRA delta: lora_B @ lora_A * scaling
            lora_A = self.original_linear.lora_A.default.weight
            lora_B = self.original_linear.lora_B.default.weight
            lora_scaling = self.original_linear.scaling['default']
            
            # Apply LoRA scaling and calculate delta
            lora_delta = lora_B @ lora_A * lora_scaling
            
            # Ensure lora_delta has the same shape as base_weight
            if lora_delta.shape != base_weight.shape:
                # This handles the case where LoRA is applied to a subset of the weight matrix
                # We assume the LoRA delta is applied to the first part of the weight matrix
                delta = torch.zeros_like(base_weight)
                delta[:lora_delta.shape[0], :lora_delta.shape[1]] = lora_delta
                lora_delta = delta
                
            # Merge base weights with LoRA delta
            return base_weight + lora_delta
            
        return base_weight
        
    def forward(self, input):
        # Get merged weights (base + LoRA delta if applicable)
        merged_weights = self._get_merged_weights()
        
        # Per-channel quantization for weights (per output channel)
        # Calculate min/max for each output channel [out_features, 1]
        weight_mins = merged_weights.min(dim=1, keepdim=True)[0]
        weight_maxs = merged_weights.max(dim=1, keepdim=True)[0]
        
        # Calculate per-channel quantization parameters
        weight_scale, weight_zero_point = self._calculate_quant_params(
            weight_mins, weight_maxs, self.num_bits
        )
        
        # PACT: symmetric clipping around zero for activations
        clip_val = torch.abs(self.activation_clip)
        clipped_input = torch.clamp(input, -clip_val, clip_val)
        
        # Calculate activation scale and zero point based on clipped input
        act_scale, act_zero_point = self._calculate_quant_params(
            -clip_val, clip_val, self.num_bits
        )
        
        # Store scale and zero point for potential use in other methods
        self.weight_scale.copy_(weight_scale.detach())
        self.weight_zero_point.copy_(weight_zero_point.detach())
        self.act_scale.copy_(act_scale.detach())
        self.act_zero_point.copy_(act_zero_point.detach())
        
        # Apply fake quantization to activations and weights
        quant_input = self.fake_quantize(clipped_input, act_scale, act_zero_point, self.num_bits)
        quant_weight = self.fake_quantize(
            merged_weights,
            weight_scale, 
            weight_zero_point, 
            self.num_bits
        )

        # Perform the linear operation with quantized input and weights
        # Note: Bias is not quantized, it's usually kept in full precision.
        # IMPORTANT: GPT-2 uses Conv1D, where weights are (in_features, out_features).
        # F.linear expects weights as (out_features, in_features) and applies transpose internally (input @ weight.T).
        # So, we need to transpose quant_weight before passing it to F.linear to match Conv1D behavior.
        output = F.linear(quant_input, quant_weight.T, self.original_linear.bias)
        return output

# Custom Quantization-Aware GPT2Block
class QuantizedGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        # Replace original Linear layers with W4A4Linear
        self.attn.c_attn = W4A4Linear(self.attn.c_attn)
        self.attn.c_proj = W4A4Linear(self.attn.c_proj)
        self.mlp.c_fc = W4A4Linear(self.mlp.c_fc)
        self.mlp.c_proj = W4A4Linear(self.mlp.c_proj)


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
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # Use bfloat16 for better precision
    device_map="auto"
)

# Add a pad token to the tokenizer if it doesn't have one
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

print(f"Model '{model_name}' loaded successfully for QAT.")
print(f"Model device: {model.device}")
print(f"Number of parameters: {model.num_parameters()}")

# 2. Define LoRA configuration for QAT
lora_config = LoraConfig(
    r=8, # LoRA attention dimension
    lora_alpha=16, # Alpha parameter for LoRA scaling
    target_modules=["c_attn", "c_proj", "c_fc"], # Common target modules for GPT-2 LoRA
    lora_dropout=0.1, # Dropout probability for LoRA layers
    bias="none", # No bias in LoRA layers
    task_type=TaskType.CAUSAL_LM, # Causal Language Modeling
)

# 3. Get the PEFT model with LoRA adapters
peft_model = get_peft_model(model, lora_config)
print("\n--- LoRA Adapters Applied ---")
peft_model.print_trainable_parameters()

def apply_quantization_to_model(model):
    """Apply W4A4Linear to all linear layers in the model"""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, torch.nn.modules.conv.Conv1D)):
            # Get parent module and attribute name
            parent = model
            attrs = name.split('.')
            for attr in attrs[:-1]:
                parent = getattr(parent, attr)
            
            # Replace with quantized version
            setattr(parent, attrs[-1], W4A4Linear(module))

# Apply quantization to all linear layers in the model
print("\n--- Applying W4A4 quantization to all linear layers ---")
apply_quantization_to_model(peft_model)

# 5. Configure Trainable Parameters
# Ensure only LoRA adapters and PACT thresholds are trainable.
print("\n--- Configuring Trainable Parameters (LoRA, PACT Clip_Val) ---")
for name, param in peft_model.named_parameters():
    # LoRA parameters are automatically set to requires_grad=True by peft.
    # We only need to ensure the PACT clip_val are trainable,
    # and other base model parameters are frozen.

    # Enable gradients for LoRA parameters and activation_clip parameters
    if "lora" in name or "activation_clip" in name:
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
if __name__ == "__main__":
    # 6. Setup and Run Training Loop
    print("\n--- Setting up Training Loop for W4A4 QAT --- ")

    # Load and preprocess the WikiText-2 dataset
    datasets, _ = load_and_preprocess_wikitext2(model_name="gpt2") # Use _ for tokenizer as it\'s already loaded
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]

    # Set up Training Arguments
    training_args = TrainingArguments(
        output_dir="./output_qat_lora", # Output directory for checkpoints and logs
        overwrite_output_dir=True,
        num_train_epochs=1, # Train for one epoch as per plan
        per_device_train_batch_size=8, # Adjust batch size as per your GPU memory
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8, # Simulate larger batch size
        save_strategy="epoch",
        eval_strategy="epoch", # Corrected based on your input
        logging_dir="./logs_qat_lora",
        logging_steps=50, # Log every 50 steps
        learning_rate=2e-4, # Standard learning rate for LoRA fine-tuning
        weight_decay=0.01,
        fp16=torch.cuda.is_available(), # Use FP16 if CUDA is available, for faster training
        report_to="none", # Disable reporting
        remove_unused_columns=False, # Important for DataCollatorForLanguageModeling
        load_best_model_at_end=True, # Load the best model based on evaluation loss
        metric_for_best_model="eval_loss",
    )

    # Define Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize Trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 7. Train the model
    print("\n--- Starting W4A4 QAT with LoRA Fine-tuning ---")
    trainer.train()

    # 8. Evaluate and Save
    print("\n--- Evaluating W4A4 QAT LoRA Model ---")
    eval_results = trainer.evaluate()
    print(f"Perplexity: {torch.exp(torch.tensor(eval_results["eval_loss"]))}")

    # Save the LoRA adapters (and implicitly, the PACT parameters as part of the PEFT model)
    peft_model.save_pretrained("./qat_lora_adapters")
    print("W4A4 QAT LoRA adapters saved to ./qat_lora_adapters") 