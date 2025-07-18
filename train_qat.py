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
        # Clamp values to the quantization range
        qmin = 0.
        qmax = 2.**num_bits - 1.
        
        # Ensure scale and zero_point are broadcastable
        scale_b = scale.view(x.shape[0], *([1] * (x.ndim - 1))) if scale.ndim > 1 and x.ndim > 1 else scale
        zero_point_b = zero_point.view(x.shape[0], *([1] * (x.ndim - 1))) if zero_point.ndim > 1 and x.ndim > 1 else zero_point

        # Quantize
        x_quant = torch.round(x / scale_b + zero_point_b)
        x_quant.clamp_(qmin, qmax)

        # Dequantize
        x_dequant = (x_quant - zero_point_b) * scale_b
        return x_dequant

    @staticmethod
    def backward(ctx, grad_output):
        # STE: simply pass the gradients through
        return grad_output, None, None, None

class W4A4Linear(nn.Module):
    def __init__(self, original_linear_layer, num_bits=4):
        super().__init__()
        self.original_linear = original_linear_layer
        self.num_bits = num_bits
        self.fake_quantize = FakeQuantize.apply

        # For simplicity in this fast-fail prototype, we'll use simple per-tensor
        # scale and zero_point, which will be estimated or set initially.
        # In a full implementation, you'd calculate these based on min/max of weights/activations.
        # These are placeholders and will need proper initialization or learning.
        self.register_buffer('weight_scale', torch.tensor(0.1, dtype=torch.bfloat16))
        self.register_buffer('weight_zero_point', torch.tensor(0., dtype=torch.bfloat16))
        self.register_buffer('act_scale', torch.tensor(0.1, dtype=torch.bfloat16))
        self.register_buffer('act_zero_point', torch.tensor(0., dtype=torch.bfloat16))

    def forward(self, input):
        # Apply fake quantization to input (activation)
        quant_input = self.fake_quantize(input, self.act_scale, self.act_zero_point, self.num_bits)

        # Apply fake quantization to weights (assuming original_linear.weight is the weight)
        # For weights, the scale and zero_point would typically be determined once after training or during QAT initialization.
        # Here, we're applying it in forward pass for simplicity, assuming they are fixed/pre-calculated.
        quant_weight = self.fake_quantize(self.original_linear.weight, self.weight_scale, self.weight_zero_point, self.num_bits)

        # Perform the linear operation with quantized input and weights
        # Note: Bias is not quantized, it's usually kept in full precision.
        output = F.linear(quant_input, quant_weight, self.original_linear.bias)
        return output

# Define the PACT activation clamp
class PACT(nn.Module):
    def __init__(self, init_clip_val=1.0):
        super().__init__()
        # The clipping threshold is a learnable parameter
        self.clip_val = nn.Parameter(torch.tensor(init_clip_val, dtype=torch.bfloat16))

    def forward(self, x):
        # Clamp x between 0 and clip_val, assuming ReLU-like activations, or symmetric otherwise.
        # Let's assume non-negative for now for simplicity, as it's common for activations that follow a ReLU-like function.
        x = torch.where(x < 0, torch.zeros_like(x), x) # Simple ReLU if not already ReLU
        return torch.min(x, self.clip_val.abs())

# Define SmoothQuant Scaler
class SmoothQuantScaler(nn.Module):
    def __init__(self, init_alpha=0.5):
        super().__init__()
        # Alpha is a learnable parameter, initialized between 0 and 1
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.bfloat16))

    def forward(self, x):
        # This module itself doesn't transform x directly in isolation. 
        # The actual smoothing logic (scaling weights and activations) will be handled in the QuantizedGPT2Block
        # by combining this alpha with the linear layer's operations.
        return x

# Custom Quantization-Aware GPT2Block
class QuantizedGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        # Initialize SmoothQuantScaler for this block
        self.smooth_quant_scaler_attn = SmoothQuantScaler()
        self.smooth_quant_scaler_mlp = SmoothQuantScaler()

        # Replace original Linear layers with W4A4Linear
        self.attn.c_attn = W4A4Linear(self.attn.c_attn)
        self.attn.c_proj = W4A4Linear(self.attn.c_proj)
        self.mlp.c_fc = W4A4Linear(self.mlp.c_fc)
        self.mlp.c_proj = W4A4Linear(self.mlp.c_proj)

        # Initialize PACT for attention and MLP outputs
        self.pact_attn = PACT()
        self.pact_mlp = PACT()

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs, # Capture all other arguments
    ):
        # Explicitly remove encoder_hidden_states and encoder_attention_mask from kwargs
        # as they are passed by GPT2Model but are not used by a causal LM block and can cause TypeError.
        encoder_hidden_states = kwargs.pop("encoder_hidden_states", None)
        encoder_attention_mask = kwargs.pop("encoder_attention_mask", None)

        # SmoothQuant application: Scale activations (hidden_states) before attention and MLP
        # and inversely scale the weights inside W4A4Linear or here if it was done outside.
        # For simplicity in this prototype, we'll apply the alpha to activations here.
        # A more rigorous SmoothQuant would also adjust weights, but W4A4Linear already handles weight quantization.

        # Attention part
        # Apply SmoothQuant scaling to input of attention (Q, K, V are derived from hidden_states by c_attn)
        # The `smooth_quant_scaler_attn.alpha` will be learned.
        # This `smooth_quant_scaler_attn(hidden_states)` call is primarily for registering alpha as trainable.
        # For a simple prototype, we'll train alpha as a parameter that directly scales the activation input to linear layers.
        
        # Apply alpha for SmoothQuant to attention input (hidden_states)
        attn_input_smoothed = hidden_states * self.smooth_quant_scaler_attn.alpha

        # Original attention forward pass, but with our W4A4Linear and PACT applied inside the block.
        # We need to call the parent's forward method to ensure all sub-modules are correctly used.
        # The key is that `self.attn.c_attn` etc are now `W4A4Linear` instances.
        # GPT2Attention expects `hidden_states`, `layer_past`, `attention_mask`, etc.
        attn_outputs = self.attn(
            attn_input_smoothed, # Pass the smoothed input to attention
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs, # Pass remaining kwargs to attention module
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        # Apply PACT after attention output
        attn_output = self.pact_attn(attn_output)

        # Residual connection and LayerNorm
        hidden_states = attn_output + hidden_states # Residual connection for attention
        hidden_states = self.ln_1(hidden_states) # LayerNorm

        # MLP part
        # Apply SmoothQuant scaling to input of MLP
        mlp_input_smoothed = self.smooth_quant_scaler_mlp(hidden_states)

        mlp_output = self.mlp(mlp_input_smoothed) # MLP forward pass with W4A4Linear inside

        # Apply PACT after MLP output
        mlp_output = self.pact_mlp(mlp_output)

        # Residual connection and LayerNorm
        hidden_states = mlp_output + hidden_states # Residual connection for MLP
        hidden_states = self.ln_2(hidden_states) # LayerNorm

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)

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

# 4. Replace original GPT2Blocks with QuantizedGPT2Blocks
print("\n--- Inserting W4A4 Fake-Quantization Hooks, PACT, and SmoothQuant into GPT2Blocks ---")
for i, block in enumerate(peft_model.transformer.h):
    peft_model.transformer.h[i] = QuantizedGPT2Block(peft_model.config, i)

# 5. Configure Trainable Parameters
# Ensure only LoRA adapters, SmoothQuant alphas, and PACT thresholds are trainable.
print("\n--- Configuring Trainable Parameters (LoRA, SmoothQuant Alpha, PACT Clip_Val) ---")
for name, param in peft_model.named_parameters():
    # LoRA parameters are automatically set to requires_grad=True by peft.
    # We only need to ensure the SmoothQuant alpha and PACT clip_val are trainable,
    # and other base model parameters are frozen.

    # SmoothQuant alphas are parameters within QuantizedGPT2Block.smooth_quant_scaler_attn/mlp.alpha
    # PACT clip_vals are parameters within QuantizedGPT2Block.pact_attn/mlp.clip_val
    if "lora" in name or "smooth_quant_scaler" in name or "pact" in name:
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
    datasets, _ = load_and_preprocess_wikitext2(model_name="gpt2") # Use _ for tokenizer as it's already loaded
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

    # Save the LoRA adapters (and implicitly, the SmoothQuant and PACT parameters as part of the PEFT model)
    peft_model.save_pretrained("./qat_lora_adapters")
    print("W4A4 QAT LoRA adapters saved to ./qat_lora_adapters") 