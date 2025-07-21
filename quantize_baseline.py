import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset, DatasetDict

# Set device to MPS if available, otherwise CPU
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

class W4A4Linear(nn.Module):
    def __init__(self, original_linear, num_bits=4, init_clip_val=6.0):
        super().__init__()
        self.num_bits = num_bits
        self.original_linear = original_linear
        
        # Store original weight shape for PEFT compatibility
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # Quantize the weights during initialization
        weight = original_linear.weight.detach()
        
        # Calculate scale for weights (symmetric quantization for 4-bit)
        qmin = -(2 ** (num_bits - 1))  # -8 for 4-bit
        qmax = 2 ** (num_bits - 1) - 1  # 7 for 4-bit
        
        # Use symmetric quantization for weights
        max_abs = torch.max(torch.abs(weight))
        if max_abs == 0:
            weight_scale = torch.tensor(1.0).to(weight.device)
        else:
            # Scale is max_abs / qmax (symmetric around 0)
            weight_scale = max_abs / qmax
        
        self.register_buffer('weight_scale', weight_scale)
        
        # Quantize weights to 4-bit range [-8, 7]
        weight_quantized = torch.clamp(
            torch.round(weight / self.weight_scale),
            qmin, qmax
        )
        
        # Store quantized weights directly as int8 (since we're using symmetric quantization)
        self.register_buffer('weight_q', weight_quantized.to(torch.int8))
        
        # Store bias if it exists
        if hasattr(original_linear, 'bias') and original_linear.bias is not None:
            self.register_parameter('bias', nn.Parameter(original_linear.bias.detach().clone()))
        else:
            self.register_parameter('bias', None)
            
        # Initialize activation quantization parameters with running statistics
        self.register_buffer('act_scale', torch.tensor(1.0))
        self.register_buffer('act_zero_point', torch.tensor(0.0))
        self.register_buffer('act_min_val', torch.tensor(float('inf')))
        self.register_buffer('act_max_val', torch.tensor(float('-inf')))
        self.momentum = 0.1  # For running statistics
        
    @property
    def weight(self):
        # Return dequantized weights for PEFT compatibility
        return self.weight_q.float() * self.weight_scale
        
    def update_activation_stats(self, x):
        """Update running min/max statistics for activations"""
        if self.training:
            with torch.no_grad():
                batch_min = torch.min(x.detach())
                batch_max = torch.max(x.detach())
                
                # Initialize or update running statistics
                if torch.isinf(self.act_min_val):
                    self.act_min_val = batch_min
                    self.act_max_val = batch_max
                else:
                    self.act_min_val = (1 - self.momentum) * self.act_min_val + self.momentum * batch_min
                    self.act_max_val = (1 - self.momentum) * self.act_max_val + self.momentum * batch_max
                
                # Update scale and zero point based on current statistics
                min_val = self.act_min_val
                max_val = self.act_max_val
                
                # Asymmetric quantization (0 to 15 for 4-bit unsigned)
                qmin = 0.0
                qmax = 2.0 ** self.num_bits - 1.0  # 15 for 4-bit
                
                if max_val > min_val:
                    scale = (max_val - min_val) / (qmax - qmin)
                    zero_point = torch.clamp(
                        torch.round(qmin - min_val / scale), 
                        qmin, qmax
                    )
                    
                    self.act_scale = scale
                    self.act_zero_point = zero_point
        
    def fake_quantize_activations(self, x):
        """Apply fake quantization to activations (4-bit)"""
        # Update statistics if in training mode
        self.update_activation_stats(x)
        
        # Skip quantization if scale is not properly initialized
        if self.act_scale <= 0 or torch.isinf(self.act_min_val):
            return x
            
        # Asymmetric quantization (0 to 15 for 4-bit unsigned)
        qmin = 0.0
        qmax = 2.0 ** self.num_bits - 1.0  # 15 for 4-bit
        
        # Quantize and dequantize (fake quantization)
        x_q = torch.clamp(
            torch.round(x / self.act_scale + self.act_zero_point), 
            qmin, qmax
        )
        x_fake_quant = (x_q - self.act_zero_point) * self.act_scale
        
        return x_fake_quant
    
    def forward(self, x):
        # Dequantize weights (symmetric quantization)
        weight = self.weight_q.float() * self.weight_scale
        
        # Apply fake quantization to input activations
        x_quant = self.fake_quantize_activations(x)
        
        # Linear operation with quantized weights and activations
        output = torch.nn.functional.linear(x_quant, weight, self.bias)
        
        # Apply fake quantization to output activations
        output_quant = self.fake_quantize_activations(output)
        
        return output_quant

def quantize_model(model, bits=4, depth=0):
    """Apply W4A4 quantization to the model with MPS support"""
    # Handle PEFT model case
    if hasattr(model, 'base_model') and hasattr(model, 'peft_config'):
        print("Warning: PEFT model detected. Quantizing base model only.")
        # For PEFT models, quantize the base model in place
        quantize_model(model.base_model, bits, depth+1)
        return model
    
    print(f"{'  ' * depth}Processing model of type: {type(model).__name__}")
    
    # Recursively replace all Linear layers in place
    def replace_linear_recursive(module, name=""):
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            if isinstance(child_module, torch.nn.Linear):
                print(f"{'  ' * depth}Quantizing Linear layer: {full_name}")
                # Replace the linear layer with quantized version
                quantized_layer = W4A4Linear(child_module, num_bits=bits)
                setattr(module, child_name, quantized_layer)
            else:
                # Recursively process child modules
                replace_linear_recursive(child_module, full_name)
    
    # Apply quantization in place
    replace_linear_recursive(model)
    return model

def evaluate_model(model, eval_dataset, tokenizer, max_samples=100):
    """Evaluate the model and return perplexity"""
    print("\n--- Evaluating Model ---")
    model.eval()
    total_loss = 0
    total_tokens = 0
    device = next(model.parameters()).device
    
    # Use a small subset for evaluation
    eval_subset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))
    
    with torch.no_grad():
        for i, example in enumerate(eval_subset):
            # Convert to tensor and move to device
            input_ids = torch.tensor(example['input_ids'], device=device).unsqueeze(0)
            attention_mask = torch.tensor(example['attention_mask'], device=device).unsqueeze(0)
            labels = torch.tensor(example['labels'], device=device).unsqueeze(0)
            
            # Only compute loss on actual tokens (not padding)
            valid_tokens = (labels != tokenizer.pad_token_id).sum().item()
            if valid_tokens == 0:
                continue
                
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
            
            try:
                # Forward pass
                outputs = model(**inputs)
                total_loss += outputs.loss.item() * valid_tokens
                total_tokens += valid_tokens
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                continue
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(eval_subset)} examples...")
    
    if total_tokens == 0:
        print("No valid tokens found for evaluation!")
        return {"eval_loss": float('inf'), "perplexity": float('inf')}
        
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"Evaluation Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    return {"eval_loss": avg_loss, "perplexity": perplexity.item()}

def load_model_and_tokenizer(model_name="gpt2", lora_adapters_path=None):
    """Load the base model, tokenizer, and optionally LoRA adapters."""
    print("Loading base model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None  # We'll handle device placement manually
    )
    
    # Load LoRA adapters if provided
    if lora_adapters_path and os.path.exists(lora_adapters_path):
        print("Loading LoRA adapters...")
        model = PeftModel.from_pretrained(
            model,
            lora_adapters_path,
            torch_dtype=torch.float32
        )
    elif lora_adapters_path:
        print(f"Warning: LoRA adapters path '{lora_adapters_path}' not found. Using base model only.")
    
    return model, tokenizer

def load_and_preprocess_wikitext2(model_name="gpt2"):
    """Load and preprocess the WikiText-2 dataset."""
    print("Loading and preprocessing data...")
    
    # Load the dataset
    dataset = load_dataset("wikitext", "wikitext-2-v1")
    
    # Filter out empty lines
    filtered_datasets = {}
    for split in dataset:
        filtered_datasets[split] = dataset[split].filter(
            lambda example: len(example["text"].strip()) > 0
        )
    
    # Create a new DatasetDict with filtered datasets
    filtered_dataset = DatasetDict({
        split: filtered_datasets[split] for split in filtered_datasets
    })
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenization function
    def tokenize_function(examples):
        # Tokenize the texts
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors=None  # Return lists instead of tensors
        )
        
        # Set labels same as input_ids for language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Apply tokenization to each split
    tokenized_datasets = {}
    for split in filtered_dataset:
        print(f"Tokenizing {split} split...")
        tokenized_datasets[split] = filtered_dataset[split].map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )
    
    # Convert to DatasetDict
    tokenized_dataset_dict = DatasetDict(tokenized_datasets)
    
    return tokenized_dataset_dict, tokenizer

def calibrate_quantization(model, calibration_data, tokenizer, num_samples=50):
    """Calibrate the quantization parameters using a few samples"""
    print("\n--- Calibrating quantization parameters ---")
    model.train()  # Set to train mode to update statistics
    device = next(model.parameters()).device
    
    calibration_subset = calibration_data.select(range(min(num_samples, len(calibration_data))))
    
    with torch.no_grad():
        for i, example in enumerate(calibration_subset):
            input_ids = torch.tensor(example['input_ids'], device=device).unsqueeze(0)
            attention_mask = torch.tensor(example['attention_mask'], device=device).unsqueeze(0)
            
            # Forward pass to update activation statistics
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if (i + 1) % 10 == 0:
                print(f"Calibrated on {i + 1}/{len(calibration_subset)} samples...")
    
    model.eval()  # Set back to eval mode
    print("Calibration complete!")

def main():
    try:
        # Configuration
        model_name = "gpt2"
        lora_adapters_path = "baseline_lora_adapters"  # Set to None to skip loading adapters
        
        # Set device
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load and preprocess data first to ensure it works
        print("\nLoading and preprocessing data...")
        dataset, tokenizer = load_and_preprocess_wikitext2(model_name)
        
        # Get the validation dataset
        eval_dataset = dataset["validation"]
        
        # Load model and tokenizer
        print("\nLoading model...")
        model, _ = load_model_and_tokenizer(model_name, lora_adapters_path)
        model = model.to(device)
        
        # Evaluate original model
        print("\n--- Evaluating Original Model ---")
        eval_results = evaluate_model(model, eval_dataset, tokenizer, max_samples=200)
        
        # Create a copy for quantization to preserve original
        print("\n--- Creating copy of model for quantization ---")
        quantized_model = copy.deepcopy(model)
        
        # Quantize the copied model
        print("\n--- Applying W4A4 quantization to the model ---")
        quantize_model(quantized_model, bits=4)
        quantized_model = quantized_model.to(device)
        
        # Calibrate quantization parameters
        print("\n--- Calibrating quantization parameters ---")
        calibrate_quantization(quantized_model, eval_dataset, tokenizer, num_samples=50)
        
        # Evaluate quantized model
        print("\n--- Evaluating Quantized Model ---")
        quantized_eval_results = evaluate_model(quantized_model, eval_dataset, tokenizer, max_samples=200)
        
        # Print comparison
        print("\n" + "="*60)
        print("RESULTS COMPARISON")
        print("="*60)
        print(f"Original Model (Baseline + LoRA):")
        print(f"  Loss: {eval_results['eval_loss']:.4f}")
        print(f"  Perplexity: {eval_results['perplexity']:.2f}")
        print(f"\nQuantized Model (W4A4):")
        print(f"  Loss: {quantized_eval_results['eval_loss']:.4f}")
        print(f"  Perplexity: {quantized_eval_results['perplexity']:.2f}")
        
        # Calculate degradation
        perplexity_increase = quantized_eval_results['perplexity'] - eval_results['perplexity']
        perplexity_increase_pct = (perplexity_increase / eval_results['perplexity']) * 100
        print(f"\nPerplexity Degradation:")
        print(f"  Absolute increase: {perplexity_increase:.2f}")
        print(f"  Relative increase: {perplexity_increase_pct:.1f}%")
        print("="*60)
        
        # Save quantized model if needed
        save_quantized = False  # Set to True if you want to save
        if save_quantized:
            print("\nSaving quantized model...")
            torch.save(quantized_model.state_dict(), "quantized_model_w4a4.pth")
            print("Quantized model saved!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()