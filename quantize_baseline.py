import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data_preparation import load_and_preprocess_wikitext2

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
            self.register_buffer('weight_scale', torch.tensor(1.0).to(weight.device))
        else:
            # Scale is range / (2^num_bits - 1)
            self.weight_scale = (2 * max_abs) / (2 ** self.num_bits - 1)
        
        # Quantize weights to 4-bit range [-8, 7] and store as uint8 [0, 15]
        weight_quantized = torch.clamp(
            torch.round(weight / self.weight_scale),
            qmin, qmax
        )
        # Shift to unsigned range [0, 15] for storage
        self.register_buffer('weight_q', (weight_quantized - qmin).to(torch.uint8))
        
        # Store bias if it exists
        if hasattr(original_linear, 'bias') and original_linear.bias is not None:
            self.register_parameter('bias', nn.Parameter(original_linear.bias.detach().clone()))
        else:
            self.register_parameter('bias', None)
            
        # Initialize activation quantization parameters
        self.register_buffer('act_scale', torch.tensor(1.0))
        self.register_buffer('act_zero_point', torch.tensor(0.0))
        
    @property
    def weight(self):
        # Return dequantized weights for PEFT compatibility
        return (self.weight_q.float() - self.weight_zero_point) * self.weight_scale
        
    def fake_quantize_activations(self, x):
        """Apply fake quantization to activations (4-bit)"""
        if not self.training and not hasattr(self, 'act_scale'):
            return x
            
        # For activations, we'll use asymmetric quantization (0 to 15 for 4-bit)
        qmin = 0.
        qmax = 2. ** self.num_bits - 1.  # 15 for 4-bit
        
        if self.training:
            # During training, calculate scale and zero point from data
            # Using a running min/max would be better, but this is simpler for now
            min_val = torch.min(x.detach())
            max_val = torch.max(x.detach())
            
            # Avoid division by zero
            if min_val >= max_val:
                return x
                
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = torch.round(qmin - min_val / scale).clamp(qmin, qmax)
            
            # Store scale and zero point for inference
            self.act_scale = scale.detach()
            self.act_zero_point = zero_point.detach()
        else:
            # During inference, use stored scale and zero point
            scale = self.act_scale
            zero_point = self.act_zero_point
        
        # Quantize and dequantize (fake quantization) - simulate 4-bit precision
        x_q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
        x_fake_quant = (x_q - zero_point) * scale
        
        return x_fake_quant
    
    def forward(self, x):
        # Dequantize weights from 4-bit storage
        # Convert from [0,15] to [-8,7] range and apply scale
        qmin = -(2 ** (self.num_bits - 1))  # -8 for 4-bit
        
        # Correct dequantization: (q - zero_point) * scale
        # Where zero_point is -qmin (to shift from [0,15] to [-8,7])
        weight = (self.weight_q.float() - (-qmin)) * self.weight_scale
        
        # Apply fake quantization to input activations (4-bit)
        x_quant = self.fake_quantize_activations(x)
        
        # Linear operation with quantized weights and activations
        output = torch.nn.functional.linear(x_quant, weight, self.bias)
        
        # Apply fake quantization to output activations (4-bit)
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
    device = next(model.parameters()).device
    
    # Use a small subset for evaluation
    eval_subset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))
    
    with torch.no_grad():
        for i, example in enumerate(eval_subset):
            # Convert to tensor and move to device
            inputs = {
                'input_ids': torch.tensor([example['input_ids']], device=device),
                'attention_mask': torch.tensor([example['attention_mask']], device=device),
                'labels': torch.tensor([example['labels']], device=device)
            }
            
            # Forward pass
            outputs = model(**inputs)
            total_loss += outputs.loss.item()
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(eval_subset)} examples...")
    
    avg_loss = total_loss / len(eval_subset)
    perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"Evaluation Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    return {"eval_loss": avg_loss, "perplexity": perplexity.item()}

def load_model_and_tokenizer(model_name="gpt2", lora_adapters_path=None):
    """Load the base model, tokenizer, and optionally LoRA adapters."""
    print("Loading base model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    # Load LoRA adapters if provided
    if lora_adapters_path:
        print("Loading LoRA adapters...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model,
            lora_adapters_path,
            torch_dtype=torch.float32,
            device_map="auto"
        )
    
    return model, tokenizer

def load_and_preprocess_wikitext2(model_name="gpt2"):
    """Load and preprocess the WikiText-2 dataset."""
    print("Loading and preprocessing data...")
    from datasets import load_dataset, DatasetDict
    
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
    from transformers import AutoTokenizer
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
            return_tensors="pt"
        )
        
        # Set labels same as input_ids for language modeling
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Apply tokenization to each split
    tokenized_datasets = {}
    for split in filtered_dataset:
        tokenized_datasets[split] = filtered_dataset[split].map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )
    
    # Convert to DatasetDict
    tokenized_dataset_dict = DatasetDict(tokenized_datasets)
    
    return tokenized_dataset_dict, tokenizer

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
        if isinstance(dataset, (list, tuple)):
            # If dataset is a tuple/list, take the second element as validation
            eval_dataset = dataset[1]  # Assuming (train, validation, test) format
        else:
            # If it's a DatasetDict, access by key
            eval_dataset = dataset["validation"]
        
        # Load model and tokenizer
        print("\nLoading model...")
        model, _ = load_model_and_tokenizer(model_name, lora_adapters_path)
        model = model.to(device)
        
        # Evaluate original model
        print("\n--- Evaluating Original Model ---")
        eval_results = evaluate_model(model, eval_dataset, tokenizer)
        
        # Create a copy for quantization to preserve original
        print("\n--- Creating copy of model for quantization ---")
        quantized_model = copy.deepcopy(model)
        
        # Quantize the copied model
        print("\n--- Applying W4A4 quantization to the model ---")
        quantize_model(quantized_model, bits=4)
        quantized_model = quantized_model.to(device)
        
        # Evaluate quantized model
        print("\n--- Evaluating Quantized Model ---")
        quantized_eval_results = evaluate_model(quantized_model, eval_dataset, tokenizer)
        
        # Print comparison
        print("\n--- Results Comparison ---")
        print(f"Original Model - Loss: {eval_results['eval_loss']:.4f}, Perplexity: {eval_results['perplexity']:.2f}")
        print(f"Quantized Model - Loss: {quantized_eval_results['eval_loss']:.4f}, Perplexity: {quantized_eval_results['perplexity']:.2f}")
        
        # Calculate degradation
        perplexity_increase = quantized_eval_results['perplexity'] - eval_results['perplexity']
        perplexity_increase_pct = (perplexity_increase / eval_results['perplexity']) * 100
        print(f"Perplexity increase: {perplexity_increase:.2f} ({perplexity_increase_pct:.1f}%)")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()