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
        
        # Quantize the weights during initialization
        weight = original_linear.weight.detach()
        
        # Calculate scale and zero point for weights
        qmin = 0.
        qmax = 2. ** num_bits - 1.
        min_val = torch.min(weight)
        max_val = torch.max(weight)
        
        if min_val == max_val:
            self.register_buffer('weight_scale', torch.tensor(1.0).to(weight.device))
            self.register_buffer('weight_zero_point', torch.tensor(0).to(weight.device))
        else:
            self.weight_scale = (max_val - min_val) / (qmax - qmin)
            self.weight_zero_point = qmin - min_val / self.weight_scale
            
        # Quantize and store weights as integers
        self.register_buffer('weight_q', torch.clamp(
            torch.round(weight / self.weight_scale + self.weight_zero_point),
            qmin, qmax
        ).to(torch.uint8))
        
        # Store bias if it exists
        if hasattr(original_linear, 'bias') and original_linear.bias is not None:
            self.register_parameter('bias', nn.Parameter(original_linear.bias.detach().clone()))
        else:
            self.register_parameter('bias', None)
            
        # Initialize activation scale and zero point
        self.register_buffer('act_scale', torch.tensor(1.0))
        self.register_buffer('act_zero_point', torch.tensor(0))
        
    def quantize_tensor(self, x, num_bits):
        if num_bits == 32:
            return x, torch.tensor(1.0).to(x.device), torch.tensor(0.0).to(x.device)
            
        qmin = 0.
        qmax = 2. ** num_bits - 1.
        
        # Calculate scale and zero point
        min_val = torch.min(x.detach())
        max_val = torch.max(x.detach())
        
        # Avoid division by zero
        if min_val == max_val:
            return x, torch.tensor(1.0).to(x.device), torch.tensor(0.0).to(x.device)
            
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
               
        # Store quantized integers instead of dequantizing
        x_quantized = torch.clamp(torch.round(x.detach() / scale + zero_point), qmin, qmax)
        
        return x_quantized, scale, zero_point
    
    def forward(self, x):
        # Dequantize weights for computation
        weight = (self.weight_q.float() - self.weight_zero_point) * self.weight_scale
        
        # Linear operation with dequantized weights
        x = torch.nn.functional.linear(x, weight, self.bias)
        
        # Quantize activations
        if self.training:
            # During training, calculate dynamic activation quantization
            x_q, scale_a, zp_a = self.quantize_tensor(x, self.num_bits)
            self.act_scale.copy_(scale_a)
            self.act_zero_point.copy_(zp_a)
            x = (x_q - zp_a) * scale_a
        else:
            # During inference, use stored scale and zero point
            x_q = torch.clamp(
                torch.round(x / self.act_scale + self.act_zero_point),
                0, 2**self.num_bits - 1
            )
            x = (x_q - self.act_zero_point) * self.act_scale
        
        return x

def quantize_model(model, bits=4, depth=0):
    """Apply W4A4 quantization to the model with MPS support"""
    # Handle PEFT model case
    if hasattr(model, 'base_model') and hasattr(model, 'peft_config'):
        print("Warning: PEFT model detected. Quantizing base model only.")
        # For PEFT models, quantize the base model and return the original PEFT wrapper
        quantized_base = quantize_model(model.base_model, bits, depth+1)
        model.base_model = quantized_base
        return model
    
    print(f"{'  ' * depth}Processing model of type: {type(model).__name__}")
    
    # Create a copy of the model
    quantized_model = copy.deepcopy(model)
    quantized_model = quantized_model.to(device)
    
    # Handle base model case - quantize all Linear layers
    for name, module in quantized_model.named_children():
        if isinstance(module, torch.nn.Linear):
            print(f"{'  ' * depth}Quantizing Linear layer: {name}")
            setattr(quantized_model, name, W4A4Linear(module, num_bits=bits))
        else:
            # Recursively quantize child modules
            setattr(
                quantized_model, 
                name, 
                quantize_model(module, bits, depth + 1)
            )
    
    return quantized_model

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
        
        # Quantize the model
        print("\n--- Applying W4A4 quantization to the model on MPS ---")
        quantized_model = quantize_model(model, bits=4)
        quantized_model = quantized_model.to(device)
        
        # Evaluate quantized model
        print("\n--- Evaluating Quantized Model ---")
        quantized_eval_results = evaluate_model(quantized_model, eval_dataset, tokenizer)
        
        # Print comparison
        print("\n--- Results Comparison ---")
        print(f"Original Model - Loss: {eval_results['eval_loss']:.4f}, Perplexity: {eval_results['perplexity']:.2f}")
        print(f"Quantized Model - Loss: {quantized_eval_results['eval_loss']:.4f}, Perplexity: {quantized_eval_results['perplexity']:.2f}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
