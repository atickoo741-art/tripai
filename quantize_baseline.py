import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
import random
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig
from datasets import load_dataset, DatasetDict
import copy
from train_qat import QATW4A4Linear
# ============================================
# FakeQuantize with STE (Straight-Through Estimator)
# ============================================
# class FakeQuantize(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, scale, zero_point, qmin, qmax):
#         # Quantize
#         x_int = torch.round(x / scale + zero_point)
#         x_quant = torch.clamp(x_int, qmin, qmax)
#         # Dequantize
#         x_dequant = (x_quant - zero_point) * scale
#         return x_dequant

#     @staticmethod
#     def backward(ctx, grad_output):
#         # Straight-Through Estimator (STE)
#         return grad_output, None, None, None, None

# ============================================
# PACT (PArameterized Clipping acTivation) for activation quantization
# ============================================
# class PACT(torch.nn.Module):
#     def __init__(self, init_val=6.0):
#         super().__init__()
#         self.alpha = torch.nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
#         self.relu = torch.nn.ReLU6()
        
#     def forward(self, x):
#         return self.relu(x) - self.relu(x - self.alpha)

# ============================================
# Uniform W4A4 Linear (for base GPT-2 and baseline LoRA)
# ============================================

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Enable deterministic algorithms if available
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device selection: prioritize MPS (Apple Silicon), then CUDA, then CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: CUDA")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

class W4A4Linear(nn.Module):
    """True 4-bit integer linear layer for inference.
    Uses symmetric quantization for weights and asymmetric quantization for activations.
    """
    def __init__(self, original_linear, num_bits=4):
        super().__init__()
        self.num_bits = num_bits
        
        # Store shapes and original linear reference
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # Quantize weights (symmetric)
        weight = original_linear.weight.detach().float()
        qmin, qmax = -(2 ** (num_bits - 1)), (2 ** (num_bits - 1)) - 1  # -8 to 7 for 4-bit
        
        # Calculate scale and quantize
        max_val = torch.max(torch.abs(weight)).item()
        if max_val == 0:
            max_val = 1.0
            
        weight_scale = max_val / qmax
        
        # Quantize and store weights
        weight_quantized = torch.clamp(
            torch.round(weight / weight_scale),
            qmin, qmax
        ).to(torch.int8)
        
        self.register_buffer('weight_scale', torch.tensor(weight_scale))
        self.register_buffer('weight_q', weight_quantized)
        
        # Store bias if exists
        if hasattr(original_linear, 'bias') and original_linear.bias is not None:
            self.register_parameter('bias', nn.Parameter(original_linear.bias.detach().clone()))
        else:
            self.register_parameter('bias', None)
            
        # Activation scales (calibrated during calibration phase)
        self.register_buffer('input_scale', torch.tensor(1.0))
        self.register_buffer('output_scale', torch.tensor(1.0))
        
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
        # Input is expected in FP32
        
        # Move to CPU if needed for integer operations
        # device = x.device
        # if str(device).startswith('mps'):
        #     x = x.cpu()
        
        # 1. Quantize input to int8 (asymmetric per-tensor quantization)
        input_scale = self.input_scale
        input_zero_point = 0  # Symmetric quantization for input
        
        # Scale and quantize input
        x_int8 = torch.clamp(
            torch.round(x / input_scale) + input_zero_point,
            0, 255
        ).to(torch.uint8)
        
        # 2. Convert to int32 for the matmul
        x_int32 = x_int8.to(torch.int32) - input_zero_point
        
        # 3. Perform integer matrix multiplication
        # weight_q is int8, x_int32 is int32 -> result is int32
        try:
            # Try integer matmul first
            output_int32 = F.linear(x_int32, self.weight_q.to(torch.int32), None)
            # Apply bias and scale
            output_scale = self.weight_scale * input_scale
            output = output_int32.to(torch.float32) * output_scale
        except RuntimeError as e:
            if "does not support linear for non-float inputs" in str(e):
                # Fallback to float32 if integer matmul not supported
                weight_dequant = self.weight_q.to(torch.float32) * self.weight_scale
                output = F.linear(x, weight_dequant, self.bias)
            else:
                raise
        
        if self.bias is not None and 'output' not in locals():
            output = output + self.bias
            
        # 5. Apply output scale (if needed)
        if 'output_scale' in locals():
            output = output * self.output_scale
        
        # Move back to original device
        if str(device).startswith('mps'):
            output = output.to(device)
            
        return output

# class W4A4LinearQAT(nn.Module):
#     """
#     Advanced W4A4 Linear layer with SmoothQuant and PACT for QAT LoRA models.
#     Matches the implementation used during QAT training.
#     """
#     def __init__(self, original_linear, num_bits=4, init_clip_val=6.0):
#         super().__init__()
#         self.num_bits = num_bits
#         self.original_linear = original_linear
        
#         # Store original weight shape for PEFT compatibility
#         self.in_features = original_linear.in_features
#         self.out_features = original_linear.out_features
        
#         # PACT activation clipping
#         self.activation_clip = nn.Parameter(
#             torch.tensor(init_clip_val, dtype=torch.float32)
#         )
        
#         # Quantization parameters for weights
#         self.weight_scale = nn.Parameter(torch.tensor(1.0))
#         self.weight_zero_point = nn.Parameter(torch.tensor(0.0))
        
#         # SmoothQuant parameters
#         self.smooth_quant_alpha = 0.5  # Can be tuned
        
#         # Initialize weights
#         self.register_buffer('weight', original_linear.weight.detach().clone())
#         if hasattr(original_linear, 'bias') and original_linear.bias is not None:
#             self.register_parameter('bias', nn.Parameter(original_linear.bias.detach().clone()))
#         else:
#             self.register_parameter('bias', None)
    
#     def forward(self, x):
#         # PACT activation clipping
#         x = torch.clamp(x, 0.0, self.activation_clip.item())
        
#         # Quantize weights
#         qmin = -(2 ** (self.num_bits - 1))
#         qmax = (2 ** (self.num_bits - 1)) - 1
        
#         # Apply SmoothQuant to weights
#         weight = self.weight
#         weight_scale = self.weight_scale.abs() + 1e-6
#         weight_int = torch.round(weight / weight_scale + self.weight_zero_point)
#         weight_quant = torch.clamp(weight_int, qmin, qmax)
#         weight_dequant = (weight_quant - self.weight_zero_point) * weight_scale
        
#         # Linear operation with quantized weights and activations
#         return F.linear(x, weight_dequant, self.bias)

def quantize_model(model, bits=4, quantization_method='uniform', depth=0):
    """
    Apply quantization to the model.
    Args:
        model: The model to quantize
        bits: Number of bits for quantization (default: 4)
        quantization_method: 'uniform' for W4A4Linear, 'qat' for W4A4LinearQAT
        depth: Current depth in model (for pretty printing)
    """
    if hasattr(model, 'base_model') and hasattr(model, 'peft_config'):
        print("Warning: PEFT model detected. Quantizing base model only.")
        quantize_model(model.base_model, bits, quantization_method, depth+1)
        return model
    print(f"{'  ' * depth}Processing model of type: {type(model).__name__}")
    linear_class = QATW4A4Linear if quantization_method == 'qat' else W4A4Linear
    def replace_linear_recursive(module, name=""):
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child_module, torch.nn.Linear):
                print(f"{'  ' * depth}Quantizing Linear layer: {full_name}")
                quantized_layer = linear_class(child_module, num_bits=bits)
                setattr(module, child_name, quantized_layer)
            else:
                replace_linear_recursive(child_module, full_name)
    replace_linear_recursive(model)
    return model

def evaluate_model(model, eval_dataset, tokenizer, max_samples=100, description=""):
    """Evaluate the model and return perplexity with improved error handling"""
    print(f"\n--- Evaluating {description} Model ---")
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    device = next(model.parameters()).device
    # Ensure we don't exceed dataset size
    num_samples = min(max_samples, len(eval_dataset))
    eval_subset = eval_dataset.select(range(num_samples))
    processed_samples = 0
    failed_samples = 0
    # Remove forced .cpu() calls; keep model on device
    with torch.no_grad():
        for i, example in enumerate(tqdm(eval_subset, desc=f"Evaluating {description}")):
            try:
                # Create tensors directly on the model's device
                input_ids = torch.tensor(example['input_ids'], device=device).unsqueeze(0)
                attention_mask = torch.tensor(example['attention_mask'], device=device).unsqueeze(0)
                labels = input_ids.clone()
                # Skip empty examples
                if input_ids.numel() == 0 or attention_mask.sum() == 0:
                    print(f"Skipping empty example {i}")
                    failed_samples += 1
                    continue
                # Forward pass on device
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                # Get the loss and handle potential NaN/Inf
                loss = outputs.loss
                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss ({loss}) for example {i}")
                    failed_samples += 1
                    continue
                # Update running loss
                total_loss += loss.item()
                processed_samples += 1
                # Calculate number of non-padding tokens
                num_tokens = attention_mask.sum().item()
                if num_tokens == 0:
                    print(f"Warning: No valid tokens in example {i}")
                    failed_samples += 1
                    continue
                # Accumulate token count
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                processed_samples += 1
            except Exception as e:
                print(f"Error processing example {i}: {str(e)}")
                failed_samples += 1
                continue
    # Print evaluation summary
    print(f"\nEvaluation complete for {description}")
    print(f"Processed: {processed_samples} examples")
    print(f"Failed: {failed_samples} examples")
    if total_tokens == 0:
        print("No valid tokens found for evaluation!")
        return {"eval_loss": float('inf'), "perplexity": float('inf')}
    # Calculate final metrics
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"Evaluation Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print("-" * 50)
    return {
        "eval_loss": avg_loss, 
        "perplexity": perplexity.item(),
        "processed_samples": processed_samples,
        "failed_samples": failed_samples,
        "total_tokens": total_tokens
    }

# Strict LoRA adapter loading
import sys

def load_model_and_tokenizer(model_name="gpt2", lora_adapters_path=None, require_lora=False):
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None
    )
    if lora_adapters_path:
        if not os.path.exists(lora_adapters_path):
            print(f"ERROR: LoRA adapters path '{lora_adapters_path}' not found.")
            if require_lora:
                sys.exit(1)
            else:
                print("Using base model only.")
        else:
            print("Loading LoRA adapters...")
            model = PeftModel.from_pretrained(
                model,
                lora_adapters_path,
                torch_dtype=torch.float32
            )
    elif require_lora:
        print("ERROR: LoRA adapters path required but not provided.")
        sys.exit(1)
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

def calibrate_quantization(model, calibration_data, tokenizer, num_samples=50, description=""):
    """Calibrate the quantization parameters using a separate calibration set"""
    print(f"\n--- Calibrating quantization parameters for {description} ---")
    
    # Set model to eval mode for calibration and move to CPU
    was_training = model.training
    model.eval()
    original_device = next(model.parameters()).device
    # model = model.cpu()
    
    # Ensure we don't exceed dataset size
    num_samples = min(num_samples, len(calibration_data))
    calibration_subset = calibration_data.select(range(num_samples))
    
    print(f"Using {len(calibration_subset)} samples for calibration...")
    
    # Collect activation ranges
    input_ranges = []
    output_ranges = []
    
    # Hook to collect activation ranges
    def forward_hook(module, input, output):
        if isinstance(input, tuple):
            input = input[0]
        if isinstance(output, tuple):
            output = output[0]
            
        # Get absolute max for input and output
        input_max = input.detach().abs().max().item()
        output_max = output.detach().abs().max().item()
        
        input_ranges.append(input_max)
        output_ranges.append(output_max)
    
    # Register hooks on all W4A4Linear layers
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (W4A4Linear, QATW4A4Linear)):
            handles.append(module.register_forward_hook(forward_hook))
    
    # Run calibration on CPU
    with torch.no_grad():
        for i, example in enumerate(tqdm(calibration_subset, desc="Calibrating")):
            try:
                # Convert to tensor and keep on CPU
                input_ids = torch.tensor(example['input_ids'], device='mps').unsqueeze(0)
                attention_mask = torch.tensor(example['attention_mask'], device='mps').unsqueeze(0)
                
                # Skip empty examples
                if input_ids.numel() == 0 or attention_mask.sum() == 0:
                    continue
                
                # Forward pass to collect activation statistics
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
                
            except Exception as e:
                print(f"Warning: Error during calibration on example {i}: {str(e)}")
                continue
    
    # Calculate and set scales based on collected ranges
    if input_ranges and output_ranges:
        input_scale = max(input_ranges) / 7.0  # For int4 symmetric
        output_scale = max(output_ranges) / 7.0
        
        # Set scales for all layers
        for module in model.modules():
            if isinstance(module, (W4A4Linear, QATW4A4Linear)):
                module.input_scale.fill_(input_scale)
                module.output_scale.fill_(output_scale)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Restore original training mode and device
    model.train(was_training)
    if str(original_device) != 'cpu':
        model = model.to(original_device)
        
    print("Calibration complete!")

# User-configurable sample sizes
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib_samples', type=int, default=100, help='Number of samples for calibration')
    parser.add_argument('--eval_samples', type=int, default=500, help='Number of samples for evaluation')
    args = parser.parse_args()
    set_seed(42)
    model_name = "gpt2"
    baseline_lora_path = "baseline_lora_adapters"
    qat_lora_path = "output_qat_lora_2/checkpoint-292"
    # Device selection (already handled above)
    print("\nLoading and preprocessing data...")
    dataset, tokenizer = load_and_preprocess_wikitext2(model_name)
    val_dataset = dataset["validation"]
    val_size = len(val_dataset)
    calibration_dataset = val_dataset.select(range(min(args.calib_samples, val_size)))
    eval_dataset = val_dataset.select(range(args.calib_samples, min(args.calib_samples + args.eval_samples, val_size)))
    print(f"Using {len(calibration_dataset)} samples for calibration")
    print(f"Using {len(eval_dataset)} samples for evaluation")
    def test_model(model_name, lora_path, model_type, quantization_method, require_lora=False):
        print(f"\n{'='*80}")
        print(f"TESTING {model_type.upper()} MODEL")
        print(f"{'='*80}")
        print("\n--- Loading model ---")
        model, _ = load_model_and_tokenizer(model_name, lora_path, require_lora=require_lora)
        model = model.to(device)
        eval_results = evaluate_model(
            model,
            eval_dataset,
            tokenizer,
            max_samples=args.eval_samples,
            description=f"Original {model_type}"
        )
        print("\n--- Creating new model instance for quantization ---")
        quantized_model, _ = load_model_and_tokenizer(model_name, lora_path, require_lora=require_lora)
        quantized_model = quantized_model.to(device)
        print(f"\n--- Applying quantization ({quantization_method}) ---")
        quantize_model(quantized_model, bits=4, quantization_method=quantization_method)
        calibrate_quantization(
            quantized_model,
            calibration_dataset,
            tokenizer,
            num_samples=args.calib_samples,
            description=model_type
        )
        quantized_eval_results = evaluate_model(
            quantized_model,
            eval_dataset,
            tokenizer,
            max_samples=args.eval_samples,
            description=f"Quantized {model_type}"
        )
        original_ppl = eval_results['perplexity']
        quantized_ppl = quantized_eval_results['perplexity']
        ppl_increase = quantized_ppl - original_ppl
        ppl_increase_pct = (ppl_increase / original_ppl) * 100
        print(f"\n{'='*60}")
        print(f"PERPLEXITY COMPARISON - {model_type.upper()}")
        print(f"{'='*60}")
        print(f"Original PPL:    {original_ppl:10.2f}")
        print(f"Quantized PPL:   {quantized_ppl:10.2f}")
        print(f"-" * 60)
        print(f"Absolute change: {ppl_increase:10.2f}")
        print(f"Relative change: {ppl_increase_pct:9.1f}%")
        print(f"Quantization method: {quantization_method}")
        print("="*60)
        del model
        del quantized_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            'original_loss': eval_results['eval_loss'],
            'original_ppl': eval_results['perplexity'],
            'quantized_loss': quantized_eval_results['eval_loss'],
            'quantized_ppl': quantized_eval_results['perplexity'],
            'quantization_method': quantization_method
        }
    # Test base GPT-2 model (no LoRA)
    print("\n" + "="*60)
    print("TESTING BASE GPT-2 (NO FINE-TUNING)")
    print("="*60)
    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    base_model.eval()
    print("\n--- Evaluating Base GPT-2 Model ---")
    base_eval_results = evaluate_model(base_model, eval_dataset, tokenizer, max_samples=args.eval_samples)
    print("\n--- Creating copy of base model for quantization ---")
    quantized_base = copy.deepcopy(base_model)
    print("\n--- Applying quantization (uniform) ---")
    quantize_model(quantized_base, bits=4, quantization_method='uniform')
    quantized_base = quantized_base.to(device)
    print("\n--- Calibrating quantization parameters for base model ---")
    calibrate_quantization(quantized_base, calibration_dataset, tokenizer, num_samples=args.calib_samples)
    print("\n--- Evaluating Quantized Base Model ---")
    quantized_base_results = evaluate_model(quantized_base, eval_dataset, tokenizer, max_samples=args.eval_samples)
    base_results = {
        'original_loss': base_eval_results['eval_loss'],
        'original_ppl': base_eval_results['perplexity'],
        'quantized_loss': quantized_base_results['eval_loss'],
        'quantized_ppl': quantized_base_results['perplexity'],
        'quantization_method': 'uniform'
    }
    baseline_results = test_model(model_name, baseline_lora_path, "Baseline LoRA", quantization_method='uniform', require_lora=True)
    qat_results = test_model(model_name, qat_lora_path, "QAT LoRA", quantization_method='qat', require_lora=True)
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"{'Model':<20} | {'Original PPL':>12} | {'Quantized PPL':>12} | {'Degradation':>12} | {'Quant Method':>15}")
    print("-" * 80)
    print(f"{'Base GPT-2':<20} | {base_results['original_ppl']:12.2f} | {base_results['quantized_ppl']:12.2f} | {base_results['quantized_ppl'] - base_results['original_ppl']:12.2f} | {base_results['quantization_method']:>15}")
    print(f"{'Baseline LoRA':<20} | {baseline_results['original_ppl']:12.2f} | {baseline_results['quantized_ppl']:12.2f} | {baseline_results['quantized_ppl'] - baseline_results['original_ppl']:12.2f} | {baseline_results['quantization_method']:>15}")
    print(f"{'QAT LoRA':<20} | {qat_results['original_ppl']:12.2f} | {qat_results['quantized_ppl']:12.2f} | {qat_results['quantized_ppl'] - qat_results['original_ppl']:12.2f} | {qat_results['quantization_method']:>15}")
    print("="*80)
    # Save quantized model if needed (optional)
    save_quantized = False  # Set to True if you want to save
    if save_quantized:
        print("\nSaving quantized model...")
        torch.save(quantized_model.state_dict(), "quantized_model_w4a4.pth")
        print("Quantized model saved!")

if __name__ == "__main__":
    main()