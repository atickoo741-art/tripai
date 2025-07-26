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
    """Fixed 4-bit linear layer with proper fake quantization for inference"""
    
    def __init__(self, original_linear, num_bits=4):
        super().__init__()
        self.num_bits = num_bits
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # Store original weights and bias
        self.register_parameter('weight', nn.Parameter(original_linear.weight.detach().clone()))
        if hasattr(original_linear, 'bias') and original_linear.bias is not None:
            self.register_parameter('bias', nn.Parameter(original_linear.bias.detach().clone()))
        else:
            self.register_parameter('bias', None)
        
        # Quantization parameters (will be set during calibration)
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('weight_zero_point', torch.tensor(0.0))
        self.register_buffer('input_scale', torch.tensor(1.0))
        self.register_buffer('input_zero_point', torch.tensor(0.0))
        # self.register_buffer('output_scale', torch.tensor(1.0))
        # self.register_buffer('output_zero_point', torch.tensor(0.0))
        
        # Initialize weight quantization parameters
        self._init_weight_quantization()
    
    def _init_weight_quantization(self):
        """Initialize weight quantization parameters"""
        weight = self.weight.detach()
        
        # Symmetric quantization for weights
        qmin, qmax = -(2 ** (self.num_bits - 1)), (2 ** (self.num_bits - 1)) - 1  # -8 to 7
        
        max_val = torch.max(torch.abs(weight)).item()
        if max_val == 0:
            max_val = 1.0
        
        scale = max_val / qmax
        self.weight_scale.fill_(scale)
        self.weight_zero_point.fill_(0.0)  # Symmetric
    
    def fake_quantize_tensor(self, x, scale, zero_point, qmin, qmax):
        """Apply fake quantization with straight-through estimator"""
        # Quantize
        x_int = torch.round(x / scale + zero_point)
        x_quant = torch.clamp(x_int, qmin, qmax)
        # Dequantize (fake quantization)
        x_fake_quant = (x_quant - zero_point) * scale
        return x_fake_quant
    
    def forward(self, x):
        # Quantize weights (symmetric, -8 to 7 for 4-bit)
        qmin_w, qmax_w = -(2 ** (self.num_bits - 1)), (2 ** (self.num_bits - 1)) - 1
        weight_quant = self.fake_quantize_tensor(
            self.weight, self.weight_scale, self.weight_zero_point, qmin_w, qmax_w
        )
        
        # Quantize activations (asymmetric, 0 to 15 for 4-bit)
        qmin_a, qmax_a = 0, (2 ** self.num_bits) - 1
        # qmin_a, qmax_a = -(2 ** (self.num_bits - 1)), (2 ** (self.num_bits - 1)) - 1
        input_quant = self.fake_quantize_tensor(
            x, self.input_scale, self.input_zero_point, qmin_a, qmax_a
        )
        
        # Linear operation with quantized tensors
        output = F.linear(input_quant, weight_quant, self.bias)
        
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
def replace_layers_qat(model):
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
            quantized_layer = QATW4A4Linear(module)
            setattr(parent, attrs[-1], quantized_layer)
            
            # Print the parameters after conversion
            print(f"Layer: {name}")
            print(f"PACT clip: {quantized_layer.activation_clip.data}")
            print(f"SmoothQuant alpha: {quantized_layer.alpha.data}")
            print(f"Weight scale: {quantized_layer.weight_scale}")
            print(f"Weight zero point: {quantized_layer.weight_zero_point}")
            print(f"Activation scale: {quantized_layer.act_scale}")
            print(f"Activation zero point: {quantized_layer.act_zero_point}")
            print("-" * 50)


def quantize_model(model, bits=4, quantization_method='uniform', depth=0, source_model=None):
    """
    Apply quantization to the model.
    
    Args:
        model: The model to quantize
        bits: Number of bits for quantization (default: 4)
        quantization_method: 'qat' for quantization-aware training or 'uniform' for post-training quantization
        depth: Current depth in the model (for pretty printing)
        source_model: Optional pre-trained QAT model to copy parameters from
    """
    if hasattr(model, 'base_model') and hasattr(model, 'peft_config'):
        print("Warning: PEFT model detected. Quantizing base model only.")
        source_base = source_model.base_model if source_model and hasattr(source_model, 'base_model') else None
        quantize_model(model.base_model, bits, quantization_method, depth+1, source_base)
        return model
        
    print(f"{'  ' * depth}Processing model of type: {type(model).__name__}")
    
    def replace_linear_recursive(module, name="", source_module=None):
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            source_child = getattr(source_module, child_name, None) if source_module else None
            
            if isinstance(child_module, torch.nn.Linear):
                print(f"{'  ' * depth}Quantizing Linear layer: {full_name}")
                
                if quantization_method == 'qat':
                    # Use QATW4A4Linear which already has weight property implemented above
                    quantized_layer = QATW4A4Linear(child_module, num_bits=bits)
                    print(f"Layer: {name}")
                    print(f"PACT clip: {quantized_layer.activation_clip.data}")
                    print(f"SmoothQuant alpha: {quantized_layer.alpha.data}")  # sigmoid(raw_alpha)
                    print(f"Weight scale: {quantized_layer.weight_scale}")
                    print(f"Weight zero point: {quantized_layer.weight_zero_point}")
                    print(f"Activation scale: {quantized_layer.act_scale}")
                    print(f"Activation zero point: {quantized_layer.act_zero_point}")
                    print("-" * 50)
                    # If we have a source model, copy the learned parameters
                    if source_child and hasattr(source_child, 'activation_clip'):
                        with torch.no_grad():
                            quantized_layer.activation_clip.copy_(source_child.activation_clip)
                            quantized_layer.raw_alpha.copy_(source_child.raw_alpha)
                            quantized_layer.input_absmax.copy_(source_child.input_absmax)
                            quantized_layer.weight_scale.copy_(source_child.weight_scale)
                            quantized_layer.weight_zero_point.copy_(source_child.weight_zero_point)
                            quantized_layer.act_scale.copy_(source_child.act_scale)
                            quantized_layer.act_zero_point.copy_(source_child.act_zero_point)
                            
                else:
                    # Use regular W4A4Linear
                    quantized_layer = W4A4Linear(child_module, num_bits=bits)
                
                setattr(module, child_name, quantized_layer)
            else:
                replace_linear_recursive(child_module, full_name, source_child)
    
    replace_linear_recursive(model, source_module=source_model)
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
                labels[input_ids == tokenizer.pad_token_id] = -100
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
                # total_loss += loss.item()
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
    
    # Set model to eval mode for calibration
    was_training = model.training
    model.eval()
    original_device = next(model.parameters()).device
    
    # Ensure we don't exceed dataset size
    num_samples = min(num_samples, len(calibration_data))
    calibration_subset = calibration_data.select(range(num_samples))
    
    print(f"Using {len(calibration_subset)} samples for calibration...")
    
    # Collect activation ranges and minimum values for W4A4Linear only (QATW4A4Linear handles its own calibration)
    input_ranges = []
    input_mins = []
    output_ranges = []
    
    # Hook to collect activation ranges for W4A4Linear only
    def forward_hook(module, input, output):
        if not isinstance(module, W4A4Linear):
            return
            
        if isinstance(input, tuple):
            input = input[0]
        if isinstance(output, tuple):
            output = output[0]
            
        # Get min/max for input and max for output
        input_tensor = input.detach()
        input_min = input_tensor.min().item()
        input_max = input_tensor.max().item()
        output_max = output.detach().abs().max().item()
        
        # input_ranges.append(input_max-input_min)
        input_ranges.append(input_max)
        input_mins.append(input_min)
        output_ranges.append(output_max)
    
    # Register hooks only on W4A4Linear layers (not QATW4A4Linear)
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, W4A4Linear):
            handles.append(module.register_forward_hook(forward_hook))
    
    # Run calibration
    with torch.no_grad():
        for i, example in enumerate(tqdm(calibration_subset, desc="Calibrating")):
            try:
                # Convert to tensor and move to device
                input_ids = torch.tensor(example['input_ids'], device=original_device).unsqueeze(0)
                attention_mask = torch.tensor(example['attention_mask'], device=original_device).unsqueeze(0)
                
                # Skip empty examples
                if input_ids.numel() == 0 or attention_mask.sum() == 0:
                    continue
                
                # Forward pass to collect activation statistics
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
                
            except Exception as e:
                print(f"Warning: Error during calibration on example {i}: {str(e)}")
                continue
    
    # Calculate and set scales only for W4A4Linear layers
    if input_ranges and output_ranges:
        input_scale = max(input_ranges)-min(input_mins) / 15 
        input_zero_point = round(min(input_mins)/input_scale)
        # input_zero_point = 0

        # output_scale = max(output_ranges) / 15
        
        # Set scales only for W4A4Linear layers
        for module in model.modules():
            if isinstance(module, W4A4Linear):
                module.input_scale.fill_(input_scale)
                module.input_zero_point.fill_(input_zero_point)
                
                # module.output_scale.fill_(output_scale)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Restore original training mode
    model.train(was_training)
        
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
    # baseline_lora_path = "baseline_lora_adapters"
    baseline_lora_path = "output_baseline_lora/checkpoint-292"
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

    def load_quantization_parameters(model, checkpoint_dir):
        """Load the saved quantization parameters"""
        quant_params_path = os.path.join(checkpoint_dir, "quantization_params.pth")
        
        if not os.path.exists(quant_params_path):
            print(f"Warning: No quantization parameters found at {quant_params_path}")
            return False
        
        quant_params = torch.load(quant_params_path, map_location='mps')
        
        # Apply parameters to QATW4A4Linear layers
        for name, module in model.named_modules():
            if isinstance(module, QATW4A4Linear):
                param_prefix = f"{name}."
                
                # Load each parameter if it exists
                for param_name in ['activation_clip', 'raw_alpha', 'input_absmax', 
                                'weight_scale', 'weight_zero_point', 'act_scale', 'act_zero_point']:
                    full_key = f"{param_prefix}{param_name}"
                    if full_key in quant_params:
                        getattr(module, param_name).copy_(quant_params[full_key])
                        
        print(f"Loaded quantization parameters from {quant_params_path}")
        return True


    def test_model(model_name, lora_path, model_type, quantization_method, require_lora=False):
        model, _ = load_model_and_tokenizer(model_name, lora_path, require_lora=require_lora)
        model = model.to(device)
        if quantization_method == "qat":
            print(f"\n--- Applying QAT quantization architecture ---")
            apply_quantization_to_model(model)
            
            # Load the trained quantization parameters
            if not load_quantization_parameters(model, lora_path):
                print("ERROR: Failed to load quantization parameters!")
                return None
                
            quantized_model = model
        else:
            print(f"\n--- Creating new model instance for uniform quantization ---")
            quantized_model, _ = load_model_and_tokenizer(model_name, lora_path, require_lora=require_lora)
            quantized_model = quantized_model.to(device)
            quantize_model(quantized_model, bits=4, quantization_method='uniform')
            calibrate_quantization(quantized_model, calibration_dataset, tokenizer, 
                             num_samples=args.calib_samples, description=model_type)
        print(f"\n{'='*80}")
        print(f"TESTING {model_type.upper()} MODEL")
        print(f"{'='*80}")
        print("\n--- Loading model ---")
        
        eval_results = evaluate_model(
            model,
            eval_dataset,
            tokenizer,
            max_samples=args.eval_samples,
            description=f"Original {model_type}"
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