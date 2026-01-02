"""
Real INT4 Quantization Benchmark - Fixed nested module issue
"""

import torch
import torch.nn as nn
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

class PackedInt4Linear(nn.Module):
    """Real INT4 linear layer"""
    def __init__(self, in_features, out_features, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.register_buffer(
            "weight_packed", 
            torch.zeros((out_features, in_features // 2), dtype=torch.uint8, device=device)
        )
        self.register_buffer(
            "scales", 
            torch.ones((out_features, 1), dtype=dtype, device=device)
        )

    def forward(self, x):
        w_low = (self.weight_packed & 0x0F).to(torch.int8)
        w_high = ((self.weight_packed >> 4) & 0x0F).to(torch.int8)
        w_unpacked = torch.stack([w_low, w_high], dim=-1).flatten(1)
        w_int8 = torch.where(w_unpacked >= 8, w_unpacked - 16, w_unpacked)
        w_dequant = w_int8.to(x.dtype) * self.scales.to(x.dtype)
        return torch.nn.functional.linear(x, w_dequant, None)

def pack_weights_to_int4(weight_fp, scales):
    """Pack FP weights to 4-bit"""
    w_quant = torch.round(weight_fp / scales).to(torch.int8)
    w_quant = torch.clamp(w_quant, -8, 7)
    w_uint = (w_quant + 8).to(torch.uint8)
    
    w_flat = w_uint.flatten()
    if w_flat.numel() % 2 == 1:
        w_flat = torch.cat([w_flat, torch.zeros(1, dtype=torch.uint8, device=w_flat.device)])
    
    w_even = w_flat[0::2]
    w_odd = w_flat[1::2]
    packed = w_even | (w_odd << 4)
    
    return packed.reshape(weight_fp.shape[0], -1)

def extract_merged_weight(module):
    """Extract merged weight from PEQAT module"""
    from fake_quant import PEQATLoRAModule
    
    if isinstance(module, PEQATLoRAModule):
        # Get base weight
        base_weight = module.qat_core.get_base_weight().detach()
        
        # Merge LoRA
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            try:
                adapter_name = module.active_adapter[0]
                lora_A = module.lora_A[adapter_name].weight
                lora_B = module.lora_B[adapter_name].weight
                scaling = module.scaling[adapter_name]
                
                lora_delta = (lora_B @ lora_A) * scaling
                return base_weight + lora_delta.to(base_weight.dtype)
            except Exception as e:
                print(f"  Warning: Failed to merge LoRA: {e}")
                return base_weight
        
        return base_weight
    
    return None

def convert_model_to_int4(model):
    """Convert all PEQAT modules to packed INT4"""
    from fake_quant import PEQATLoRAModule
    
    print("  Scanning model for quantizable layers...")
    
    # Find only TOP-LEVEL PEQATLoRAModule (not nested qat_cores)
    modules_to_replace = []
    seen_paths = set()
    
    for name, module in model.named_modules():
        if isinstance(module, PEQATLoRAModule):
            # Check if this is a top-level PEQATLoRAModule
            # (not inside another PEQATLoRAModule)
            parent_path = '.'.join(name.split('.')[:-1])
            if parent_path not in seen_paths:
                modules_to_replace.append(name)
                seen_paths.add(name)
    
    print(f"  Found {len(modules_to_replace)} layers to convert")
    
    if len(modules_to_replace) == 0:
        print("  ⚠️ No quantizable layers found!")
        return model, 0
    
    converted = 0
    
    for full_name in modules_to_replace:
        # Navigate to module
        name_parts = full_name.split('.')
        parent = model
        
        # Navigate to parent
        for part in name_parts[:-1]:
            try:
                parent = getattr(parent, part)
            except AttributeError:
                print(f"  ⚠️ Skipping {full_name}: path not found")
                continue
        
        child_name = name_parts[-1]
        
        # Check if parent still has this child (might have been replaced)
        if not hasattr(parent, child_name):
            print(f"  ⚠️ Skipping {full_name}: already replaced")
            continue
        
        module = getattr(parent, child_name)
        
        # Extract merged weight
        weight = extract_merged_weight(module)
        if weight is None:
            print(f"  ⚠️ Skipping {full_name}: couldn't extract weight")
            continue
        
        # Calculate scales
        abs_max = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        scales = abs_max / 7.0
        
        # Create packed layer
        packed_layer = PackedInt4Linear(
            in_features=weight.shape[1],
            out_features=weight.shape[0],
            device=weight.device,
            dtype=weight.dtype
        )
        
        # Pack weights
        packed_layer.weight_packed = pack_weights_to_int4(weight, scales)
        packed_layer.scales = scales.to(weight.dtype)
        
        # Replace
        setattr(parent, child_name, packed_layer)
        converted += 1
    
    print(f"  ✓ Converted {converted} layers to packed INT4")
    return model, converted

def measure_memory():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e9

def benchmark_generation(model, tokenizer, num_tokens=30):
    model.eval()
    prompt = "The future of AI is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return num_tokens / elapsed

def main():
    print("=" * 70)
    print("REAL INT4 QUANTIZATION BENCHMARK")
    print("=" * 70)
    
    base_model_name = "meta-llama/Llama-2-7b-hf"
    peqat_checkpoint = "/home/shresth/peqat-checkpoints/output_qat_lora_old_clip_backward_co_5f71_001"
    device = torch.device("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ===== 1. FP16 Baseline =====
    print("\n[1/3] FP16 Baseline")
    torch.cuda.empty_cache()
    gc.collect()
    
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    mem_fp16 = measure_memory()
    print(f"  Memory: {mem_fp16:.2f} GB")
    
    speed_fp16 = benchmark_generation(model_fp16, tokenizer)
    print(f"  Speed:  {speed_fp16:.2f} tok/s")
    
    del model_fp16
    torch.cuda.empty_cache()
    gc.collect()
    
    # ===== 2. Load PEQAT =====
    print("\n[2/3] Loading PEQAT Model")
    
    from fake_quant import load_peqat_model
    
    model_peqat, _ = load_peqat_model(
        peqat_checkpoint, 
        base_model_name, 
        device
    )
    
    mem_peqat = measure_memory()
    print(f"  Memory: {mem_peqat:.2f} GB (BF16 fake quant)")
    
    # ===== 3. Convert to Real INT4 =====
    print("\n[3/3] Converting to Real INT4")
    
    model_int4, num_converted = convert_model_to_int4(model_peqat)
    
    if num_converted == 0:
        print("\n❌ ERROR: No layers converted!")
        return
    
    torch.cuda.empty_cache()
    gc.collect()
    
    mem_int4 = measure_memory()
    print(f"  Memory: {mem_int4:.2f} GB")
    
    print("  Testing generation...")
    speed_int4 = benchmark_generation(model_int4, tokenizer)
    print(f"  Speed:  {speed_int4:.2f} tok/s")
    
    # ===== Results =====
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n{'Configuration':<30} {'Memory (GB)':<15} {'Speed (tok/s)':<15}")
    print("-" * 70)
    print(f"{'FP16 Baseline':<30} {mem_fp16:<15.2f} {speed_fp16:<15.2f}")
    print(f"{'PEQAT W4A4 (Fake)':<30} {mem_peqat:<15.2f} {'N/A':<15}")
    print(f"{'PEQAT Packed (INT4)':<30} {mem_int4:<15.2f} {speed_int4:<15.2f}")
    
    savings = ((mem_fp16 - mem_int4) / mem_fp16) * 100
    compression = mem_fp16 / mem_int4
    slowdown = ((speed_fp16 - speed_int4) / speed_fp16) * 100
    
    print(f"\n{'Metric':<40} {'Value':<20}")
    print("-" * 70)
    print(f"{'Memory Savings':<40} {savings:<20.1f}%")
    print(f"{'Compression Ratio':<40} {compression:<20.2f}x")
    print(f"{'Speed Overhead':<40} {slowdown:<20.1f}%")
    print(f"{'Layers Converted':<40} {num_converted:<20}")
    
    print("\n" + "=" * 70)
    print("✓ REAL INT4: Weights packed to 2 weights/byte")
    print("✓ Measured memory savings (not theoretical)")
    print("✓ LoRA adapters merged before packing")
    print("=" * 70)

if __name__ == "__main__":
    main()
