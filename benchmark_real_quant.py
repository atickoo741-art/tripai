import torch
import torch.nn as nn
import time
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import psutil
from typing import Tuple

# Import your fakequant classes to load the model
from fakequant import QATW4A4LinearCore

class W4A4PackedLinear(nn.Module):
    """
    True INT4 Packed Linear Layer for Inference
    - Weights packed: 2x 4-bit weights in 1x 8-bit byte
    - Activations: Quantized to INT4 at runtime
    - Compute: Dequantize -> Compute -> Quantize (Simulated Kernel)
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Packed weights: uint8 tensor, half the size
        # Shape: [out_features, in_features // 2]
        self.register_buffer(
            "packed_weight", 
            torch.zeros((out_features, in_features // 2), dtype=torch.uint8)
        )
        
        # Scales: FP16/BF16
        self.register_buffer("weight_scale", torch.ones(out_features, 1, dtype=torch.bfloat16))
        self.register_buffer("input_scale", torch.ones(1, 1, dtype=torch.bfloat16))
        
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def forward(self, x):
        # 1. Quantize Input (Fake Kernel Operation)
        # In real kernel: this happens in SRAM
        x_q = (x / self.input_scale).round().clamp(-7, 7).to(torch.int8)
        
        # 2. Unpack Weights (Simulating hardware unpack)
        # In real kernel: unpack on fly
        w_unpacked = self._unpack_weights()
        
        # 3. Integer Matmul (INT8 * INT8 -> INT32)
        # PyTorch only supports INT8, so we use that as proxy for INT4 speed
        out_int32 = torch.nn.functional.linear(x_q.float(), w_unpacked.float(), self.bias)
        
        # 4. Dequantize Output
        out = out_int32 * (self.input_scale * self.weight_scale.T)
        
        return out.to(x.dtype)

    def _unpack_weights(self):
        # Unpack 2x INT4 from UINT8
        # Lower 4 bits
        w_low = (self.packed_weight & 0x0F).to(torch.int8)
        # Upper 4 bits
        w_high = (self.packed_weight >> 4).to(torch.int8)
        
        # Correct 4-bit sign extension
        w_low = (w_low << 4) >> 4
        w_high = (w_high << 4) >> 4
        
        # Stack and reshape
        w_unpacked = torch.stack([w_low, w_high], dim=-1).reshape(self.out_features, self.in_features)
        return w_unpacked

def convert_to_packed_model(qat_model):
    """Convert QAT FakeQuant Model to Packed INT4 Model"""
    print("Converting QAT model to packed INT4...")
    
    total_params = 0
    converted_layers = 0
    
    for name, module in qat_model.named_modules():
        if isinstance(module, QATW4A4LinearCore):
            # Get QAT weights and scales
            weight_fp = module.weight.data
            scale_w = module.weight_quantizer.scale
            scale_x = module.input_quantizer.scale
            
            # Quantize weights to INT4
            w_int = (weight_fp / scale_w).round().clamp(-7, 7).to(torch.int8)
            
            # Pack weights (2 -> 1 byte)
            # Reshape to [out, in/2, 2]
            w_reshaped = w_int.reshape(w_int.shape[0], -1, 2)
            
            # Pack: (high << 4) | (low & 0x0F)
            # Note: Need careful handling of negative numbers in packing
            w_low = w_reshaped[..., 0] & 0x0F
            w_high = w_reshaped[..., 1] & 0x0F
            packed = (w_high << 4) | w_low
            
            # Create new layer
            packed_layer = W4A4PackedLinear(
                module.in_features, 
                module.out_features, 
                bias=module.bias is not None
            )
            
            # Copy data
            packed_layer.packed_weight.copy_(packed.to(torch.uint8))
            packed_layer.weight_scale.copy_(scale_w.reshape(-1, 1).to(torch.bfloat16))
            packed_layer.input_scale.copy_(scale_x.reshape(1, 1).to(torch.bfloat16))
            
            if module.bias is not None:
                packed_layer.bias.copy_(module.bias.to(torch.bfloat16))
            
            # Replace in model (in-place)
            # NOTE: This requires recursive replacement logic
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            parent = qat_model.get_submodule(parent_name)
            setattr(parent, child_name, packed_layer)
            
            converted_layers += 1
            
            # Clear memory
            del w_int, w_reshaped, packed
            
    print(f"Converted {converted_layers} layers to W4A4 packed format")
    return qat_model

def measure_memory(model, device):
    """Measure actual GPU memory usage"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model.to(device)
    dummy_input = torch.randint(0, 1000, (1, 128)).to(device)
    
    # Run once to allocate activations
    with torch.no_grad():
        _ = model(dummy_input)
    
    mem_bytes = torch.cuda.max_memory_allocated(device)
    mem_gb = mem_bytes / (1024**3)
    return mem_gb

def measure_latency(model, tokenizer, device, num_tokens=100):
    """Measure inference latency (tokens/sec)"""
    model.eval()
    input_ids = tokenizer("Hello, I am benchmarking", return_tensors="pt").input_ids.to(device)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=10)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=num_tokens, min_new_tokens=num_tokens)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    tps = num_tokens / total_time
    
    return tps, total_time

def main():
    model_path = "meta-llama/Llama-2-7b-hf"
    adapter_path = "output_qat_lora/final_model"
    device = "cuda"
    
    print("--- BENCHMARKING W4A4 vs FP16 vs INT8 ---")
    
    # 1. FP16 Baseline
    print("\n1. Benchmarking FP16 Baseline...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    mem_fp16 = measure_memory(model_fp16, device)
    tps_fp16, _ = measure_latency(model_fp16, tokenizer, device)
    
    print(f"FP16 Memory: {mem_fp16:.2f} GB")
    print(f"FP16 Speed:  {tps_fp16:.2f} tokens/sec")
    
    del model_fp16
    torch.cuda.empty_cache()
    
    # 2. W4A4 Packed Model (Yours)
    print("\n2. Benchmarking PEQAT W4A4 (Packed)...")
    # Load QAT model
    model_qat = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model_qat = PeftModel.from_pretrained(model_qat, adapter_path)
    
    # Convert to packed INT4
    model_packed = convert_to_packed_model(model_qat)
    
    mem_w4a4 = measure_memory(model_packed, device)
    # Note: Speed will be SLOWER because PyTorch doesn't have native INT4 kernels
    # We measure memory accuracy, but speed requires custom CUDA kernel
    tps_w4a4, _ = measure_latency(model_packed, tokenizer, device)
    
    print(f"W4A4 Memory: {mem_w4a4:.2f} GB")
    print(f"W4A4 Speed:  {tps_w4a4:.2f} tokens/sec (Simulated)")
    
    # Calculate Theoretical Memory
    # 7B params * 0.5 bytes (4-bit) = 3.5 GB + overhead
    print(f"\nTheoretical W4A4 Weight Size: ~3.5 GB")
    print(f"Actual Measured W4A4 Size:    {mem_w4a4:.2f} GB")
    
    # Compare
    print("\n--- RESULTS ---")
    print(f"Memory Savings: {100 * (1 - mem_w4a4/mem_fp16):.1f}% reduction vs FP16")
    print(f"Compression Ratio: {mem_fp16/mem_w4a4:.1f}x")

if __name__ == "__main__":
    main()
