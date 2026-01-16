# verify_checkpoint.py
from safetensors.torch import load_file
import torch
import os

checkpoint_dir = "../peqat-checkpoints/output_qat_lora_old_clip_backward_co_5f71_001"

# Check adapter_model.safetensors
print("Checking adapter_model.safetensors...")
adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
try:
    tensors = load_file(adapter_path)
    print(f"✓ Valid! Contains {len(tensors)} tensors")
    lora_params = sum(t.numel() for k, t in tensors.items() if 'lora' in k.lower())
    print(f"  LoRA parameters: {lora_params:,}")
except Exception as e:
    print(f"❌ Corrupted: {e}")
    exit(1)

# Check quantization_params.pth
print("\nChecking quantization_params.pth...")
quant_path = os.path.join(checkpoint_dir, "quantization_params.pth")
try:
    params = torch.load(quant_path, map_location='cpu')
    print(f"✓ Valid! Contains {len(params)} modules")
    sample_key = list(params.keys())[0]
    print(f"  Sample module: {sample_key}")
    print(f"  Keys: {list(params[sample_key].keys())}")
except Exception as e:
    print(f"❌ Corrupted: {e}")
    exit(1)

print("\n✅ All files are valid!")
