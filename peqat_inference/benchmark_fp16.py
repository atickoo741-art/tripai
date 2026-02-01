import torch
import time
from transformers import AutoModelForCausalLM
from peft import PeftModel

device = torch.device('cuda')

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16
).to(device)

peft_model = PeftModel.from_pretrained(
    base_model,
    "../../peqat-checkpoints/output_qat_lora_old_clip_backward_co_4867_001"
).to(device)

# Get test layer
layer = peft_model.base_model.model.model.layers[0].self_attn.q_proj

# Test input
N, k = 128, 4096
X = torch.randn(N, k, device=device, dtype=torch.bfloat16)

# Warmup
for _ in range(10):
    _ = layer(X)
torch.cuda.synchronize()

# Benchmark
start = time.perf_counter()
for _ in range(100):
    Y = layer(X)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) / 100 * 1000  # ms

print(f"=== FP16 Baseline ===")
print(f"Avg time: {elapsed:.3f} ms")
print(f"Output shape: {Y.shape}")
print(f"Output range: [{Y.min():.4f}, {Y.max():.4f}]")
