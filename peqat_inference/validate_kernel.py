import torch
import numpy as np
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load trained model
device = torch.device('cuda')
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16
).to(device)

peft_model = PeftModel.from_pretrained(
    base_model,
    "../../peqat-checkpoints/output_qat_lora_old_clip_backward_co_4867_001"
).to(device)

# Get one layer for testing
layer = peft_model.base_model.model.model.layers[0].self_attn.q_proj

# Generate test input
torch.manual_seed(42)
X = torch.randn(128, 4096, device=device, dtype=torch.bfloat16)

# Forward pass
with torch.no_grad():
    Y_pytorch = layer(X)

# Save for comparison
np.save("test_input.npy", X.cpu().float().numpy())
np.save("reference_output.npy", Y_pytorch.cpu().float().numpy())

print(f"✓ Saved reference outputs")
print(f"  Input shape: {X.shape}")
print(f"  Output shape: {Y_pytorch.shape}")
print(f"  Output range: [{Y_pytorch.min():.4f}, {Y_pytorch.max():.4f}]")
