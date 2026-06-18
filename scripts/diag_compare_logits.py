"""Compare original vs deployed model logits on same input to diagnose PPL gap."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from peqat_deploy.runtime.loader import build_cuda_model
from datasets import load_dataset

device = "cuda"

# Load both models
print("Loading deployed model...")
deployed, tokenizer = build_cuda_model(
    sys.argv[1], "meta-llama/Llama-3.1-8B", device
)

print("Loading original model...")
from fake_quant import load_peqat_model
orig, _ = load_peqat_model(sys.argv[1].replace("_export", ""), "meta-llama/Llama-3.1-8B", torch.device(device))
orig.to(torch.bfloat16)

# Get one batch of input
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
enc = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids.to(device)
batch = enc[:, :2048]

# Run both
print("Running deployed model...")
with torch.no_grad():
    logits_deployed = deployed(batch).logits.float()

print("Running original model...")
with torch.no_grad():
    logits_orig = orig(batch).logits.float()

# Compare
diff = (logits_deployed - logits_orig).abs()
print(f"\nLogit comparison on first 2048-token batch:")
print(f"  Max absolute diff: {diff.max().item():.4f}")
print(f"  Mean absolute diff: {diff.mean().item():.6f}")
print(f"  Max relative diff: {(diff / (logits_orig.abs() + 1e-8)).max().item():.4f}")

# Check the argmax difference
pred_deployed = logits_deployed.argmax(dim=-1)
pred_orig = logits_orig.argmax(dim=-1)
agreement = (pred_deployed == pred_orig).float().mean().item()
print(f"\n  Token prediction agreement: {agreement*100:.2f}%")

# First few tokens
n_show = 20
print(f"\nFirst {n_show} token predictions:")
print(f"  Orig:     {tokenizer.decode(pred_orig[0, :n_show])}")
print(f"  Deployed: {tokenizer.decode(pred_deployed[0, :n_show])}")

# Check middle layer
print(f"\nChecking specific layers...")
for name, mod in deployed.named_modules():
    if hasattr(mod, 'packed_w'):
        print(f"  Layer {name}: out_f={mod.out_features}, in_f={mod.in_features}")
        break
