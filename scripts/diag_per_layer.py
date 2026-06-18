"""Per-layer logit comparison to find where deployed model diverges from original."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from peqat_deploy.runtime.loader import build_cuda_model, CUDAQuantizedLinear
from datasets import load_dataset
from fake_quant import load_peqat_model, PEQATLoRAModule

device = "cuda"
export_dir = sys.argv[1]
orig_dir = export_dir.replace("_export", "")

# Load both models
print("Loading deployed model...")
deployed, tokenizer = build_cuda_model(export_dir, "meta-llama/Llama-3.1-8B", device)

print("Loading original model...")
orig, _ = load_peqat_model(orig_dir, "meta-llama/Llama-3.1-8B", torch.device(device))
orig.to(torch.bfloat16)

# Get one batch
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
enc = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids.to(device)
batch = enc[:, :2048]

# Hook: capture output at each replaced linear
orig_acts, dep_acts = {}, {}

def make_hook(d, name):
    def hook(module, inp, out):
        d[name] = out[0].float() if isinstance(out, tuple) else out.float()
    return hook

# Register hooks on original
orig_handles = []
for name, mod in orig.named_modules():
    if isinstance(mod, PEQATLoRAModule):
        orig_handles.append(mod.register_forward_hook(make_hook(orig_acts, name)))

# Register hooks on deployed
dep_handles = []
for name, mod in deployed.named_modules():
    if isinstance(mod, CUDAQuantizedLinear):
        dep_handles.append(mod.register_forward_hook(make_hook(dep_acts, name)))

# Run original
print("Running original model...")
with torch.no_grad():
    orig(batch)
for h in orig_handles:
    h.remove()

# Run deployed
print("Running deployed model...")
with torch.no_grad():
    deployed(batch)
for h in dep_handles:
    h.remove()

# Compare
shared = sorted(set(orig_acts.keys()) & set(dep_acts.keys()))
print(f"\nFound {len(shared)} shared modules to compare")
if not shared:
    print(f"Orig keys sample: {list(orig_acts.keys())[:3]}")
    print(f"Dep keys sample:  {list(dep_acts.keys())[:3]}")
    sys.exit(1)

print(f"\n{'Layer':55s} {'MaxDiff':>10s} {'MeanDiff':>10s}")
print("-" * 75)
first_bad = None
for name in shared:
    o = orig_acts[name]
    d = dep_acts[name]
    diff = (o - d).abs()
    maxd = diff.max().item()
    meand = diff.mean().item()
    flag = " <<<" if maxd > 1.0 else ""
    if maxd > 1.0 and first_bad is None:
        first_bad = name
    print(f"  {name:53s} {maxd:10.4f} {meand:10.6f}{flag}")

if first_bad:
    print(f"\nFirst layer with max_diff > 1.0: {first_bad}")

# Final logits
print("\nFinal logits:")
diff = (orig(batch).logits.float() - deployed(batch).logits.float()).abs()
print(f"  max_diff={diff.max().item():.4f} mean_diff={diff.mean().item():.6f}")
