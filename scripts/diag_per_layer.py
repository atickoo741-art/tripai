"""Per-layer logit comparison to find where deployed model diverges from original."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from peqat_deploy.runtime.loader import build_cuda_model
from datasets import load_dataset

device = "cuda"
export_dir = sys.argv[1]
orig_dir = export_dir.replace("_export", "")

# Load deployed model
print("Loading deployed model...")
deployed, tokenizer = build_cuda_model(export_dir, "meta-llama/Llama-3.1-8B", device)

# Load original model
print("Loading original model...")
from fake_quant import load_peqat_model, PEQATLoRAModule
orig, _ = load_peqat_model(orig_dir, "meta-llama/Llama-3.1-8B", torch.device(device))
orig.to(torch.bfloat16)

# Get one batch
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
enc = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids.to(device)
batch = enc[:, :2048]

# Hook on PEQATLoRAModule in original and CUDAQuantizedLinear in deployed
from peqat_deploy.runtime.loader import CUDAQuantizedLinear

orig_acts = {}
dep_acts = {}

def make_hook(d, name):
    def hook(module, inp, out):
        d[name] = out[0].float() if isinstance(out, tuple) else out.float()
    return hook

orig_handles = []
dep_handles = []

for name, mod in orig.named_modules():
    if isinstance(mod, (PEQATLoRAModule, torch.nn.Linear)):
        if any(t in name for t in ['lora_A', 'lora_B', 'qat_core', 'base_layer']):
            continue
        if 'lm_head' in name:
            continue
        orig_handles.append(mod.register_forward_hook(make_hook(orig_acts, name)))

for name, mod in deployed.named_modules():
    if isinstance(mod, CUDAQuantizedLinear):
        dep_handles.append(mod.register_forward_hook(make_hook(dep_acts, name)))

# Get the shared module names
shared = sorted(set(orig_acts.keys()) & set(dep_acts.keys()))
print(f"\nFound {len(shared)} shared modules to compare")

# Run original
print("Running original model...")
with torch.no_grad():
    orig_acts.clear()
    orig(batch)
for h in orig_handles:
    h.remove()

# Run deployed 
print("Running deployed model...")
# Remove old hooks and re-register
for name, mod in deployed.named_modules():
    if isinstance(mod, CUDAQuantizedLinear):
        dep_handles.append(mod.register_forward_hook(make_hook(dep_acts, name)))

with torch.no_grad():
    dep_acts.clear()
    deployed(batch)
for h in dep_handles:
    h.remove()

# Compare
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

# Also compare final logits
print("\nRunning both for final logit comparison...")
with torch.no_grad():
    logits_orig = orig(batch).logits.float()
    logits_dep = deployed(batch).logits.float()
diff = (logits_orig - logits_dep).abs()
print(f"Final logits: max_diff={diff.max().item():.4f} mean_diff={diff.mean().item():.6f}")
