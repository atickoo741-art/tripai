"""Compare exported quantized weights vs training's on-the-fly WeightSTEQuant in bf16."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from peqat_deploy.runtime.loader import load_export
from fake_quant import load_peqat_model, WeightSTEQuant

device = "cuda"
export_dir = sys.argv[1]
orig_dir = export_dir.replace("_export", "")

# Load original model (just to get the first layer's weights)
print("Loading original model...")
orig, _ = load_peqat_model(orig_dir, "meta-llama/Llama-3.1-8B", torch.device(device))
orig.to(torch.bfloat16)

# Load export artifacts
print("Loading export artifacts...")
export = load_export(export_dir, device)
smooth, clip = export["smooth"], export["clip"]

# Get a representative projection to compare (e.g., layer 0 gate_proj)
mod_name = None
for k in smooth:
    if "layers.0.mlp.gate_proj" in k:
        mod_name = k
        break

print(f"\nComparing layer: {mod_name}")

# Get original model's module
for name, mod in orig.named_modules():
    if mod_name.endswith(name) or name.endswith("mlp.gate_proj"):
        if hasattr(mod, 'get_merged_weight'):
            print(f"  Found PEQATLoRAModule at: {name}")
            # Training's forward would compute:
            merged = mod.get_merged_weight()  # bf16
            smooth_val = mod.qat_core.smooth_factor  # bf16
            clip_val = mod.qat_core.constrained_clip_val  # bf16
            scaled_w = merged * smooth_val.unsqueeze(0)  # bf16
            
            # Training's WeightSTEQuant (bf16):
            qmin, qmax = -8, 7
            row_max = scaled_w.detach().abs().amax(dim=1, keepdim=True)
            scale_bf16 = (row_max / qmax).clamp(min=1e-8)
            w_q_bf16 = torch.clamp(torch.round(scaled_w / scale_bf16), qmin, qmax)
            w_dq_bf16 = w_q_bf16 * scale_bf16
            
            # Get export artifacts (matching dequant path)
            layer = export["layers"][mod_name]
            packed_w = layer["packed_w"]  # uint8
            w_scale_fp16 = layer["w_scale"]  # fp16 from export
            
            # Dequant exported weights in bf16 (our runtime path)
            K = layer["cols"]
            N = layer["rows"]
            q_w_export = torch.zeros(N, K, dtype=torch.int8, device=device)
            for r in range(N):
                for c in range(K):
                    packed_idx = (r * K + c) // 2
                    byte = packed_w[packed_idx].item()
                    nibble = byte & 0xF if c % 2 == 0 else (byte >> 4) & 0xF
                    q = nibble if nibble <= 7 else nibble - 16
                    q_w_export[r, c] = q
            
            w_scale_bf16 = w_scale_fp16.bfloat16().unsqueeze(1)
            w_dq_export = q_w_export.float() * w_scale_bf16.float()  # dequant in FP32 for comparison
            
            # Compare
            diff = (w_dq_export - w_dq_bf16.float()).abs()
            q_diff = (q_w_export != w_q_bf16.to(torch.int8))
            n_diff = q_diff.sum().item()
            total = N * K
            
            print(f"\n  Weight comparison:")
            print(f"  Scale (bf16 training): mean={scale_bf16.mean().item():.6f}")
            print(f"  Scale (export fp16→bf16): mean={w_scale_fp16.bfloat16().mean().item():.6f}")
            print(f"  INT4 values differing: {n_diff}/{total} ({100*n_diff/total:.2f}%)")
            print(f"  Max dequant diff: {diff.max().item():.6f}")
            print(f"  Mean dequant diff: {diff.mean().item():.8f}")
            
            if n_diff > 0:
                print(f"\n  Sample differing indices (first 20):")
                diffs = torch.where(q_diff)[0]
                for idx in diffs[:20]:
                    r, c = idx // K, idx % K
                    print(f"    [{r},{c}]: train_q={w_q_bf16[r,c].item()}, export_q={q_w_export[r,c].item()}, "
                          f"train_dq={w_dq_bf16[r,c].item():.6f}, export_dq={w_dq_export[r,c].item():.6f}")
            
            # Also compare activation preprocessing
            print(f"\n  Activation clip comparison:")
            print(f"  Clip (training bf16): mean={clip_val.float().mean().item():.4f}")
            print(f"  Clip (export FP32): mean={clip[mod_name].mean().item():.4f}")
            print(f"  Max clip diff: {(clip_val.float() - clip[mod_name]).abs().max().item():.6f}")
            
            break
