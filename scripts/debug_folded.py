"""Debug: compare a single layer's output between original and folded path."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from peqat_deploy.runtime.loader import load_export, CUDAQuantizedLinear
from peqat_deploy.quant.fold_act_scale import unpack_int4, fold_act_scale_into_weights

device = "cuda"
export_dir = os.path.expanduser(
    "~/peqat-checkpoints/output_qat_lora_llama3.1_8b_wiki_fix_b8e7_001_export_4"
)

print("=== Loading export artifacts ===")
export = load_export(export_dir, device=device, use_folded=True)

mod_name = "base_model.model.model.layers.0.self_attn.q_proj"
layer = export["layers"][mod_name]
smooth = export["smooth"][mod_name]
clip = export["clip"][mod_name]

print(f"Module: {mod_name}")
print(f"  rows={layer['rows']}, cols={layer['cols']}")
print(f"  packed_w.shape  = {list(layer['packed_w'].shape)}")
print(f"  w_scale.shape   = {list(layer['w_scale'].shape)}")
print(f"  smooth.shape    = {list(smooth.shape)}")
print(f"  clip.shape      = {list(clip.shape)}")
has_folded = "packed_w_folded" in layer
print(f"  has folded      = {has_folded}")
if has_folded:
    print(f"  pw_folded.shape = {list(layer['packed_w_folded'].shape)}")
    print(f"  ws_folded.shape = {list(layer['w_scale_folded'].shape)}")

# --- Build two modules: one original, one with folded weights ---
mod_orig = CUDAQuantizedLinear(
    packed_w=layer["packed_w"], w_scale=layer["w_scale"],
    smooth=smooth, clip=clip,
    out_features=layer["rows"], in_features=layer["cols"],
)
mod_fold = CUDAQuantizedLinear(
    packed_w=layer["packed_w"], w_scale=layer["w_scale"],
    smooth=smooth, clip=clip,
    out_features=layer["rows"], in_features=layer["cols"],
    packed_w_folded=layer.get("packed_w_folded"),
    w_scale_folded=layer.get("w_scale_folded"),
)
print(f"\nmod_orig._fold_available = {mod_orig._fold_available}")
print(f"mod_fold._fold_available = {mod_fold._fold_available}")

# Also build a module specifically for weight comparison
mod_unpack = CUDAQuantizedLinear(
    packed_w=layer["packed_w"], w_scale=layer["w_scale"],
    smooth=smooth, clip=clip,
    out_features=layer["rows"], in_features=layer["cols"],
    packed_w_folded=layer.get("packed_w_folded"),
    w_scale_folded=layer.get("w_scale_folded"),
)

# --- Forward on same input ---
torch.manual_seed(42)
x = torch.randn(128, layer["cols"], dtype=torch.bfloat16, device=device)

with torch.no_grad():
    y_orig = mod_orig(x)
    y_fold = mod_fold(x)

diff = (y_orig.float() - y_fold.float()).abs()
print(f"\n=== Output comparison (128 x {layer['rows']}) ===")
print(f"  y_orig: mean={y_orig.float().mean().item():.6f}  std={y_orig.float().std().item():.6f}")
print(f"  y_fold: mean={y_fold.float().mean().item():.6f}  std={y_fold.float().std().item():.6f}")
print(f"  |diff|: max={diff.max().item():.6f}  mean={diff.mean().item():.6f}")
ref_scale = y_orig.float().abs().mean().item() + 1e-8
print(f"  |diff|/|y_orig| = {diff.mean().item() / ref_scale:.6f}")

if diff.mean().item() > 0.1:
    print("\n=== LARGE DISCREPANCY — digging deeper ===")

    # 1. Manually recompute folded weights vs what's on disk
    print("\n[1] Folded weight validation...")
    packed_demo = {
        mod_name: {
            "weight_int4_bytes": layer["packed_w"].cpu().numpy().tobytes(),
            "weight_scale": layer["w_scale"].cpu(),
            "shape": (layer["rows"], layer["cols"]),
        }
    }
    clip_demo = {mod_name: clip.cpu()}
    folded_demo = fold_act_scale_into_weights(packed_demo, clip_demo)

    w_q_folded_recomp = unpack_int4(
        folded_demo[mod_name]["weight_int4_bytes"],
        layer["rows"], layer["cols"]
    )
    ws_recomp = folded_demo[mod_name]["weight_scale"]
    w_q_folded_loaded = mod_unpack._get_folded_weights()

    wt_diff = (w_q_folded_recomp.float() - w_q_folded_loaded.cpu().float()).abs()
    print(f"  recomputed vs loaded |w_q|: max diff = {wt_diff.max().item()}")
    ws_diff = (ws_recomp - layer["w_scale_folded"].cpu()).abs()
    print(f"  recomputed vs loaded w_scale: max diff = {ws_diff.max().item()}")

    # 2. Preprocess sanity
    print("\n[2] Preprocess sanity...")
    import peqat_cuda
    x_q, x_dq, x_scale = peqat_cuda.preprocess_forward_bf16(x, smooth, clip)
    x_dq_reconst = x_q.float() * x_scale.unsqueeze(0).float()
    recon_diff = (x_dq.float() - x_dq_reconst).abs()
    print(f"  x_dq vs x_q * s_x: max={recon_diff.max().item():.6f}  mean={recon_diff.mean().item():.6f}")

    # 3. Manual float64 reference
    print("\n[3] Manual reference computation...")
    w_q_orig = unpack_int4(
        layer["packed_w"].cpu().numpy().tobytes(),
        layer["rows"], layer["cols"]
    )
    w_dq = w_q_orig.float() * layer["w_scale"].cpu().float().unsqueeze(1)
    s_x = clip.cpu().float() / 7.0
    y_ref = x_q.float().cpu() @ (w_dq * s_x.unsqueeze(0)).T

    print(f"  y_ref: mean={y_ref.float().mean().item():.6f}  std={y_ref.float().std().item():.6f}")

    # 4. Compare both paths to reference
    y_orig_err = (y_orig.float().cpu() - y_ref).abs()
    y_fold_err = (y_fold.float().cpu() - y_ref).abs()
    print(f"\n[4] Error vs ideal reference:")
    print(f"  y_orig |err|: max={y_orig_err.max().item():.6f}  mean={y_orig_err.mean().item():.6f}")
    print(f"  y_fold |err|: max={y_fold_err.max().item():.6f}  mean={y_fold_err.mean().item():.6f}")
    print(f"  ratio (fold/orig) = {y_fold_err.mean().item() / (y_orig_err.mean().item() + 1e-8):.4f}")

    # 5. NaN/Inf check
    print(f"\n[5] NaN/Inf:")
    print(f"  y_orig: nan={torch.isnan(y_orig).sum().item()}  inf={torch.isinf(y_orig).sum().item()}")
    print(f"  y_fold: nan={torch.isnan(y_fold).sum().item()}  inf={torch.isinf(y_fold).sum().item()}")

    # 6. Scale stats
    print(f"\n[6] Scale distributions:")
    print(f"  s_x:          mean={s_x.mean().item():.6f}  std={s_x.std().item():.6f}  "
          f"min={s_x.min().item():.6f}  max={s_x.max().item():.6f}")
    ws_o = layer['w_scale'].cpu().float()
    ws_f = layer['w_scale_folded'].cpu().float()
    ratio = ws_f / ws_o.clamp(min=1e-8)
    print(f"  w_scale orig:  mean={ws_o.mean().item():.6f}  std={ws_o.std().item():.6f}")
    print(f"  w_scale fold:  mean={ws_f.mean().item():.6f}  std={ws_f.std().item():.6f}")
    print(f"  fold/orig:     mean={ratio.mean().item():.4f}  std={ratio.std().item():.4f}  "
          f"min={ratio.min().item():.4f}  max={ratio.max().item():.4f}")

    # 7. Re-quantization error
    print(f"\n[7] Re-quantization error:")
    w_new_dq = w_q_folded_recomp.float() * ws_recomp.unsqueeze(1).float()
    w_new_target = w_dq * s_x.unsqueeze(0)
    requant_err = (w_new_dq - w_new_target).abs()
    print(f"  |err|: mean={requant_err.mean().item():.8f}  max={requant_err.max().item():.8f}")
    print(f"  relative: mean={requant_err.mean().item() / (w_new_target.abs().mean().item() + 1e-8):.8f}")

    # 8. Conclusion
    print(f"\n[8] Conclusion:")
    if y_fold_err.mean().item() > 1.0 and y_orig_err.mean().item() < 0.1:
        print("  FOLDED path diverges from reference, ORIGINAL path is correct")
        print("  → Bug is in the folded weight data or scale application")
    elif y_fold_err.mean().item() < 0.1 and y_orig_err.mean().item() > 1.0:
        print("  ORIGINAL path diverges, FOLDED path is correct")
        print("  → Bug is in the bf16 WMMA kernel")
    elif y_fold_err.mean().item() > 1.0 and y_orig_err.mean().item() > 1.0:
        print("  BOTH paths diverge → bug in shared code (preprocess kernel)")
    else:
        print("  Both paths match reference ✓")
else:
    print("\n✓ Both paths agree within tolerance")
