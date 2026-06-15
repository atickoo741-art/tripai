import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from safetensors.torch import load_file as safetensors_load

from peqat_deploy.quant.reference import peqat_linear_reference


def _reconstruct_from_packed(layer_dir: str, x: torch.Tensor) -> torch.Tensor:
    pack_meta_path = os.path.join(layer_dir, "pack_meta.json")
    with open(pack_meta_path) as f:
        meta = json.load(f)

    weight_int4_path = os.path.join(layer_dir, "weight_int4.bin")
    with open(weight_int4_path, "rb") as f:
        packed_bytes = f.read()

    weight_scale_path = os.path.join(layer_dir, "weight_scale.fp16")
    with open(weight_scale_path, "rb") as f:
        scale_bytes = f.read()

    shape_path = os.path.join(layer_dir, "shape.json")
    with open(shape_path) as f:
        shape = json.load(f)

    rows, cols = shape["rows"], shape["cols"]
    scale = torch.frombuffer(scale_bytes, dtype=torch.float16).clone()

    packed = torch.frombuffer(packed_bytes, dtype=torch.uint8).clone()
    num_packed_elements = (rows * cols + 1) // 2
    packed = packed[:num_packed_elements]

    pack_order = meta.get("pack_order", "low_high")
    if pack_order == "low_high":
        low = packed & 0xF
        high = (packed >> 4) & 0xF
    else:
        low = (packed >> 4) & 0xF
        high = packed & 0xF

    interleaved = torch.stack([low, high], dim=1).flatten()
    w_q = interleaved[: rows * cols].reshape(rows, cols).to(torch.int8)
    w_q = torch.clamp(w_q, -8, 7)

    w_dq = w_q.float() * scale.unsqueeze(1).float()

    y = x.float() @ w_dq.T
    return y


def main():
    parser = argparse.ArgumentParser(
        description="Validate a PE-QAT export against Algorithm 1 reference"
    )
    parser.add_argument(
        "export_dir",
        type=str,
        help="Path to exported PE-QAT directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for validation",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length for validation",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="Max allowed relative error per layer",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing validation results",
    )
    args = parser.parse_args()

    export_dir = args.export_dir
    manifest_path = os.path.join(export_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"Error: manifest.json not found in {export_dir}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    activation_dir = os.path.join(export_dir, "activation")
    weights_dir = os.path.join(export_dir, "weights")
    validation_dir = os.path.join(export_dir, "validation")

    os.makedirs(validation_dir, exist_ok=True)

    sample_outputs_path = os.path.join(validation_dir, "sample_outputs.json")
    if os.path.exists(sample_outputs_path) and not args.overwrite:
        print(f"Validation results already exist: {sample_outputs_path}")
        print("Use --overwrite to re-run.")
        sys.exit(0)

    smooth_dict = safetensors_load(os.path.join(activation_dir, "smooth.safetensors"))
    clip_dict = safetensors_load(os.path.join(activation_dir, "clip.safetensors"))

    module_names = list(smooth_dict.keys())
    print(f"Validating {len(module_names)} modules...")

    results = {}
    all_passed = True

    for module_name in module_names:
        safe_name = module_name.replace(".", "_")
        layer_dir = os.path.join(weights_dir, safe_name)
        if not os.path.isdir(layer_dir):
            print(f"  Skipping {module_name}: no weight artifacts")
            continue

        shape_path = os.path.join(layer_dir, "shape.json")
        with open(shape_path) as f:
            shape = json.load(f)
        d, k = shape["rows"], shape["cols"]

        x = torch.randn(args.batch_size, args.seq_len, k)
        w_rand = torch.randn(d, k)
        smooth = smooth_dict[module_name].float()
        clip = clip_dict[module_name].float()

        ref_out = peqat_linear_reference(x, w_rand, smooth, clip)
        packed_out = _reconstruct_from_packed(layer_dir, x)

        ref_flat = ref_out.flatten()
        packed_flat = packed_out.flatten()

        diff = (ref_flat - packed_flat).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        ref_norm = ref_flat.norm().item() + 1e-8
        rel_err = diff.sum().item() / ref_norm

        passed = rel_err < args.tolerance
        if not passed:
            all_passed = False

        results[module_name] = {
            "shape": {"rows": d, "cols": k},
            "max_abs_error": max_err,
            "mean_abs_error": mean_err,
            "relative_error": rel_err,
            "tolerance": args.tolerance,
            "passed": passed,
        }

        status = "PASS" if passed else "FAIL"
        print(f"  {module_name}: [{status}] rel_err={rel_err:.6f} (tol={args.tolerance})")

    sample_outputs = {
        "config": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "tolerance": args.tolerance,
        },
        "module_count": len(results),
        "all_passed": all_passed,
        "modules": results,
    }

    with open(sample_outputs_path, "w") as f:
        json.dump(sample_outputs, f, indent=2)

    print(f"\nResults saved to: {sample_outputs_path}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
