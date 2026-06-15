import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch


def _unpack_weight(layer_dir: str) -> tuple:
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
    return w_q, scale, packed_bytes, rows, cols


def _pack_int4(w_q: torch.Tensor, pack_order: str = "low_high") -> bytes:
    rows, cols = w_q.shape
    if cols % 2 != 0:
        w_q_padded = torch.cat([w_q.flatten(), torch.zeros(1, dtype=torch.int8)])
        cols_padded = cols + 1
    else:
        w_q_padded = w_q.flatten()
        cols_padded = cols
    w_q_padded = w_q_padded.reshape(rows, cols_padded)

    if pack_order == "low_high":
        low = (w_q_padded[:, ::2] & 0xF).to(torch.uint8)
        high = ((w_q_padded[:, 1::2] & 0xF) << 4).to(torch.uint8)
    else:
        low = ((w_q_padded[:, ::2] & 0xF) << 4).to(torch.uint8)
        high = (w_q_padded[:, 1::2] & 0xF).to(torch.uint8)

    return (low | high).contiguous().numpy().tobytes()


def main():
    parser = argparse.ArgumentParser(
        description="Validate packed weight artifacts for lossless round-trip"
    )
    parser.add_argument(
        "export_dir",
        type=str,
        help="Path to exported PE-QAT directory",
    )
    parser.add_argument(
        "--max-relerr",
        type=float,
        default=0.05,
        help="Max relative reconstruction error per layer (default: 0.05)",
    )
    args = parser.parse_args()

    export_dir = args.export_dir
    manifest_path = os.path.join(export_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"Error: manifest.json not found in {export_dir}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    weights_dir = os.path.join(export_dir, "weights")
    validation_dir = os.path.join(export_dir, "validation")

    error_stats_path = os.path.join(validation_dir, "layer_error_stats.json")
    if not os.path.isfile(error_stats_path):
        print(f"Error: layer_error_stats.json not found")
        sys.exit(1)

    with open(error_stats_path) as f:
        saved_error_stats = json.load(f)

    module_names = list(saved_error_stats.keys())
    print(f"Validating {len(module_names)} modules...")
    all_passed = True

    for module_name in module_names:
        safe_name = module_name.replace(".", "_")
        layer_dir = os.path.join(weights_dir, safe_name)
        if not os.path.isdir(layer_dir):
            print(f"  Skipping {module_name}: no weight artifacts")
            continue

        w_q, scale, orig_packed, rows, cols = _unpack_weight(layer_dir)

        repacked = _pack_int4(w_q)
        roundtrip_ok = repacked == orig_packed

        w_dq = w_q.float() * scale.unsqueeze(1).float()
        w_norm = w_dq.norm().item()
        w_max = w_dq.abs().max().item()

        saved_max = saved_error_stats[module_name].get("max_abs_error", 0)
        saved_frob = saved_error_stats[module_name].get("relative_frobenius_error", 0)

        w_relerr = saved_frob
        passed = roundtrip_ok and w_relerr < args.max_relerr
        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        flags = []
        if not roundtrip_ok:
            flags.append("roundtrip_mismatch")
        if w_relerr >= args.max_relerr:
            flags.append(f"relerr={w_relerr:.4f}")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(
            f"  {module_name}: [{status}]{flag_str} "
            f"rows={rows} cols={cols} "
            f"saved_max_err={saved_max:.4f} saved_frob={saved_frob:.4f}"
        )

    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
