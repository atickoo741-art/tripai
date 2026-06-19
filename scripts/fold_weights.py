import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from safetensors.torch import load_file as safetensors_load
from peqat_deploy.quant.fold_act_scale import fold_act_scale_into_weights


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Add folded (s_x-absorbed) INT4 weights to an existing export directory"
    )
    parser.add_argument("export_dir", type=str, help="Existing export directory")
    args = parser.parse_args()

    export_dir = args.export_dir
    manifest_path = os.path.join(export_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"Error: manifest.json not found in {export_dir}")
        return 1

    with open(manifest_path) as f:
        manifest = json.load(f)

    clip_path = os.path.join(export_dir, "activation", "clip.safetensors")
    if not os.path.isfile(clip_path):
        print(f"Error: clip.safetensors not found at {clip_path}")
        return 1
    clip_dict = safetensors_load(clip_path)

    packed = {}
    for module_name, art in manifest["weight_artifact_paths"].items():
        w4_path = os.path.join(export_dir, art["weight_int4"])
        ws_path = os.path.join(export_dir, art["weight_scale"])
        shape_path = os.path.join(export_dir, art["shape"])

        with open(w4_path, "rb") as f:
            w4_bytes = f.read()
        with open(ws_path, "rb") as f:
            ws_bytes = f.read()
        with open(shape_path) as f:
            shape = json.load(f)

        import torch
        ws_tensor = torch.frombuffer(bytearray(ws_bytes), dtype=torch.float16).clone()

        packed[module_name] = {
            "weight_int4_bytes": w4_bytes,
            "weight_scale": ws_tensor,
            "shape": (shape["rows"], shape["cols"]),
        }

    print(f"Loaded {len(packed)} modules from {export_dir}")
    print("Computing folded weights (absorbing s_x[k] into INT4 weights)...")
    packed_folded = fold_act_scale_into_weights(packed, clip_dict)
    print(f"Computed folded weights for {len(packed_folded)} modules")

    for module_name, fd in packed_folded.items():
        safe_name = module_name.replace(".", "_")
        layer_dir = os.path.join(export_dir, "weights", safe_name)

        w4_path = os.path.join(layer_dir, "weight_int4_folded.bin")
        with open(w4_path, "wb") as f:
            f.write(fd["weight_int4_bytes"])

        ws_path = os.path.join(layer_dir, "weight_scale_folded.bf16")
        with open(ws_path, "wb") as f:
            f.write(fd["weight_scale_bytes"])

    print(f"Saved folded weight files to {export_dir}/weights/*/")
    print("\nTo validate PPL with folded weights:")
    print(f"  python scripts/eval_ppl.py --mode export-folded {export_dir} --n_samples 141")
    print(f"(Where --mode export-folded loads folded weights via build_cuda_model(..., use_folded=True))")


if __name__ == "__main__":
    main()
