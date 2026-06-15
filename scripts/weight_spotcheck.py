import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from safetensors.torch import load_file as safetensors_load
from transformers import AutoModelForCausalLM


def _unpack(layer_dir: str) -> torch.Tensor:
    with open(os.path.join(layer_dir, "pack_meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(layer_dir, "weight_int4.bin"), "rb") as f:
        raw = f.read()
    with open(os.path.join(layer_dir, "weight_scale.fp16"), "rb") as f:
        sraw = f.read()
    with open(os.path.join(layer_dir, "shape.json")) as f:
        shape = json.load(f)
    rows, cols = shape["rows"], shape["cols"]
    scale = torch.frombuffer(sraw, dtype=torch.float16).clone()
    packed = torch.frombuffer(raw, dtype=torch.uint8).clone()[: (rows * cols + 1) // 2]
    if meta["pack_order"] == "low_high":
        w = torch.stack([packed & 0xF, (packed >> 4) & 0xF], dim=1).flatten()
    else:
        w = torch.stack([(packed >> 4) & 0xF, packed & 0xF], dim=1).flatten()
    w = w[: rows * cols].reshape(rows, cols)
    w = torch.where(w.to(torch.uint8) > 7, w - 16, w).to(torch.int8)
    return w.float() * scale.unsqueeze(1).float(), scale


def main():
    parser = argparse.ArgumentParser(
        description="Spot-check a few layers: expected weight vs packed weight"
    )
    parser.add_argument("export_dir")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--layers", default="0,5,15,30", help="Layer indices to check")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(os.path.join(args.export_dir, "manifest.json")) as f:
        m = json.load(f)
    base_model = args.base_model or m["base_model_name"]
    lora_rank = m.get("lora_rank", 32)
    lora_alpha = m.get("lora_alpha", 64)
    scaling = lora_alpha / lora_rank if lora_rank else 1.0
    target_modules = m.get("target_modules", [])

    adapter_weights = safetensors_load(
        os.path.join(args.checkpoint, "adapter_model.safetensors")
    )
    qp = torch.load(
        os.path.join(args.checkpoint, "quantization_params.pth"),
        map_location="cpu", weights_only=True,
    )
    grouped_qp = {}
    for key, t in qp.items():
        for suffix, tag in [(".qat_core.activation_clip", "clip"), (".qat_core.smooth_factor", "smooth")]:
            if key.endswith(suffix):
                grouped_qp.setdefault(key[: -len(suffix)], {})[tag] = t

    layer_indices = [int(x) for x in args.layers.split(",")]

    print("Loading base model (once)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.eval()

    for li in layer_indices:
        print(f"\n{'='*70}")
        print(f"Layer {li}")
        print(f"{'='*70}")
        for tm in target_modules:
            name = f"model.layers.{li}.self_attn.{tm}" if tm in ("q_proj","k_proj","v_proj","o_proj") else f"model.layers.{li}.mlp.{tm}"
            full_name = f"base_model.model.{name}"
            safe_name = full_name.replace(".", "_")
            layer_dir = os.path.join(args.export_dir, "weights", safe_name)
            if not os.path.isdir(layer_dir):
                continue

            # Original weight
            parts = name.split(".")
            mod = model
            for p in parts:
                mod = getattr(mod, p) if hasattr(mod, p) else mod[int(p)]
            w_base = mod.weight.detach().clone().float()

            # LoRA delta
            lora_B_key = f"{full_name}.lora_B.weight"
            lora_A_key = f"{full_name}.lora_A.weight"
            if lora_B_key in adapter_weights and lora_A_key in adapter_weights:
                delta = (adapter_weights[lora_B_key].float() @ adapter_weights[lora_A_key].float()) * scaling
                w_eff = w_base + delta
            else:
                w_eff = w_base

            # Smooth
            smooth = grouped_qp.get(full_name, {}).get("smooth", None)
            if smooth is None:
                print(f"  {full_name}: no smooth, skip"); continue

            # Expected: merged + smoothing folded
            w_expected = w_eff * smooth.unsqueeze(0)

            # Packed
            w_actual, scale = _unpack(layer_dir)

            d, k = w_expected.shape
            rel_err = (w_expected - w_actual).norm().item() / w_expected.norm().item()

            print(f"\n  {full_name} [{d}x{k}]")
            print(f"  scale range: [{scale.min().item():.4f}, {scale.max().item():.4f}]")
            print(f"  rel_err: {rel_err:.6f}")
            print(f"\n  First 4 output rows, first 8 columns:")
            print(f"  {'Expected':>30s}  {'Packed':>30s}  {'Diff':>30s}")
            for r in range(min(4, d)):
                e_row = " ".join(f"{v:8.4f}" for v in w_expected[r, :8])
                a_row = " ".join(f"{v:8.4f}" for v in w_actual[r, :8])
                d_row = " ".join(f"{v:8.4f}" for v in (w_expected[r,:8] - w_actual[r,:8]))
                print(f"  row{r}: {e_row}  {a_row}  {d_row}")

    del model; gc.collect()

if __name__ == "__main__":
    main()
