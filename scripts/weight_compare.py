import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_packed_weight(layer_dir: str) -> torch.Tensor:
    with open(os.path.join(layer_dir, "pack_meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(layer_dir, "weight_int4.bin"), "rb") as f:
        packed_bytes = f.read()
    with open(os.path.join(layer_dir, "weight_scale.fp16"), "rb") as f:
        scale_bytes = f.read()
    with open(os.path.join(layer_dir, "shape.json")) as f:
        shape = json.load(f)

    rows, cols = shape["rows"], shape["cols"]
    scale = torch.frombuffer(scale_bytes, dtype=torch.float16).clone()
    packed = torch.frombuffer(packed_bytes, dtype=torch.uint8).clone()
    packed = packed[: (rows * cols + 1) // 2]

    if meta["pack_order"] == "low_high":
        low = packed & 0xF
        high = (packed >> 4) & 0xF
    else:
        low = (packed >> 4) & 0xF
        high = packed & 0xF

    w_q = torch.stack([low, high], dim=1).flatten()[: rows * cols].reshape(rows, cols)
    w_q = torch.where(w_q.to(torch.uint8) > 7, w_q - 16, w_q).to(torch.int8)
    return w_q.float() * scale.unsqueeze(1).float()


def main():
    parser = argparse.ArgumentParser(
        description="Direct weight comparison: expected vs packed"
    )
    parser.add_argument(
        "export_dir",
        type=str,
        help="Path to export directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to original checkpoint dir (e.g., .../checkpoint-114)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (default: from manifest)",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=3,
        help="Number of layers to compare (default: 3, use -1 for all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device",
    )
    args = parser.parse_args()

    with open(os.path.join(args.export_dir, "manifest.json")) as f:
        manifest = json.load(f)
    base_model = args.base_model or manifest["base_model_name"]

    lora_rank = manifest.get("lora_rank", 32)
    lora_alpha = manifest.get("lora_alpha", 64)
    scaling = lora_alpha / lora_rank if lora_rank else 1.0
    target_modules = manifest.get("target_modules", [])

    weights_dir = os.path.join(args.export_dir, "weights")

    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.eval()

    # Load adapter weights
    adapter_weights = safetensors_load(
        os.path.join(args.checkpoint, "adapter_model.safetensors")
    )

    # Load quant params
    quant_params = torch.load(
        os.path.join(args.checkpoint, "quantization_params.pth"),
        map_location="cpu",
        weights_only=True,
    )

    # Group quant params by module name
    grouped_qp = {}
    for key, tensor in quant_params.items():
        for suffix, tag in [(".qat_core.activation_clip", "clip"),
                            (".qat_core.smooth_factor", "smooth")]:
            if key.endswith(suffix):
                mod_name = key[: -len(suffix)]
                grouped_qp.setdefault(mod_name, {})[tag] = tensor

    matched = 0
    total_err = 0.0
    layer_count = 0

    for name, module in model.named_modules():
        full_name = f"base_model.model.{name}"
        safe_name = full_name.replace(".", "_")
        layer_dir = os.path.join(weights_dir, safe_name)
        if not os.path.isdir(layer_dir):
            continue
        if not any(t in full_name for t in target_modules):
            continue

        layer_count += 1
        if args.max_layers != -1 and layer_count > args.max_layers:
            break

        # Get base weight
        if not hasattr(module, "weight"):
            continue
        w_base = module.weight.detach().clone()

        # Get LoRA delta
        lora_key_base = full_name
        lora_delta = None
        for wk in adapter_weights:
            if lora_key_base in wk and "lora_B" in wk:
                b_key = wk
                a_key = wk.replace("lora_B", "lora_A")
                if a_key in adapter_weights:
                    lora_delta = (
                        adapter_weights[b_key] @ adapter_weights[a_key]
                    ) * scaling
                    break

        if lora_delta is not None:
            w_eff = w_base.float() + lora_delta.float().to(w_base.device)
        else:
            w_eff = w_base.float()

        # Get smooth factor
        qp = grouped_qp.get(full_name, {})
        smooth = qp.get("smooth")
        if smooth is None:
            print(f"  {full_name}: no smooth factor found, skipping")
            continue

        # Expected weight: after smoothing fold (pre-quantization)
        w_expected = w_eff * smooth.to(w_eff.device).unsqueeze(0)

        # Actual weight: from packed export
        w_actual = _load_packed_weight(layer_dir)

        # Compare
        err = (w_expected - w_actual).abs()
        max_err = err.max().item()
        mean_err = err.mean().item()
        rel_err = err.norm().item() / w_expected.norm().item()

        matched += 1
        total_err += rel_err

        status = "OK" if rel_err < 0.5 else "HIGH"
        print(
            f"  [{status}] {full_name}: "
            f"rel_err={rel_err:.4f} max_err={max_err:.4f} mean_err={mean_err:.4f}"
        )

    print(f"\nCompared {matched} layers")
    if matched:
        print(f"Average relative error: {total_err / matched:.4f}")

    del model
    gc.collect()


if __name__ == "__main__":
    main()
