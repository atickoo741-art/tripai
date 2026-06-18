import os
import torch


def extract_quant_params(quant_params_path: str, merged_weights: dict) -> dict:
    if not os.path.isfile(quant_params_path):
        raise FileNotFoundError(f"quantization_params.pth not found: {quant_params_path}")

    raw = torch.load(quant_params_path, map_location="cpu", weights_only=True)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict in quantization_params.pth, got {type(raw)}")

    grouped = {}
    for key, tensor in raw.items():
        for suffix in (".qat_core.activation_clip", ".qat_core.smooth_factor"):
            if key.endswith(suffix):
                module_name = key[: -len(suffix)]
                if module_name not in grouped:
                    grouped[module_name] = {}
                if suffix == ".qat_core.activation_clip":
                    grouped[module_name]["clip"] = tensor
                elif suffix == ".qat_core.smooth_factor":
                    grouped[module_name]["smooth"] = tensor

    normalized = {}
    for module_name in merged_weights:
        if module_name not in grouped:
            raise KeyError(
                f"Module '{module_name}' not found in quantization_params.pth. "
                f"Available module keys (first 10): {sorted(grouped.keys())[:10]}"
            )
        entry = grouped[module_name]
        smooth = entry.get("smooth")
        clip = entry.get("clip")
        if smooth is None or clip is None:
            raise KeyError(
                f"Module '{module_name}' missing 'smooth' or 'clip' in quantization params. "
                f"Found keys: {list(entry.keys())}"
            )

        w = merged_weights[module_name]
        k_in = w.shape[1]
        if smooth.numel() != k_in:
            raise ValueError(
                f"Module '{module_name}': smooth size {smooth.numel()} != input dim {k_in}"
            )
        if clip.numel() != k_in:
            raise ValueError(
                f"Module '{module_name}': clip size {clip.numel()} != input dim {k_in}"
            )

        normalized[module_name] = {
            "smooth": smooth.detach().clone().bfloat16(),
            "clip": clip.detach().clone().bfloat16(),
        }

    return normalized
