import json
import os

import torch
import torch.nn as nn


def _get_linear_weight(module) -> torch.Tensor | None:
    if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
        return module.weight
    return None


def _is_target_module(name: str, target_modules: list) -> bool:
    return any(tm in name for tm in target_modules)


def merge_lora_weights(
    peft_model: nn.Module,
    target_modules: list,
    lora_rank: int,
    lora_alpha: int,
    debug_layer: str | None = None,
) -> dict:
    scaling = lora_alpha / lora_rank if lora_rank > 0 else 1.0
    merged_weights = {}
    debug_saved = False

    for name, module in peft_model.named_modules():
        if not _is_target_module(name, target_modules):
            continue

        base_weight = _get_linear_weight(module)
        if base_weight is None:
            continue

        lora_A = getattr(module, "lora_A", None)
        lora_B = getattr(module, "lora_B", None)
        if lora_A is None or lora_B is None:
            continue

        lora_A_weight = lora_A.default.weight
        lora_B_weight = lora_B.default.weight
        lora_delta = (lora_B_weight @ lora_A_weight) * scaling
        w_eff = base_weight + lora_delta.to(dtype=base_weight.dtype, device=base_weight.device)

        merged_weights[name] = w_eff.detach().clone().to(torch.bfloat16).cpu()

        if debug_layer and debug_layer in name and not debug_saved:
            debug_path = f"debug_{name.replace('.', '_')}_merged_fp16.pt"
            torch.save(merged_weights[name], debug_path)
            print(f"  Saved debug merged weights: {debug_path}")
            debug_saved = True

    return merged_weights


def build_layer_map(merged_weights: dict, export_dir: str) -> dict:
    layer_map = {}
    for module_name in merged_weights:
        safe_name = module_name.replace(".", "_")
        artifact_dir = os.path.join(export_dir, "weights", safe_name)
        layer_map[module_name] = {
            "artifact_dir": artifact_dir,
            "weight_int4": os.path.join(artifact_dir, "weight_int4.bin"),
            "weight_scale": os.path.join(artifact_dir, "weight_scale.fp16"),
            "shape": os.path.join(artifact_dir, "shape.json"),
            "pack_meta": os.path.join(artifact_dir, "pack_meta.json"),
        }
    return layer_map
