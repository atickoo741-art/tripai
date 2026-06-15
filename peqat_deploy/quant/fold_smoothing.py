import torch


def fold_smoothing(merged_weights: dict, quant_params: dict) -> dict:
    for module_name, w in merged_weights.items():
        smooth = quant_params[module_name]["smooth"]
        w.mul_(smooth.unsqueeze(0))

    return merged_weights
