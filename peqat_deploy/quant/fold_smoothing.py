import torch


def fold_smoothing(merged_weights: dict, quant_params: dict) -> dict:
    smoothed_weights = {}

    for module_name, w in merged_weights.items():
        smooth = quant_params[module_name]["smooth"]
        w = w.clone()
        w = w * smooth.unsqueeze(0)
        smoothed_weights[module_name] = w

    return smoothed_weights
