import torch


def quantize_weights(smoothed_weights: dict) -> tuple:
    qmin = -8
    qmax = 7

    quantized = {}
    error_stats = {}

    for module_name, w in smoothed_weights.items():
        w_float = w.float()
        d, k = w_float.shape

        row_max = w_float.abs().amax(dim=1)
        scale = (row_max / qmax).clamp(min=1e-8)
        scale = scale.unsqueeze(1)

        w_scaled = w_float / scale
        w_q = torch.clamp(torch.round(w_scaled), qmin, qmax).to(torch.int8)
        w_dequant = w_q.float() * scale

        abs_err = (w_float - w_dequant).abs()
        max_err = abs_err.max().item()
        mean_err = abs_err.mean().item()
        frob_err = (w_float - w_dequant).norm().item() / w_float.norm().item()

        quantized[module_name] = {
            "qweight": w_q,
            "scale": scale.squeeze(1).half(),
        }

        error_stats[module_name] = {
            "max_abs_error": max_err,
            "mean_abs_error": mean_err,
            "relative_frobenius_error": frob_err,
        }

    return quantized, error_stats
