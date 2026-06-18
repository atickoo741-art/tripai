import torch


def quantize_weights(smoothed_weights: dict) -> tuple:
    qmin = -8
    qmax = 7

    quantized = {}
    error_stats = {}

    for module_name, w in smoothed_weights.items():
        w_bf16 = w.bfloat16()
        d, k = w_bf16.shape

        row_max = w_bf16.abs().amax(dim=1)
        scale = (row_max / qmax).clamp(min=1e-8)
        scale = scale.unsqueeze(1)

        w_scaled = w_bf16 / scale
        w_q = torch.clamp(torch.round(w_scaled), qmin, qmax).to(torch.int8)
        w_dequant = w_q.float() * scale.float()

        abs_err = (w_bf16.float() - w_dequant.float()).abs()
        max_err = abs_err.max().item()
        mean_err = abs_err.mean().item()
        frob_err = (w_bf16.float() - w_dequant).norm().item() / w_bf16.float().norm().item()

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
