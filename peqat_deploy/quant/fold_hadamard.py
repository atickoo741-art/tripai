import math

import torch


def fast_hadamard(x):
    """Walsh-Hadamard along last dim, normalized sqrt(K). Same as training."""
    K = x.shape[-1]
    *rest, K = x.shape
    x_flat = x.reshape(-1, K)
    h = 1
    while h < K:
        x_flat = x_flat.reshape(-1, h * 2)
        a, b = x_flat[:, :h].clone(), x_flat[:, h:].clone()
        x_flat[:, :h] = a + b
        x_flat[:, h:] = a - b
        h *= 2
    x_flat /= math.sqrt(K)
    return x_flat.reshape(*rest, K) if rest else x_flat.reshape(K)


def fold_hadamard(merged_weights: dict) -> dict:
    """
    In-place: W' = H @ W along in_features dimension.
    
    Same H (Walsh-Hadamard, normalized) applied during training.
    This matches the weight rotation used in HadamardPEQATLoRAModule.forward.
    
    After this step, weights are ready for per-row INT4 quantization.
    No smooth or clip parameters needed — H handles outlier diffusion.
    """
    for module_name, w in merged_weights.items():
        with torch.no_grad():
            w.copy_(fast_hadamard(w))
    
    return merged_weights
