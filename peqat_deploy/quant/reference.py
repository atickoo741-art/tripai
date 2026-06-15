import torch


def peqat_linear_reference(
    x: torch.Tensor,
    w: torch.Tensor,
    smooth: torch.Tensor,
    clip: torch.Tensor,
    num_bits: int = 4,
) -> torch.Tensor:
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -(1 << (num_bits - 1))

    smooth = smooth.to(dtype=x.dtype, device=x.device)
    clip = clip.to(dtype=x.dtype, device=x.device)

    x_smooth = x / smooth.unsqueeze(0)
    x_clipped = torch.clamp(x_smooth, -clip, clip)

    s_x = clip / qmax
    x_q = torch.clamp(torch.round(x_clipped / s_x), qmin, qmax)
    x_dq = x_q * s_x

    w_row_max = w.abs().amax(dim=1)
    s_w = (w_row_max / qmax).clamp(min=1e-8)
    w_q = torch.clamp(torch.round(w / s_w.unsqueeze(1)), qmin, qmax)
    w_dq = w_q * s_w.unsqueeze(1)

    y = x_dq @ w_dq.T
    return y
