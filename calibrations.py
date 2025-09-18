import torch
# from train_qat import QATW4A4Linear
from fake_quant import QATW4A4Linear

def calibrate_activation_clips(model, dataloader, steps=200, percentile=99.5, device=None):
    """Initialize per-channel PACT clips using a short percentile calibration pass.
    Returns a dict mapping id(module) -> per-channel percentile tensor for use in SmoothQuant init.
    """
    model_device = next(model.parameters()).device
    device = device or model_device
    model.eval()

    # Collect per-module per-channel absolute percentiles via pre-hooks
    handles = []
    per_module_abs_pctl = {}

    def pre_hook(mod, inputs):
        if not inputs:
            return
        x = inputs[0]
        if not torch.is_tensor(x):
            return
        x = x.detach()
        # Flatten all but last dim (channel)
        x = x.abs().reshape(-1, x.shape[-1])
        q = torch.quantile(x, torch.tensor(percentile / 100.0, device=x.device), dim=0)
        key = id(mod)
        if key in per_module_abs_pctl:
            per_module_abs_pctl[key] = torch.maximum(per_module_abs_pctl[key], q)
        else:
            per_module_abs_pctl[key] = q

    for m in model.modules():
        if isinstance(m, QATW4A4Linear):
            handles.append(m.register_forward_pre_hook(pre_hook))

    # Run a few batches
    seen = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_on_device = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch_on_device[k] = v.to(device)
                else:
                    batch_on_device[k] = v
            _ = model(**batch_on_device)
            seen += 1
            if seen >= steps:
                break

    # Apply initialization
    for m in model.modules():
        if isinstance(m, QATW4A4Linear):
            key = id(m)
            if key not in per_module_abs_pctl:
                continue
            A = per_module_abs_pctl[key].to(m.activation_clip.device)
            # init_a = torch.clamp(A, min=0.5, max=6.0)
            with torch.no_grad():
                m.activation_clip.data.copy_(A)

    for h in handles:
        h.remove()
    model.train()
    return per_module_abs_pctl


def calibrate_smooth_factors(model, per_module_abs_pctl, alpha: float = 0.5):
    """Initialize per-channel SmoothQuant scaling factors using activation percentiles and weight stats."""
    model.eval()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, QATW4A4Linear):
                key = id(m)
                if key not in per_module_abs_pctl:
                    continue
                A = per_module_abs_pctl[key].to(m.smooth_factor.device)
                # Per-input-channel absmax of merged weights (before any smooth scaling)
                W = m._get_merged_weights().detach().abs().amax(dim=0)  # [in_features]
                s = (A.pow(alpha) / (W.pow(1.0 - alpha) + 1e-8)).clamp_(0.1, 10.0)
                m.smooth_factor.data.copy_(s)
    model.train()