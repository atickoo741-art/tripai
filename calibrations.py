import torch
# from train_qat import QATW4A4Linear
from fake_quant import PEQATLoRAModule


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
        if isinstance(m, PEQATLoRAModule):
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
        if isinstance(m, PEQATLoRAModule):
            key = id(m)
            if key not in per_module_abs_pctl:
                continue
            A = per_module_abs_pctl[key].to(m.qat_core.activation_clip.device)
            # init_a = torch.clamp(A, min=0.5, max=6.0)
            with torch.no_grad():
                m.qat_core.activation_clip.data.copy_(A)

    for h in handles:
        h.remove()
    model.train()
    return per_module_abs_pctl


def calibrate_smooth_factors(model, per_module_abs_pctl, alpha: float = 0.5):
    """Initialize per-channel SmoothQuant scaling factors using activation percentiles and weight stats."""
    model.eval()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, PEQATLoRAModule):
                key = id(m)
                if key not in per_module_abs_pctl:
                    continue
                A = per_module_abs_pctl[key].to(m.qat_core.smooth_factor.device)
                # Per-input-channel absmax of merged weights (before any smooth scaling)
                W = m.get_merged_weight().detach().abs().amax(dim=0)  # [in_features]
                s = (A.pow(alpha) / (W.pow(1.0 - alpha) + 1e-8))
                m.qat_core.smooth_factor.data.copy_(s)
    model.train()

def unified_calibration(model, dataloader, steps=200, percentile=99.5, alpha=0.5, device=None):
    """
    Joint calibration that ensures activation clips and smooth factors are compatible.
    
    Process:
    1. Collect raw activation and weight statistics
    2. Compute smooth factors from raw stats
    3. Apply smooth factors
    4. Compute activation clips from SCALED activation range
    """
    model_device = next(model.parameters()).device
    device = device or model_device
    model.eval()

    # Phase 1: Collect raw statistics
    print("Phase 1: Collecting raw activation/weight statistics...")
    handles = []
    per_module_stats = {}  # {id(module): {'act_pctl': tensor, 'weight_max': tensor}}

    def pre_hook(mod, inputs):
        if not inputs:
            return
        x = inputs[0]
        if not torch.is_tensor(x):
            return
        x = x.detach()
        
        # Raw activation statistics
        x_flat = x.abs().reshape(-1, x.shape[-1])
        q = torch.quantile(x_flat, torch.tensor(percentile / 100.0, device=x.device), dim=0)
        
        key = id(mod)
        if key in per_module_stats:
            per_module_stats[key]['act_pctl'] = torch.maximum(per_module_stats[key]['act_pctl'], q)
        else:
            # Get base weight statistics
            base_weight = mod.qat_core.get_base_weight().detach()
            W_max = base_weight.abs().amax(dim=0)
            per_module_stats[key] = {'act_pctl': q, 'weight_max': W_max}

    for m in model.modules():
        if isinstance(m, PEQATLoRAModule):
            handles.append(m.register_forward_pre_hook(pre_hook))

    # Run calibration forward passes
    seen = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_on_device = {k: v.to(device) if torch.is_tensor(v) else v 
                             for k, v in batch.items()}
            _ = model(**batch_on_device)
            seen += 1
            if seen >= steps:
                break

    # Remove hooks
    for h in handles:
        h.remove()

    # Phase 2: Compute and apply smooth factors
    print("Phase 2: Computing smooth factors...")
    with torch.no_grad():
        for m in model.modules():
            if not isinstance(m, PEQATLoRAModule):
                continue
            
            key = id(m)
            if key not in per_module_stats:
                continue
            
            A = per_module_stats[key]['act_pctl'].to(m.qat_core.smooth_factor.device)
            W = per_module_stats[key]['weight_max'].to(m.qat_core.smooth_factor.device)
            
            # SmoothQuant formula
            s = (A.pow(alpha) / (W.pow(1.0 - alpha) + 1e-8))
            m.qat_core.smooth_factor.data.copy_(s)
            
            print(f"  Smooth factor: mean={s.mean().item():.3f}, range=[{s.min().item():.3f}, {s.max().item():.3f}]")

    # Phase 3: Recalibrate activation clips on SCALED activations
    print("Phase 3: Calibrating activation clips on scaled activations...")
    per_module_scaled_pctl = {}
    
    def scaled_pre_hook(mod, inputs):
        if not inputs:
            return
        x = inputs[0]
        if not torch.is_tensor(x):
            return
        x = x.detach()
        
        # Apply smooth scaling
        if x.dim() == 3:
            s = mod.qat_core.smooth_factor.view(1, 1, -1)
        else:
            s = mod.qat_core.smooth_factor.view(1, -1)
        x_scaled = x / s
        
        # Get percentile of SCALED activations
        x_flat = x_scaled.abs().reshape(-1, x_scaled.shape[-1])
        q = torch.quantile(x_flat, torch.tensor(percentile / 100.0, device=x.device), dim=0)
        
        key = id(mod)
        if key in per_module_scaled_pctl:
            per_module_scaled_pctl[key] = torch.maximum(per_module_scaled_pctl[key], q)
        else:
            per_module_scaled_pctl[key] = q

    handles = []
    for m in model.modules():
        if isinstance(m, PEQATLoRAModule):
            handles.append(m.register_forward_pre_hook(scaled_pre_hook))

    # Run calibration again
    seen = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_on_device = {k: v.to(device) if torch.is_tensor(v) else v 
                             for k, v in batch.items()}
            _ = model(**batch_on_device)
            seen += 1
            if seen >= steps:
                break

    for h in handles:
        h.remove()

    # Apply calibrated clips
    with torch.no_grad():
        for m in model.modules():
            if not isinstance(m, PEQATLoRAModule):
                continue
            
            key = id(m)
            if key not in per_module_scaled_pctl:
                continue
            
            clip = per_module_scaled_pctl[key].to(m.qat_core.activation_clip.device)
            m.qat_core.activation_clip.data.copy_(clip)
            
            print(f"  Activation clip: mean={clip.mean().item():.3f}, range=[{clip.min().item():.3f}, {clip.max().item():.3f}]")

    model.train()
    print("✓ unified calibration complete\n")