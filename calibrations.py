# calibrations.py

import torch
from fake_quant import PEQATLoRAModule
from tqdm import tqdm  # Add this import at the top

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
    print("\n" + "="*60)
    print("Phase 1: Collecting raw activation/weight statistics")
    print("="*60)
    
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

    # Run calibration forward passes with progress bar
    with torch.no_grad():
        pbar = tqdm(
            enumerate(dataloader),
            total=steps,
            desc="Phase 1: Collecting stats",
            unit="batch",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        for batch_idx, batch in pbar:
            if batch_idx >= steps:
                break
            
            batch_on_device = {k: v.to(device) if torch.is_tensor(v) else v 
                             for k, v in batch.items()}
            _ = model(**batch_on_device)
            
            # Update progress bar with stats
            pbar.set_postfix({
                'modules': len(per_module_stats),
            })

    # Remove hooks
    for h in handles:
        h.remove()

    print(f"✓ Collected statistics from {len(per_module_stats)} modules\n")

    # Phase 2: Compute and apply smooth factors
    print("="*60)
    print("Phase 2: Computing smooth factors")
    print("="*60)
    
    smooth_factor_stats = []
    
    with torch.no_grad():
        for m in tqdm(
            model.modules(),
            desc="Computing smooth factors",
            unit="module",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
        ):
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
            
            smooth_factor_stats.append({
                'mean': s.mean().item(),
                'min': s.min().item(),
                'max': s.max().item()
            })
    
    # Print aggregate stats
    if smooth_factor_stats:
        avg_mean = sum(x['mean'] for x in smooth_factor_stats) / len(smooth_factor_stats)
        global_min = min(x['min'] for x in smooth_factor_stats)
        global_max = max(x['max'] for x in smooth_factor_stats)
        print(f"✓ Smooth factors computed: mean={avg_mean:.3f}, range=[{global_min:.3f}, {global_max:.3f}]\n")

    # Phase 3: Recalibrate activation clips on SCALED activations
    print("="*60)
    print("Phase 3: Calibrating activation clips on scaled activations")
    print("="*60)
    
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

    # Run calibration again with progress bar
    with torch.no_grad():
        pbar = tqdm(
            enumerate(dataloader),
            total=steps,
            desc="Phase 3: Calibrating clips",
            unit="batch",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        for batch_idx, batch in pbar:
            if batch_idx >= steps:
                break
            
            batch_on_device = {k: v.to(device) if torch.is_tensor(v) else v 
                             for k, v in batch.items()}
            _ = model(**batch_on_device)
            
            pbar.set_postfix({
                'clips_computed': len(per_module_scaled_pctl),
            })

    for h in handles:
        h.remove()

    # Apply calibrated clips
    clip_stats = []
    
    with torch.no_grad():
        for m in tqdm(
            model.modules(),
            desc="Applying activation clips",
            unit="module",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
        ):
            if not isinstance(m, PEQATLoRAModule):
                continue
            
            key = id(m)
            if key not in per_module_scaled_pctl:
                continue
            
            clip = per_module_scaled_pctl[key].to(m.qat_core.activation_clip.device)
            m.qat_core.activation_clip.data.copy_(clip)
            
            clip_stats.append({
                'mean': clip.mean().item(),
                'min': clip.min().item(),
                'max': clip.max().item()
            })
    
    # Print aggregate stats
    if clip_stats:
        avg_mean = sum(x['mean'] for x in clip_stats) / len(clip_stats)
        global_min = min(x['min'] for x in clip_stats)
        global_max = max(x['max'] for x in clip_stats)
        print(f"✓ Activation clips set: mean={avg_mean:.3f}, range=[{global_min:.3f}, {global_max:.3f}]")

    model.train()
    print("\n" + "="*60)
    print("✓ Unified calibration complete")
    print("="*60 + "\n")
