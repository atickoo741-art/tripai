import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def load_params(path):
    d = torch.load(path, map_location="cpu", weights_only=True)
    by_module = {}
    for k, v in d.items():
        parts = k.split(".")
        # base_model.model.model.layers.N.{attn|mlp}.{proj}.qat_core.{param}
        layer_idx = int(parts[3])
        proj = parts[4]
        param_name = parts[-1]
        key = (layer_idx, proj)
        by_module.setdefault(key, {})[param_name] = v.float().numpy()
    return by_module

def make_plots(params, out_dir="quant_diag"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    layers = sorted(set(k[0] for k in params))
    projs = sorted(set(k[1] for k in params))

    clip_all = np.concatenate([params[(l,p)]["activation_clip"] for l in layers for p in projs])
    smooth_all = np.concatenate([params[(l,p)]["smooth_factor"] for l in layers for p in projs])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Global Parameter Distributions", fontsize=14)

    axes[0,0].hist(clip_all, bins=200, density=True)
    axes[0,0].set_title(f"activation_clip (global)\nmean={clip_all.mean():.3f}  med={np.median(clip_all):.3f}")
    axes[0,0].set_xlabel("clip value")

    axes[0,1].hist(smooth_all, bins=200, density=True)
    axes[0,1].set_title(f"smooth_factor (global)\nmean={smooth_all.mean():.3f}  med={np.median(smooth_all):.3f}")
    axes[0,1].set_xlabel("smooth factor")

    axes[1,0].hist(clip_all, bins=200, density=True)
    axes[1,0].set_yscale("log")
    axes[1,0].set_title("activation_clip (log y)")

    axes[1,1].hist(smooth_all, bins=200, density=True)
    axes[1,1].set_yscale("log")
    axes[1,1].set_title("smooth_factor (log y)")

    plt.tight_layout()
    fig.savefig(f"{out_dir}/00_global_distributions.png", dpi=150)
    plt.close()

    clip_by_proj = {p: [] for p in projs}
    smooth_by_proj = {p: [] for p in projs}
    for l in layers:
        for p in projs:
            if (l,p) in params:
                clip_by_proj[p].append(params[(l,p)]["activation_clip"])
                smooth_by_proj[p].append(params[(l,p)]["smooth_factor"])

    fig, axes = plt.subplots(2, len(projs), figsize=(4*len(projs), 8))
    for i, p in enumerate(projs):
        arr = np.concatenate(clip_by_proj[p])
        axes[0,i].hist(arr, bins=100, density=True)
        axes[0,i].set_title(f"clip {p}\nμ={arr.mean():.3f} σ={arr.std():.3f}")
        arr2 = np.concatenate(smooth_by_proj[p])
        axes[1,i].hist(arr2, bins=100, density=True)
        axes[1,i].set_title(f"smooth {p}\nμ={arr2.mean():.3f} σ={arr2.std():.3f}")
    plt.tight_layout()
    fig.savefig(f"{out_dir}/01_by_projection.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for p in projs:
        means = []
        for l in layers:
            if (l,p) in params:
                means.append(params[(l,p)]["activation_clip"].mean())
        axes[0].plot(layers[:len(means)], means, marker='.', label=p)
    axes[0].set_title("activation_clip mean per layer by projection")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("mean clip")
    axes[0].legend()

    for p in projs:
        means = []
        for l in layers:
            if (l,p) in params:
                means.append(params[(l,p)]["smooth_factor"].mean())
        axes[1].plot(layers[:len(means)], means, marker='.', label=p)
    axes[1].set_title("smooth_factor mean per layer by projection")
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("mean smooth")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(f"{out_dir}/02_layer_trends.png", dpi=150)
    plt.close()

    layer_sample = layers[::4] if len(layers) > 8 else layers
    for p in projs:
        fig, axes = plt.subplots(2, len(layer_sample), figsize=(4*len(layer_sample), 6))
        for i, l in enumerate(layer_sample):
            if (l,p) in params:
                c = params[(l,p)]["activation_clip"]
                s = params[(l,p)]["smooth_factor"]
                axes[0,i].hist(c, bins=80, density=True)
                axes[0,i].set_title(f"L{l} {p} clip\nμ={c.mean():.2f}")
                axes[1,i].hist(s, bins=80, density=True)
                axes[1,i].set_title(f"L{l} {p} smooth\nμ={s.mean():.2f}")
        plt.tight_layout()
        fig.savefig(f"{out_dir}/03_{p}_layer_detail.png", dpi=150)
        plt.close()

    print(f"Plots saved to {out_dir}/")
    proj_keys = {p: [params[(l,p)] for l in layers if (l,p) in params] for p in projs}
    print(f"\n{'proj':<12} {'clip_mean':>12} {'clip_std':>12} {'clip_p99':>12} {'smooth_mean':>12} {'smooth_std':>12} {'smooth_p99':>12}")
    print("-"*84)
    for p in projs:
        if proj_keys[p]:
            c = np.concatenate([m["activation_clip"] for m in proj_keys[p]])
            s = np.concatenate([m["smooth_factor"] for m in proj_keys[p]])
            print(f"{p:<12} {c.mean():>12.4f} {c.std():>12.4f} {np.percentile(c,99):>12.4f} {s.mean():>12.4f} {s.std():>12.4f} {np.percentile(s,99):>12.4f}")
    total_trainable = sum(
        p.numel() for m in params.values()
        for pname in ["activation_clip", "smooth_factor"]
        if pname in m
    )
    print(f"\nTotal trainable QAT params: {total_trainable:,}")
    print(f"  activation_clip:  {sum(m['activation_clip'].size for m in params.values()):,}")
    print(f"  smooth_factor:    {sum(m['smooth_factor'].size for m in params.values()):,}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../peqat-deployment/output_qat_lora_llama2_7b_per_tok_ac_05fa_001/quantization_params.pth"
    params = load_params(path)
    make_plots(params, sys.argv[2] if len(sys.argv) > 2 else "quant_diag")
