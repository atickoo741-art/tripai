import os
import json
import argparse
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from train_qat import QATW4A4Linear


QMIN_4BIT = -(2 ** (4 - 1))
QMAX_4BIT = (2 ** (4 - 1)) - 1


class Int4LinearRuntime(nn.Module):
    """
    Runtime INT4 linear layer for inference:
    - Stores per-row 4-bit quantized weights (in int8 container) and per-row float scales
    - Folds per-input-channel activation scales into the weights during conversion, so runtime
      integer GEMM uses x_q (per-channel int4) and w_q (per-row int4), with a single per-row
      float rescale after matmul.
    """

    def __init__(self, weight_q: torch.Tensor, weight_scale: torch.Tensor, act_clip: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        assert weight_q.dtype == torch.int8
        assert weight_scale.dtype in (torch.float16, torch.float32, torch.bfloat16)
        assert act_clip.dtype in (torch.float16, torch.float32, torch.bfloat16)
        self.register_buffer("weight_q", weight_q)  # [out_features, in_features], int8 in [-8,7]
        self.register_buffer("weight_scale", weight_scale)  # [out_features]
        self.register_buffer("act_clip", act_clip)  # [in_features]
        if bias is not None:
            self.register_buffer("bias", bias.to(dtype=torch.float32))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize activations per-input-channel to int4 in int8 container
        # x shape: [batch, seq, in_features] or [N, in_features]
        x_shape = x.shape
        if x.dim() == 3:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x
        # Clamp to per-channel clip and quantize to INT4
        act_scale = (self.act_clip / max(QMAX_4BIT, 1)).clamp(min=1e-8)
        x_clamped = torch.clamp(x_2d, -self.act_clip, self.act_clip)
        x_q = torch.clamp(torch.round(x_clamped / act_scale), QMIN_4BIT, QMAX_4BIT).to(torch.int8)
        # Integer matmul: int32 accumulation
        y_int = x_q.to(torch.int32) @ self.weight_q.t().to(torch.int32)  # [N, out_features]
        # Dequantize using per-row weight scales (activation scales are folded into weights during conversion)
        y = y_int.to(torch.float32) * self.weight_scale.view(1, -1)
        if self.bias is not None:
            y = y + self.bias.view(1, -1)
        # Restore original shape
        if x.dim() == 3:
            y = y.view(x_shape[0], x_shape[1], -1)
        return y


def _fold_act_scales_into_weights(merged_weight: torch.Tensor, act_scale: torch.Tensor) -> torch.Tensor:
    """Scale each input column of weights by the corresponding activation scale.
    merged_weight: [out, in], act_scale: [in]
    returns: [out, in]
    """
    return merged_weight * act_scale.view(1, -1)


def _quantize_per_row_int4(weight_f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize float weights per row to symmetric INT4.
    Returns (weight_q int8 in [-8,7], row_scales float)
    """
    row_max = weight_f.abs().amax(dim=1)  # [out]
    # Avoid zero scales
    row_scale = (row_max / max(QMAX_4BIT, 1)).clamp(min=1e-8)
    w_q = torch.round(weight_f / row_scale.view(-1, 1)).clamp(QMIN_4BIT, QMAX_4BIT).to(torch.int8)
    return w_q, row_scale


def _try_load_quant_params(qat_dir: str, device: torch.device):
    qp_path = os.path.join(qat_dir, "quantization_params.pth")
    if os.path.isfile(qp_path):
        return torch.load(qp_path, map_location=device)
    # Try to find latest checkpoint directory containing quantization_params.pth
    if os.path.isdir(qat_dir):
        subdirs = [os.path.join(qat_dir, d) for d in os.listdir(qat_dir) if os.path.isdir(os.path.join(qat_dir, d))]
        subdirs.sort()
        for sd in reversed(subdirs):
            cand = os.path.join(sd, "quantization_params.pth")
            if os.path.isfile(cand):
                return torch.load(cand, map_location=device)
    return None


def replace_qat_with_int4_runtime(model: nn.Module, qat_dir: str | None = None) -> nn.Module:
    device = next(model.parameters()).device
    quant_params = _try_load_quant_params(qat_dir, device) if qat_dir else None
    for name, module in list(model.named_modules()):
        if isinstance(module, QATW4A4Linear):
            # Get merged base+LoRA weight in [out, in]
            with torch.no_grad():
                merged_w = module._get_merged_weights().to(torch.float32)
                # Prefer saved quant params if available
                if quant_params is not None:
                    prefix = f"{name}."
                    smooth = quant_params.get(prefix + "smooth_factor", None)
                    clip = quant_params.get(prefix + "activation_clip", None)
                else:
                    smooth = getattr(module, 'smooth_factor', None)
                    clip = getattr(module, 'activation_clip', None)

                if smooth is None:
                    smooth = torch.ones(merged_w.shape[1], device=device, dtype=torch.float32)
                else:
                    smooth = smooth.to(device=device, dtype=torch.float32).view(-1)

                if clip is None:
                    clip = torch.ones(merged_w.shape[1], device=device, dtype=torch.float32) * 2.0
                else:
                    clip = clip.to(device=device, dtype=torch.float32).abs().clamp(min=1e-6).view(-1)

                # Fold SmoothQuant scaling into weights
                w_fold = _fold_act_scales_into_weights(merged_w, smooth)
                # Effective activation clip in original input domain is s * a
                act_clip = (smooth * clip).to(torch.float32)
                bias = module.original_linear.bias.data if hasattr(module.original_linear, 'bias') else None
                # Quantize per-row to INT4
                w_q, row_scale = _quantize_per_row_int4(w_fold)

            # Build replacement module on same device
            int4_layer = Int4LinearRuntime(
                weight_q=w_q.to(merged_w.device),
                weight_scale=row_scale.to(merged_w.device),
                act_clip=act_clip.to(merged_w.device),
                bias=bias.to(merged_w.device) if bias is not None else None,
            )

            # Swap into parent
            parent = model
            parts = name.split('.')
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], int4_layer)
    return model


def load_qat_lora_model(model_name: str, qat_dir: str, device: torch.device) -> Tuple[nn.Module, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map=None)
    base = base.to(device)
    model = PeftModel.from_pretrained(base, qat_dir, torch_dtype=torch.float32)
    model = model.to(device)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--qat_dir', type=str, required=True, help='Directory with saved QAT LoRA adapters and quant params')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for INT4 runtime model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    # Load QAT LoRA model (with QATW4A4Linear wrappers expected present when applying for training)
    model, tokenizer = load_qat_lora_model(args.model_name, args.qat_dir, device)

    # Convert QATW4A4Linear layers to INT4 runtime layers (use saved quant params if present)
    model_int4 = replace_qat_with_int4_runtime(model, qat_dir=args.qat_dir)
    model_int4.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    # Save a lightweight state dict and config for later loading
    torch.save(model_int4.state_dict(), os.path.join(args.out_dir, 'int4_model_state.pt'))
    # Also save a full pickled model for direct loading in CLI chat
    torch.save(model_int4, os.path.join(args.out_dir, 'int4_model.pt'))
    tokenizer.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, 'int4_info.json'), 'w') as f:
        json.dump({
            'quant': 'W4A4',
            'weight_per_channel': 'per-row(out)',
            'act_per_channel': 'per-input folded into weights',
            'qmin': int(QMIN_4BIT),
            'qmax': int(QMAX_4BIT)
        }, f, indent=2)
    print(f"Saved INT4 runtime model to: {args.out_dir}")


if __name__ == '__main__':
    main()


