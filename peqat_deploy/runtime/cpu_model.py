import json
import os

import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_packed_weight(layer_dir: str) -> tuple:
    pack_meta_path = os.path.join(layer_dir, "pack_meta.json")
    with open(pack_meta_path) as f:
        meta = json.load(f)

    weight_int4_path = os.path.join(layer_dir, "weight_int4.bin")
    with open(weight_int4_path, "rb") as f:
        packed_bytes = f.read()

    weight_scale_path = os.path.join(layer_dir, "weight_scale.fp16")
    with open(weight_scale_path, "rb") as f:
        scale_bytes = f.read()

    shape_path = os.path.join(layer_dir, "shape.json")
    with open(shape_path) as f:
        shape = json.load(f)

    rows, cols = shape["rows"], shape["cols"]
    scale = torch.frombuffer(scale_bytes, dtype=torch.float16).clone()

    packed = torch.frombuffer(packed_bytes, dtype=torch.uint8).clone()
    num_packed = (rows * cols + 1) // 2
    packed = packed[:num_packed]

    pack_order = meta.get("pack_order", "low_high")
    if pack_order == "low_high":
        low = packed & 0xF
        high = (packed >> 4) & 0xF
    else:
        low = (packed >> 4) & 0xF
        high = packed & 0xF

    interleaved = torch.stack([low, high], dim=1).flatten()
    w_q = interleaved[: rows * cols].reshape(rows, cols).to(torch.uint8)
    w_q = torch.where(w_q > 7, w_q - 16, w_q).to(torch.int8)

    return w_q, scale.float(), rows, cols


class PackedQuantizedLinear(nn.Module):
    def __init__(self, layer_dir: str, smooth: torch.Tensor, clip: torch.Tensor):
        super().__init__()
        w_q, scale, rows, cols = _load_packed_weight(layer_dir)
        self.register_buffer("w_q", w_q)
        self.register_buffer("scale", scale)
        self.register_buffer("smooth", smooth.float())
        self.register_buffer("clip", clip.float())
        self.in_features = cols
        self.out_features = rows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        smooth = self.smooth.to(dtype=dtype)
        clip = self.clip.to(dtype=dtype)
        scale = self.scale.to(dtype=dtype)
        w_dq = self.w_q.float().to(dtype=dtype) * scale.unsqueeze(1)
        x_smooth = x / smooth.unsqueeze(0)
        x_clipped = torch.clamp(x_smooth, -clip, clip)
        s_x = (clip / 7.0).unsqueeze(0)
        x_q = torch.clamp(torch.round(x_clipped / s_x), -8, 7)
        x_dq = x_q.to(dtype=dtype) * s_x
        return x_dq @ w_dq.T


def build_packed_model(
    export_dir: str,
    base_model_name: str,
    device: str = "cpu",
) -> tuple:
    print(f"Loading base model {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    smooth_dict = safetensors_load(os.path.join(export_dir, "activation", "smooth.safetensors"))
    clip_dict = safetensors_load(os.path.join(export_dir, "activation", "clip.safetensors"))
    weights_dir = os.path.join(export_dir, "weights")

    replacements = {}
    for name, module in model.named_modules():
        safe_name = f"base_model.model.{name}"
        if safe_name not in smooth_dict:
            continue
        layer_dir = os.path.join(weights_dir, safe_name.replace(".", "_"))
        if not os.path.isdir(layer_dir):
            continue
        smooth = smooth_dict[safe_name].float()
        clip = clip_dict[safe_name].float()
        replacements[name] = PackedQuantizedLinear(layer_dir, smooth, clip)

    for name, new_module in replacements.items():
        parent_path, child_name = name.rsplit(".", 1)
        parent = model
        for part in parent_path.split("."):
            parent = getattr(parent, part)
        setattr(parent, child_name, new_module)

    model = model.to(device)

    print(f"  Replaced {len(replacements)} target linear layers with packed quantized versions")
    return model, tokenizer, replacements


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 20, device: str = "cpu"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = inputs["input_ids"]

    for _ in range(max_new_tokens):
        logits = model(output_ids).logits
        next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        output_ids = torch.cat([output_ids, next_token], dim=-1)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
