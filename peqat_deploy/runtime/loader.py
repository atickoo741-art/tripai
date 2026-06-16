import json
import os

import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load
from transformers import AutoModelForCausalLM, AutoTokenizer

import peqat_cuda


class CUDAQuantizedLinear(nn.Module):
    def __init__(
        self,
        packed_w: torch.Tensor,
        w_scale: torch.Tensor,
        smooth: torch.Tensor,
        clip: torch.Tensor,
        out_features: int,
        in_features: int,
    ):
        super().__init__()
        self.register_buffer("packed_w", packed_w)
        self.register_buffer("w_scale", w_scale)
        self.register_buffer("smooth", smooth)
        self.register_buffer("clip", clip)
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        ndim = x_f32.dim()
        if ndim == 3:
            B, S, K = x_f32.shape
            x_2d = x_f32.reshape(B * S, K)
        else:
            x_2d = x_f32
        x_dq = peqat_cuda.preprocess_forward(x_2d, self.smooth, self.clip)[1]
        y = peqat_cuda.linear_forward(
            x_dq, self.packed_w, self.w_scale, self.out_features, self.in_features,
        )
        if ndim == 3:
            y = y.reshape(B, S, self.out_features)
        return y


def load_export(export_dir: str, device: str = "cuda") -> dict:
    with open(os.path.join(export_dir, "manifest.json")) as f:
        manifest = json.load(f)

    smooth = safetensors_load(
        os.path.join(export_dir, "activation", "smooth.safetensors"),
    )
    clip = safetensors_load(
        os.path.join(export_dir, "activation", "clip.safetensors"),
    )
    for k in smooth:
        smooth[k] = smooth[k].to(device).float()
        clip[k] = clip[k].to(device).float()

    layers = {}
    for module_name in manifest.get("weight_artifact_paths", {}):
        safe_name = module_name.replace(".", "_")
        layer_dir = os.path.join(export_dir, "weights", safe_name)

        with open(os.path.join(layer_dir, "shape.json")) as f:
            shape = json.load(f)

        with open(os.path.join(layer_dir, "weight_int4.bin"), "rb") as f:
            packed_bytes = f.read()
        packed_w = torch.frombuffer(packed_bytes, dtype=torch.uint8).clone()
        packed_w = packed_w[:(shape["rows"] * shape["cols"] + 1) // 2]

        with open(os.path.join(layer_dir, "weight_scale.fp16"), "rb") as f:
            w_scale = torch.frombuffer(f.read(), dtype=torch.float16).clone()

        layers[module_name] = {
            "packed_w": packed_w.to(device),
            "w_scale": w_scale.to(device),
            "rows": shape["rows"],
            "cols": shape["cols"],
        }

    return {"smooth": smooth, "clip": clip, "layers": layers, "manifest": manifest}


def build_cuda_model(
    export_dir: str,
    base_model_name: str,
    device: str = "cuda",
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

    print(f"Loading export artifacts from {export_dir}...")
    export = load_export(export_dir, device=device)
    smooth, clip = export["smooth"], export["clip"]

    replacements = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        safe_name = f"base_model.model.{name}"
        if safe_name not in smooth:
            continue

        layer = export["layers"][safe_name]
        replacements[name] = CUDAQuantizedLinear(
            packed_w=layer["packed_w"],
            w_scale=layer["w_scale"],
            smooth=smooth[safe_name],
            clip=clip[safe_name],
            out_features=layer["rows"],
            in_features=layer["cols"],
        )

    for name, new_module in replacements.items():
        parent_path, child_name = name.rsplit(".", 1)
        parent = model
        for part in parent_path.split("."):
            parent = getattr(parent, part)
        setattr(parent, child_name, new_module)

    model = model.to(device)
    print(f"  Replaced {len(replacements)} target linear layers with CUDA quantized versions")
    return model, tokenizer
