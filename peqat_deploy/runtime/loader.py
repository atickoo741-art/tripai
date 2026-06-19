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
        packed_w_folded: torch.Tensor | None = None,
        w_scale_folded: torch.Tensor | None = None,
    ):
        super().__init__()
        self.register_buffer("packed_w", packed_w)
        self.register_buffer("w_scale", w_scale)
        self.register_buffer("smooth", smooth)
        self.register_buffer("clip", clip)
        self.out_features = out_features
        self.in_features = in_features

        self._fold_available = packed_w_folded is not None
        if packed_w_folded is not None:
            self.register_buffer("packed_w_folded", packed_w_folded)
            self.register_buffer("w_scale_folded", w_scale_folded)
            self._fold_w_new_q: torch.Tensor | None = None

    def _get_folded_weights(self) -> torch.Tensor:
        if self._fold_w_new_q is not None:
            return self._fold_w_new_q
        rows, K = self.out_features, self.in_features
        packed = self.packed_w_folded[:(rows * K + 1) // 2]
        packed = packed.reshape(rows, -1)
        lo = (packed & 0xF).to(torch.int8)
        hi = ((packed >> 4) & 0xF).to(torch.int8)
        lo = torch.where(lo > 7, lo - 16, lo)
        hi = torch.where(hi > 7, hi - 16, hi)
        w_q = torch.empty(rows, K if K % 2 == 0 else K + 1, dtype=torch.int8, device=packed.device)
        w_q[:, ::2] = lo
        w_q[:, 1::2] = hi
        if K % 2 != 0:
            w_q = w_q[:, :K]
        self._fold_w_new_q = w_q
        return w_q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ndim = x.dim()
        if ndim == 3:
            B, S, K = x.shape
            x_2d = x.reshape(B * S, K)
        else:
            x_2d = x

        if self._fold_available:
            x_q, _, _ = peqat_cuda.preprocess_forward_bf16(x_2d, self.smooth, self.clip)
            w_new_q = self._get_folded_weights()
            w_new_scale = self.w_scale_folded.bfloat16()
            M, N = x_q.size(0), self.out_features
            y = torch.matmul(x_q.float(), w_new_q.T.float()) * w_new_scale.unsqueeze(0).float()
            y = y.bfloat16()
        else:
            _, x_dq, _ = peqat_cuda.preprocess_forward_bf16(x_2d, self.smooth, self.clip)
            if x_2d.size(0) == 1:
                y = peqat_cuda.fused_gemv_bf16_forward(
                    x_dq, self.packed_w, self.w_scale.bfloat16(),
                    self.out_features, self.in_features,
                )
            else:
                y = peqat_cuda.fused_gemm_bf16_wmma_forward(
                    x_dq, self.packed_w, self.w_scale.bfloat16(),
                    self.out_features, self.in_features,
                )

        if ndim == 3:
            y = y.reshape(B, S, self.out_features)
        return y.to(x.dtype)


def load_export(export_dir: str, device: str = "cuda", use_folded: bool = True) -> dict:
    with open(os.path.join(export_dir, "manifest.json")) as f:
        manifest = json.load(f)

    smooth = safetensors_load(
        os.path.join(export_dir, "activation", "smooth.safetensors"),
    )
    clip = safetensors_load(
        os.path.join(export_dir, "activation", "clip.safetensors"),
    )
    for k in smooth:
        smooth[k] = smooth[k].to(device).bfloat16()
        clip[k] = clip[k].to(device).bfloat16()

    layers = {}
    for module_name in manifest.get("weight_artifact_paths", {}):
        art = manifest["weight_artifact_paths"][module_name]
        safe_name = module_name.replace(".", "_")
        layer_dir = os.path.join(export_dir, "weights", safe_name)

        with open(os.path.join(layer_dir, "shape.json")) as f:
            shape = json.load(f)

        path = os.path.join(layer_dir, "weight_int4.bin")
        with open(path, "rb") as f:
            packed_w = torch.frombuffer(f.read(), dtype=torch.uint8).clone()
        packed_w = packed_w[:(shape["rows"] * shape["cols"] + 1) // 2]

        with open(os.path.join(layer_dir, "weight_scale.fp16"), "rb") as f:
            w_scale = torch.frombuffer(f.read(), dtype=torch.float16).clone()

        entry = {
            "packed_w": packed_w.to(device),
            "w_scale": w_scale.to(device),
            "rows": shape["rows"],
            "cols": shape["cols"],
        }

        folded_path = os.path.join(layer_dir, "weight_int4_folded.bin")
        folded_scale_path = os.path.join(layer_dir, "weight_scale_folded.bf16")
        if use_folded and os.path.isfile(folded_path) and os.path.isfile(folded_scale_path):
            with open(folded_path, "rb") as f:
                pw_folded = torch.frombuffer(f.read(), dtype=torch.uint8).clone()
            pw_folded = pw_folded[:(shape["rows"] * shape["cols"] + 1) // 2]
            with open(folded_scale_path, "rb") as f:
                ws_folded = torch.frombuffer(f.read(), dtype=torch.float16).clone()
            entry["packed_w_folded"] = pw_folded.to(device)
            entry["w_scale_folded"] = ws_folded.to(device)

        layers[module_name] = entry

    return {"smooth": smooth, "clip": clip, "layers": layers, "manifest": manifest}


def build_cuda_model(
    export_dir: str,
    base_model_name: str,
    device: str = "cuda",
    use_folded: bool = False,
) -> tuple:
    print(f"Loading base model {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(f"Loading export artifacts from {export_dir}...")
    export = load_export(export_dir, device=device, use_folded=use_folded)
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
            packed_w_folded=layer.get("packed_w_folded"),
            w_scale_folded=layer.get("w_scale_folded"),
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
