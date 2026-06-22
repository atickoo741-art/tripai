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
        self.register_buffer("smooth", smooth.contiguous().bfloat16())
        self.register_buffer("clip", clip.contiguous().bfloat16())
        self.out_features = out_features
        self.in_features = in_features

        w_packed_tc = self._repack_for_tc(packed_w, out_features, in_features)
        self.register_buffer("w_packed_tc", w_packed_tc)

        self._has_tc_kernel = hasattr(peqat_cuda, "fused_int4_tc_per_token_forward")
        self._has_gemv_kernel = hasattr(peqat_cuda, "fused_int4_gemv_per_token_forward")
        self._has_preprocess_per_token = hasattr(peqat_cuda, "preprocess_per_token_forward")

    @staticmethod
    def _unpack_int4(packed_bytes, rows, cols):
        packed = packed_bytes[: (rows * cols + 1) // 2]
        packed = packed.reshape(rows, -1)
        lo = (packed & 0xF).to(torch.int8)
        hi = ((packed >> 4) & 0xF).to(torch.int8)
        lo = torch.where(lo > 7, lo - 16, lo)
        hi = torch.where(hi > 7, hi - 16, hi)
        w = torch.empty(rows, cols, dtype=torch.int8, device=packed.device)
        w[:, ::2] = lo[:, : (cols + 1) // 2]
        w[:, 1::2] = hi[:, : cols // 2]
        return w

    @staticmethod
    def _repack_for_tc(packed_uint8, rows, cols):
        K = cols
        w_q = CUDAQuantizedLinear._unpack_int4(packed_uint8, rows, cols)
        w_i32 = w_q.to(torch.int32) & 0xF
        K8 = K // 8
        w_groups = w_i32[:, :K8 * 8].reshape(rows, K8, 8)
        packed = (
            w_groups[:, :, 0]
            | (w_groups[:, :, 1] << 4)
            | (w_groups[:, :, 2] << 8)
            | (w_groups[:, :, 3] << 12)
            | (w_groups[:, :, 4] << 16)
            | (w_groups[:, :, 5] << 20)
            | (w_groups[:, :, 6] << 24)
            | (w_groups[:, :, 7] << 28)
        )
        return packed.to(torch.int32)

    def _forward_tc(self, x_2d):
        M, K = x_2d.shape
        x_packed_8, s_x = peqat_cuda.preprocess_per_token_forward(
            x_2d, self.smooth, self.clip
        )

        N = self.out_features
        M_pad = (M + 15) // 16 * 16
        N_pad = (N + 63) // 64 * 64
        if M_pad > M or N_pad > N:
            x_packed_8 = torch.cat([
                x_packed_8,
                torch.zeros(M_pad - M, K // 8, dtype=torch.int32, device=x_packed_8.device),
            ], dim=0) if M_pad > M else x_packed_8
            s_x = torch.cat([
                s_x,
                torch.ones(M_pad - M, device=s_x.device),
            ], dim=0) if M_pad > M else s_x
            w_scale = self.w_scale.bfloat16()
            if N_pad > N:
                w_scale = torch.cat([
                    w_scale,
                    torch.ones(N_pad - N, device=w_scale.device),
                ], dim=0)
        else:
            w_scale = self.w_scale.bfloat16()

        y_padded = peqat_cuda.fused_int4_tc_per_token_forward(
            x_packed_8, self.w_packed_tc, w_scale.float(), s_x,
            N_pad, K,
        )

        y = y_padded[:M, :N].contiguous()
        return y

    def _forward_gemv(self, x_2d):
        M, K = x_2d.shape
        x_packed_8, s_x = peqat_cuda.preprocess_per_token_forward(
            x_2d, self.smooth, self.clip
        )

        y = peqat_cuda.fused_int4_gemv_per_token_forward(
            x_packed_8, self.w_packed_tc, self.w_scale.float(), s_x,
            self.out_features, K,
        )
        return y

    def _forward_bf16(self, x_2d):
        _, x_dq, _ = peqat_cuda.preprocess_forward_bf16(
            x_2d, self.smooth, self.clip
        )
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
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ndim = x.dim()
        if ndim == 3:
            B, S, K = x.shape
            x_2d = x.reshape(B * S, K)
        else:
            x_2d = x

        if self._has_gemv_kernel and self._has_preprocess_per_token and x_2d.size(0) == 1:
            y = self._forward_gemv(x_2d)
        elif self._has_tc_kernel and self._has_preprocess_per_token:
            y = self._forward_tc(x_2d)
        else:
            y = self._forward_bf16(x_2d)

        if ndim == 3:
            y = y.reshape(B, S, self.out_features)
        return y.to(x.dtype)


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
        dtype=torch.bfloat16,
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


class HadamardCUDAQuantizedLinear(nn.Module):
    """
    CUDA quantized linear for Hadamard PE-QAT.
    No smooth/clip buffers — Hadamard rotation is applied in the preprocess kernel.
    Weights are pre-rotated (H folded) at export time.
    """
    def __init__(
        self,
        packed_w: torch.Tensor,
        w_scale: torch.Tensor,
        out_features: int,
        in_features: int,
    ):
        super().__init__()
        self.register_buffer("packed_w", packed_w)
        self.register_buffer("w_scale", w_scale.bfloat16())
        self.out_features = out_features
        self.in_features = in_features

        w_packed_tc = CUDAQuantizedLinear._repack_for_tc(packed_w, out_features, in_features)
        self.register_buffer("w_packed_tc", w_packed_tc)

        self._has_tc_kernel = hasattr(peqat_cuda, "fused_int4_tc_per_token_forward")
        self._has_gemv_kernel = hasattr(peqat_cuda, "fused_int4_gemv_per_token_forward")
        self._has_preprocess_hadamard = hasattr(peqat_cuda, "preprocess_hadamard_forward")

    def _forward_tc(self, x_2d):
        M, K = x_2d.shape
        x_packed_8, s_x = peqat_cuda.preprocess_hadamard_forward(x_2d)

        N = self.out_features
        M_pad = (M + 15) // 16 * 16
        N_pad = (N + 63) // 64 * 64
        if M_pad > M or N_pad > N:
            x_padded = x_packed_8
            if M_pad > M:
                x_padded = torch.cat([
                    x_packed_8,
                    torch.zeros(M_pad - M, K // 8, dtype=torch.int32, device=x_packed_8.device),
                ], dim=0)
            s_x_padded = s_x
            if M_pad > M:
                s_x_padded = torch.cat([
                    s_x,
                    torch.ones(M_pad - M, device=s_x.device),
                ], dim=0)
            w_scale_bf16 = self.w_scale
            if N_pad > N:
                w_scale_bf16 = torch.cat([
                    self.w_scale,
                    torch.ones(N_pad - N, device=self.w_scale.device),
                ], dim=0)
        else:
            x_padded = x_packed_8
            s_x_padded = s_x
            w_scale_bf16 = self.w_scale

        y_padded = peqat_cuda.fused_int4_tc_per_token_forward(
            x_padded, self.w_packed_tc, w_scale_bf16.float(), s_x_padded,
            N_pad, K,
        )

        y = y_padded[:M, :N].contiguous()
        return y

    def _forward_gemv(self, x_2d):
        M, K = x_2d.shape
        x_packed_8, s_x = peqat_cuda.preprocess_hadamard_forward(x_2d)

        y = peqat_cuda.fused_int4_gemv_per_token_forward(
            x_packed_8, self.w_packed_tc, self.w_scale.float(), s_x,
            self.out_features, self.in_features,
        )
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ndim = x.dim()
        if ndim == 3:
            B, S, K = x.shape
            x_2d = x.reshape(B * S, K)
        else:
            x_2d = x

        if self._has_gemv_kernel and self._has_preprocess_hadamard and x_2d.size(0) == 1:
            y = self._forward_gemv(x_2d)
        elif self._has_tc_kernel and self._has_preprocess_hadamard:
            y = self._forward_tc(x_2d)
        else:
            raise RuntimeError("Hadamard PE-QAT requires fused_int4_tc_per_token_forward and preprocess_hadamard_forward")

        if ndim == 3:
            y = y.reshape(B, S, self.out_features)
        return y.to(x.dtype)


def load_hadamard_export(export_dir: str, device: str = "cuda") -> dict:
    with open(os.path.join(export_dir, "manifest.json")) as f:
        manifest = json.load(f)

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

        layers[module_name] = {
            "packed_w": packed_w.to(device),
            "w_scale": w_scale.to(device),
            "rows": shape["rows"],
            "cols": shape["cols"],
        }

    return {"layers": layers, "manifest": manifest}


def build_hadamard_cuda_model(
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
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(f"Loading Hadamard export artifacts from {export_dir}...")
    export = load_hadamard_export(export_dir, device=device)

    replacements = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        safe_name = f"base_model.model.{name}"
        if safe_name not in export["layers"]:
            continue

        layer = export["layers"][safe_name]
        replacements[name] = HadamardCUDAQuantizedLinear(
            packed_w=layer["packed_w"],
            w_scale=layer["w_scale"],
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
    print(f"  Replaced {len(replacements)} target linear layers with Hadamard CUDA quantized versions")
    return model, tokenizer
