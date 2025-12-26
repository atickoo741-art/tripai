import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, Conv1D
from transformers.models.llama.modeling_llama import LlamaDecoderLayer  # Support Llama
from peft.peft_model import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
# from utils import Evaluation
from datasets import load_dataset
# from ppl_eval import Evaluation
import tqdm
from safetensors.torch import load_file
import json
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from transformers import DataCollatorForLanguageModeling
import itertools
import weakref
from transformers import TrainerCallback

from data_preparation import load_wikitext2_sliding_window, get_concatenated_text

from fake_quant import WeightSTEQuant, PEQATLoRAModule

# ============================================================
# FUSED OPERATIONS - JIT COMPILED
# ============================================================

@torch.jit.script
def _fused_lora_weight_quant_forward(
    base_weight: torch.Tensor,
    lora_B: torch.Tensor,
    lora_A: torch.Tensor,
    scaling: float,
    smooth_factor: torch.Tensor,
    num_bits: int
) -> torch.Tensor:
    """
    Fused operation: LoRA merge + SmoothQuant + Weight quantization
    
    Mathematical steps (MUST match unfused version):
    1. lora_delta = (lora_B @ lora_A) * scaling
    2. merged = base_weight + lora_delta
    3. scaled = merged * smooth_factor  (broadcast to [out, in])
    4. Quantize per-row symmetric
    """
    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1
    
    # Step 1-2: LoRA merge (fused)
    lora_delta = torch.mm(lora_B, lora_A) * scaling
    merged = base_weight + lora_delta
    
    # Step 3: SmoothQuant scaling
    scaled = merged * smooth_factor.unsqueeze(0)
    
    # Step 4: Per-row symmetric quantization
    row_max = scaled.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = row_max / qmax
    w_q = torch.clamp(torch.round(scaled / scale), qmin, qmax)
    w_dq = w_q * scale
    
    return w_dq


@torch.jit.script
def _fused_act_quant_forward(
    x: torch.Tensor,
    smooth_factor: torch.Tensor,
    clip_vals: torch.Tensor,
    num_bits: int,
    is_3d: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (x_dq, x_original, x_clipped, clip, scale)
    
    NOTE: x_original is UNSCALED (before smooth_factor division)
    This is important for backward pass mask computation.
    """
    qmax = (1 << (num_bits - 1)) - 1
    clip = clip_vals.abs().clamp(min=1e-4)
    scale = clip / qmax
    
    # Broadcasting
    if is_3d:
        smooth_b = smooth_factor.view(1, 1, -1)
        clip_b = clip.view(1, 1, -1)
        scale_b = scale.view(1, 1, -1)
    else:
        smooth_b = smooth_factor.view(1, -1)
        clip_b = clip.view(1, -1)
        scale_b = scale.view(1, -1)
    
    # Save ORIGINAL x (before scaling) for backward
    x_original = x
    
    # Step 1: SmoothQuant scaling
    x_scaled = x / smooth_b
    
    # Step 2: PACT clipping
    x_clipped = torch.clamp(x_scaled, -clip_b, clip_b)
    
    # Step 3: Quantization
    x_q = torch.round(x_clipped / scale_b)
    x_q = torch.clamp(x_q, -qmax, qmax)
    x_dq = x_q * scale_b
    
    return x_dq, x_original, x_clipped, clip, scale



class FusedActSTEQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, smooth_factor, clip_vals, num_bits):
        is_3d = (x.dim() == 3)
        
        x_dq, x_orig, x_clipped, clip, scale = _fused_act_quant_forward(
            x, smooth_factor, clip_vals, num_bits, is_3d
        )
        
        ctx.save_for_backward(x_orig, x_clipped, clip, scale, smooth_factor)
        ctx.qmax = (1 << (num_bits - 1)) - 1
        ctx.is_3d = is_3d
        
        return x_dq
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward with dtype handling for bfloat16 training.
        """
        x, x_clipped, clip, scale, smooth_factor = ctx.saved_tensors
        qmax = ctx.qmax
        is_3d = ctx.is_3d
        
        # CRITICAL: Ensure grad_output matches saved tensor dtype
        if grad_output.dtype != x.dtype:
            grad_output = grad_output.to(x.dtype)
        
        # Broadcast
        if is_3d:
            clip_b = clip.view(1, 1, -1)
            scale_b = scale.view(1, 1, -1)
            smooth_b = smooth_factor.view(1, 1, -1)
        else:
            clip_b = clip.view(1, -1)
            scale_b = scale.view(1, -1)
            smooth_b = smooth_factor.view(1, -1)
        
        # Compute scaled input
        x_scaled = x / smooth_b
        
        # Recompute quantization
        x_q = torch.round(x_clipped / scale_b)
        x_q = torch.clamp(x_q, -qmax, qmax)
        x_dq = x_q * scale_b
        
        # Masks
        mask_inside = (x_scaled.abs() <= clip_b)
        
        # 1. Gradient to INPUT x
        grad_x = grad_output * mask_inside.float() / smooth_b
        
        # 2. Gradient to SMOOTH_FACTOR
        grad_smooth_from_div = -(grad_output * mask_inside.float() * x / (smooth_b ** 2))
        grad_smooth_factor = grad_smooth_from_div.sum(dim=(0, 1)) if is_3d else grad_smooth_from_div.sum(dim=0)
        
        # 3. Gradient to CLIP VALUES (PACT)
        grad_clip_upper = (grad_output * (x_scaled > clip_b).float()).sum(dim=(0, 1))
        grad_clip_lower = (grad_output * (x_scaled < -clip_b).float() * -1.0).sum(dim=(0, 1))
        
        quant_error = x_dq - x_clipped
        grad_quant = (grad_output * quant_error * mask_inside.float() / (clip_b + 1e-8)).sum(dim=(0, 1))
        
        grad_alpha = grad_clip_upper + grad_clip_lower + grad_quant
        
        # Return: (x, smooth_factor, clip_vals, num_bits)
        return grad_x, grad_smooth_factor, grad_alpha, None


class FusedWeightSTEQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base_weight, lora_B, lora_A, scaling, smooth_factor, num_bits):
        w_dq = _fused_lora_weight_quant_forward(
            base_weight, lora_B, lora_A, scaling, smooth_factor, num_bits
        )
        
        # Save merged_weight BEFORE smooth scaling
        lora_delta = torch.mm(lora_B, lora_A) * scaling
        merged_weight = base_weight + lora_delta
        
        ctx.save_for_backward(lora_B, lora_A, merged_weight, smooth_factor)
        ctx.scaling = scaling
        
        return w_dq
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward with dtype handling for bfloat16 training.
        
        CRITICAL: All operations must happen in the same dtype as saved tensors.
        """
        lora_B, lora_A, merged_weight, smooth_factor = ctx.saved_tensors
        scaling = ctx.scaling
        
        # CRITICAL: Convert grad_output to match saved tensor dtype (bfloat16)
        target_dtype = lora_B.dtype
        if grad_output.dtype != target_dtype:
            grad_output = grad_output.to(target_dtype)
        
        # Gradient to smooth_factor: d(m*s)/ds = m
        # Sum over output dimension (dim=0)
        grad_smooth_factor = (grad_output * merged_weight).sum(dim=0)
        
        # Gradient to merged_weight: d(m*s)/dm = s
        # Broadcast smooth_factor: [in_features] -> [1, in_features]
        grad_merged = grad_output * smooth_factor.unsqueeze(0)
        
        # Gradient through LoRA: merged = base + lora_delta * scaling
        grad_lora_delta = grad_merged * scaling
        
        # Matrix multiplication backward (now both are same dtype)
        grad_lora_B = torch.mm(grad_lora_delta, lora_A.t())
        grad_lora_A = torch.mm(lora_B.t(), grad_lora_delta)
        
        # Gradients: (base_weight, lora_B, lora_A, scaling, smooth_factor, num_bits)
        return None, grad_lora_B, grad_lora_A, None, grad_smooth_factor, None




# ============================================================
# OPTIMIZED PEQAT MODULE WITH FUSED OPS
# ============================================================

class PEQATLoRAModuleFused(nn.Module):
    """
    Drop-in replacement for PEQATLoRAModule with fused operations.
    
    Usage:
        # After quantize_model()
        for name, module in model.named_modules():
            if isinstance(module, PEQATLoRAModule):
                parent = get_parent_module(model, name)
                setattr(parent, name.split('.')[-1], 
                       PEQATLoRAModuleFused(module))
    """
    
    def __init__(self, original_module):
        """Initialize from existing PEQATLoRAModule"""
        super().__init__()
        
        # Copy all attributes
        self.qat_core = original_module.qat_core
        self.lora_A = original_module.lora_A
        self.lora_B = original_module.lora_B
        self.scaling = original_module.scaling
        self.r = original_module.r
        self.lora_alpha = original_module.lora_alpha
        self.lora_dropout = original_module.lora_dropout
        self.active_adapter = original_module.active_adapter
        self.num_bits = original_module.num_bits
        
        # Copy optional attributes
        if hasattr(original_module, 'fan_in_fan_out'):
            self.fan_in_fan_out = original_module.fan_in_fan_out
        if hasattr(original_module, 'merged'):
            self.merged = original_module.merged
    
    def forward(self, x):
        """
        FUSED forward pass - 3 kernels instead of 18!
        
        Kernel 1: Fused weight path (LoRA + smooth + quant)
        Kernel 2: Fused activation path (smooth + PACT + quant)
        Kernel 3: Matmul
        """
        # Get base weight (frozen, so detach)
        base_weight = self.qat_core.get_base_weight().detach()
        
        # FUSED KERNEL 1: Weight quantization
        if self.r['default'] > 0 and not getattr(self, 'merged', False):
            lora_A = self.lora_A[self.active_adapter[0]].weight
            lora_B = self.lora_B[self.active_adapter[0]].weight
            scaling = self.scaling[self.active_adapter[0]]
            
            quant_weight = FusedWeightSTEQuant.apply(
                base_weight, lora_B, lora_A, scaling,
                self.qat_core.smooth_factor, self.num_bits
            )
        else:
            # No LoRA, just quantize base
            scaled = base_weight * self.qat_core.smooth_factor.unsqueeze(0)
            quant_weight = WeightSTEQuant.apply(scaled, self.num_bits)
        
        # FUSED KERNEL 2: Activation quantization
        quant_input = FusedActSTEQuant.apply(
            x,
            self.qat_core.smooth_factor,
            self.qat_core.activation_clip,
            self.num_bits
        )
        
        # KERNEL 3: Matmul
        bias = self.qat_core.original_linear.bias
        output = F.linear(quant_input, quant_weight, bias)
        
        return output
    
    def __repr__(self):
        return (f"PEQATLoRAModuleFused(in={self.qat_core.in_features}, "
                f"out={self.qat_core.out_features}, r={self.r['default']}, "
                f"num_bits={self.num_bits})")




# ============================================================
# CONVERSION UTILITY
# ============================================================

def convert_to_fused(model):
    """
    Convert all PEQATLoRAModule instances to fused version.
    """
    from fake_quant import QATW4A4LinearCore 
    
    converted = 0
    
    def get_parent_and_child(model, full_name):
        if '.' not in full_name:
            return model, full_name
        parts = full_name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]
    
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, PEQATLoRAModule):
            to_replace.append((name, module))
    
    for name, module in to_replace:
        parent, child_name = get_parent_and_child(model, name)
        fused_module = PEQATLoRAModuleFused(module)
        setattr(parent, child_name, fused_module)
        converted += 1
    
    print(f"✓ Converted {converted} modules to fused operations")
    
    # Verify parameters are accessible
    # verify_fused_parameters(model)
    
    return model



# def verify_fused_parameters(model):
#     """
#     Verify all parameters are correctly registered and accessible.
#     Call this after convert_to_fused() to ensure compatibility.
#     """
#     print("\n" + "="*70)
#     print("FUSED OPERATIONS PARAMETER VERIFICATION")
#     print("="*70)
    
#     lora_params = []
#     clip_params = []
#     smooth_params = []
#     other_params = []
    
#     for name, param in model.named_parameters():
#         if 'lora_A.default.weight' in name or 'lora_B.default.weight' in name:
#             lora_params.append((name, param.shape, param.requires_grad))
#         elif 'activation_clip' in name:
#             clip_params.append((name, param.shape, param.requires_grad))
#         elif 'smooth_factor' in name:
#             smooth_params.append((name, param.shape, param.requires_grad))
#         elif param.requires_grad:
#             other_params.append((name, param.shape))
    
#     print(f"\n📊 Parameter Count Summary:")
#     print(f"  LoRA parameters:    {len(lora_params)}")
#     print(f"  Clip parameters:    {len(clip_params)}")
#     print(f"  Smooth parameters:  {len(smooth_params)}")
#     print(f"  Other trainable:    {len(other_params)}")
    
#     # Verify all are trainable
#     lora_trainable = sum(1 for _, _, grad in lora_params if grad)
#     clip_trainable = sum(1 for _, _, grad in clip_params if grad)
#     smooth_trainable = sum(1 for _, _, grad in smooth_params if grad)
    
#     print(f"\n✓ Trainable Status:")
#     print(f"  LoRA:   {lora_trainable}/{len(lora_params)} trainable")
#     print(f"  Clip:   {clip_trainable}/{len(clip_params)} trainable")
#     print(f"  Smooth: {smooth_trainable}/{len(smooth_params)} trainable")
    
#     # Show first few of each type
#     if lora_params:
#         print(f"\n📌 Sample LoRA parameters (first 3):")
#         for name, shape, grad in lora_params[:3]:
#             print(f"  {name}")
#             print(f"    Shape: {shape}, Trainable: {grad}")
    
#     if clip_params:
#         print(f"\n📌 Sample Clip parameters (first 3):")
#         for name, shape, grad in clip_params[:3]:
#             print(f"  {name}")
#             print(f"    Shape: {shape}, Trainable: {grad}")
    
#     if smooth_params:
#         print(f"\n📌 Sample Smooth parameters (first 3):")
#         for name, shape, grad in smooth_params[:3]:
#             print(f"  {name}")
#             print(f"    Shape: {shape}, Trainable: {grad}")
    
#     # Check for QATW4A4LinearCore modules
#     qat_modules = []
#     for name, module in model.named_modules():
#         if isinstance(module, QATW4A4LinearCore):
#             qat_modules.append(name)
    
#     print(f"\n📌 QATW4A4LinearCore modules found: {len(qat_modules)}")
#     if qat_modules:
#         print(f"  First: {qat_modules[0]}")
#         print(f"  Last:  {qat_modules[-1]}")
    
#     # Check for PEQATLoRAModuleFused modules
#     fused_modules = []
#     for name, module in model.named_modules():
#         if isinstance(module, PEQATLoRAModuleFused):
#             fused_modules.append(name)
    
#     print(f"\n📌 PEQATLoRAModuleFused modules: {len(fused_modules)}")
    
#     # Warnings
#     if other_params:
#         print(f"\n⚠️  WARNING: Found {len(other_params)} unexpected trainable params:")
#         for name, shape in other_params[:5]:
#             print(f"  - {name} {shape}")
    
#     if len(lora_params) == 0:
#         print(f"\n❌ ERROR: No LoRA parameters found!")
#         return False
    
#     if len(clip_params) == 0:
#         print(f"\n❌ ERROR: No clip parameters found!")
#         return False
    
#     if len(smooth_params) == 0:
#         print(f"\n❌ ERROR: No smooth parameters found!")
#         return False
    
#     print("\n" + "="*70)
#     print("✅ ALL PARAMETERS CORRECTLY REGISTERED")
#     print("="*70 + "\n")
    
#     return True