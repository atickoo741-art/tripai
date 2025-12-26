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
    Fused operation: SmoothQuant + PACT + Activation quantization
    
    Mathematical steps (MUST match ActSTEQuant):
    1. x_scaled = x / smooth_factor
    2. x_clipped = clamp(x_scaled, -clip, clip)  [PACT]
    3. Quantize symmetric per-channel
    
    Returns: (x_dq, x, x_clipped, clip, scale) for backward pass
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
    
    # Step 1: SmoothQuant scaling
    x_scaled = x / smooth_b
    
    # Step 2: PACT clipping
    x_clipped = torch.clamp(x_scaled, -clip_b, clip_b)
    
    # Step 3: Quantization
    x_q = torch.round(x_clipped / scale_b)
    x_q = torch.clamp(x_q, -qmax, qmax)
    x_dq = x_q * scale_b
    
    return x_dq, x, x_clipped, clip, scale


# ============================================================
# FUSED AUTOGRAD FUNCTIONS WITH STE GRADIENTS
# ============================================================

class FusedActSTEQuant(torch.autograd.Function):
    """
    Fused activation quantization with STE gradients.
    
    Mathematically equivalent to: ActSTEQuant(x / smooth_factor, clip_vals)
    But executes in single kernel.
    """
    @staticmethod
    def forward(ctx, x, smooth_factor, clip_vals, num_bits):
        is_3d = (x.dim() == 3)
        
        # Use fused JIT operation
        x_dq, x_orig, x_clipped, clip, scale = _fused_act_quant_forward(
            x, smooth_factor, clip_vals, num_bits, is_3d
        )
        
        # Save for backward (same as ActSTEQuant)
        ctx.save_for_backward(x_orig, x_clipped, clip, scale)
        ctx.qmax = (1 << (num_bits - 1)) - 1
        ctx.is_3d = is_3d
        
        return x_dq
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        CRITICAL: Must match ActSTEQuant.backward() EXACTLY!
        This ensures learnable clip values receive correct gradients.
        """
        x, x_clipped, clip, scale = ctx.saved_tensors
        qmax = ctx.qmax
        is_3d = ctx.is_3d
        
        # Broadcast
        if is_3d:
            clip_b = clip.view(1, 1, -1)
            scale_b = scale.view(1, 1, -1)
        else:
            clip_b = clip.view(1, -1)
            scale_b = scale.view(1, -1)
        
        # Recompute quantization for gradient calc
        x_q = torch.round(x_clipped / scale_b)
        x_q = torch.clamp(x_q, -qmax, qmax)
        x_dq = x_q * scale_b
        
        # 1. Gradient to Input (STE)
        mask_inside = (x.abs() <= clip_b)
        grad_x = grad_output * mask_inside.float()
        
        # 2. Gradient to Clip (CRITICAL for PACT learning)
        # Component A: Clipped outliers
        grad_clip_upper = (grad_output * (x > clip_b).float()).sum(dim=(0, 1))
        grad_clip_lower = (grad_output * (x < -clip_b).float() * -1.0).sum(dim=(0, 1))
        
        # Component B: Quantization error
        mask_in_range = (x.abs() <= clip_b)
        quant_error = x_dq - x_clipped
        grad_quant = (grad_output * quant_error * mask_in_range.float() / (clip_b + 1e-8)).sum(dim=(0, 1))
        
        grad_alpha = grad_clip_upper + grad_clip_lower + grad_quant
        
        # No gradient to smooth_factor or num_bits
        return grad_x, None, grad_alpha, None


class FusedWeightSTEQuant(torch.autograd.Function):
    """
    Fused weight quantization with STE gradients.
    
    Mathematically equivalent to: WeightSTEQuant((base + LoRA) * smooth)
    But executes in single kernel.
    """
    @staticmethod
    def forward(ctx, base_weight, lora_B, lora_A, scaling, smooth_factor, num_bits):
        # Use fused JIT operation
        w_dq = _fused_lora_weight_quant_forward(
            base_weight, lora_B, lora_A, scaling, smooth_factor, num_bits
        )
        
        # No need to save anything (STE passes gradients through)
        return w_dq
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        STE: Gradient passes through unchanged to LoRA parameters.
        Base weight is frozen, so no gradient needed.
        """
        # grad_output flows to: base(frozen), lora_B, lora_A, scaling(constant), smooth, num_bits
        return None, grad_output, grad_output, None, None, None


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
    
    Usage:
        peft_model = quantize_model(peft_model)
        peft_model = convert_to_fused(peft_model)  # ← Add this
        trainer.train()
    """
    converted = 0
    
    def get_parent_and_child(model, full_name):
        """Helper to get parent module and child attribute name"""
        if '.' not in full_name:
            return model, full_name
        
        parts = full_name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]
    
    # Collect modules to replace (can't modify dict during iteration)
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, PEQATLoRAModule):
            to_replace.append((name, module))
    
    # Replace with fused versions
    for name, module in to_replace:
        parent, child_name = get_parent_and_child(model, name)
        fused_module = PEQATLoRAModuleFused(module)
        setattr(parent, child_name, fused_module)
        converted += 1
    
    print(f"✓ Converted {converted} modules to fused operations")
    return model
