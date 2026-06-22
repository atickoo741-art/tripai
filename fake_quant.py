import math
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



import glob
import re

# class ActSTEQuant(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, clip_vals, num_bits):
#         qmax = (1 << (num_bits - 1)) - 1
#         clip = clip_vals.abs().clamp(min=1e-4)
#         scale = clip / qmax
        
#         # Broadcasting
#         if x.dim() == 3:
#             clip_b = clip.view(1, 1, -1)
#             scale_b = scale.view(1, 1, -1)
#         else:
#             clip_b = clip.view(1, -1)
#             scale_b = scale.view(1, -1)
        
#         # PACT clipping
#         x_clipped = torch.clamp(x, -clip_b, clip_b)
        
#         # Quantize
#         x_q = torch.round(x_clipped / scale_b)
#         x_q = torch.clamp(x_q, -qmax, qmax)
#         x_dq = x_q * scale_b
        
#         # Save EVERYTHING we need for the full gradient
#         # ctx.save_for_backward(x, x_clipped, x_dq, clip, scale)
#         ctx.save_for_backward(x, x_clipped, clip, scale)
#         ctx.qmax = qmax
#         return x_dq
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         # x, x_clipped, x_dq, clip, scale = ctx.saved_tensors
#         x, x_clipped, clip, scale = ctx.saved_tensors
#         qmax = ctx.qmax
        
#         if x.dim() == 3:
#             clip_b = clip.view(1, 1, -1)
#             scale_b = scale.view(1, 1, -1)
#         else:
#             clip_b = clip.view(1, -1)
#             scale_b = scale.view(1, -1)
        
#         # 1. Gradient to Input (Standard STE)
#         mask_inside = (x.abs() <= clip_b)
#         grad_x = grad_output * mask_inside.float()
        
#         # 2. Gradient to Clip (THE OPTIMAL FORMULA)
#         # This accounts for BOTH clipping and quantization effects
        
#         grad_alpha = torch.zeros_like(clip)
        
#         # Component A: Gradient from CLIPPED values (Outliers)
#         # If x > α, output = α, so ∂output/∂α = 1
#         grad_clip_upper = (grad_output * (x > clip_b).float()).sum(dim=(0, 1))
#         grad_clip_lower = (grad_output * (x < -clip_b).float() * -1.0).sum(dim=(0, 1))
        
#         # Component B: Gradient from QUANTIZATION ERROR (All in-range values)
#         # The quantization error is (x_dq - x_clipped)
#         # This error is a function of the scale (α/qmax)
#         # Derivation: x_dq = round(x/s)*s where s=α/qmax
#         # For small changes in α, the change in x_dq is proportional to (x_dq - x)/α
#         # This captures: "If I increase α (bigger bins), does my reconstruction error get better or worse?"

#         x_q = torch.round(x_clipped / scale_b)
#         x_q = torch.clamp(x_q, -qmax, qmax)
#         x_dq = x_q * scale_b  # ← Recomputed here

#         mask_in_range = (x.abs() <= clip_b)
#         quant_error = x_dq - x_clipped  # The reconstruction error
        
#         # The gradient contribution from quantization error:
#         # ∂(x_dq)/∂α ≈ (x_dq - x_clipped) / α
#         # This tells α: "Adjust to minimize the reconstruction error"
#         clip_safe = torch.clamp(clip_b, min=0.1)
#         grad_quant = (grad_output * quant_error * mask_in_range.float() / (clip_safe)).sum(dim=(0, 1))
        
#         # COMBINE: Total gradient is clipping loss + quantization loss
#         grad_alpha = grad_clip_upper + grad_clip_lower + grad_quant
        
#         return grad_x, grad_alpha, None
#         # x, clip_b, scale_b = ctx.saved_tensors
#         # saved = ctx.saved_tensors
#         # x = saved[0]
#         # clip_b = saved[1]
    
#         # # Input gradient (unchanged)
#         # mask_inside = (x.abs() <= clip_b)
#         # grad_x = grad_output * mask_inside.float()
        
#         # # Clip gradient (simplified - ONLY boundary signals)
#         # grad_clip_upper = (grad_output * (x > clip_b).float()).sum(dim=(0, 1))
#         # grad_clip_lower = (grad_output * (x < -clip_b).float() * -1.0).sum(dim=(0, 1))
#         # grad_alpha = grad_clip_upper + grad_clip_lower
        
#         # return grad_x, grad_alpha, None, None


class ActSTEQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, clip_vals, num_bits, use_quant_grad=True):
        """
        Args:
            use_quant_grad: If True, use PE-QAT (boundary + quant). 
                           If False, use PACT-only (boundary only).
        """
        qmax = (1 << (num_bits - 1)) - 1
        clip = clip_vals.abs().clamp(min=1e-4)
        scale = clip / qmax
        
        # Broadcasting
        if x.dim() == 3:
            clip_b = clip.view(1, 1, -1)
            scale_b = scale.view(1, 1, -1)
        else:
            clip_b = clip.view(1, -1)
            scale_b = scale.view(1, -1)
        
        # PACT clipping
        x_clipped = torch.clamp(x, -clip_b, clip_b)
        
        # Quantize
        x_q = torch.round(x_clipped / scale_b)
        x_q = torch.clamp(x_q, -qmax, qmax)
        x_dq = x_q * scale_b
        
        # Save ONLY the 1D tensors (match original)
        ctx.save_for_backward(x, x_clipped, clip, scale)
        ctx.qmax = qmax
        ctx.use_quant_grad = use_quant_grad
        
        return x_dq
    
    @staticmethod
    def backward(ctx, grad_output):
        x, x_clipped, clip, scale = ctx.saved_tensors
        qmax = ctx.qmax
        use_quant_grad = ctx.use_quant_grad
        
        # Rebroadcast for operations
        if x.dim() == 3:
            clip_b = clip.view(1, 1, -1)
            scale_b = scale.view(1, 1, -1)
        else:
            clip_b = clip.view(1, -1)
            scale_b = scale.view(1, -1)
        
        # ============ INPUT GRADIENT (STE) ============
        mask_inside = (x.abs() <= clip_b)
        grad_x = grad_output * mask_inside.float()
        
        # ============ CLIP GRADIENT ============
        
        # Component 1: Boundary gradient (PACT-style, symmetric)
        grad_clip_upper = (grad_output * (x > clip_b).float()).sum(dim=(0, 1))
        grad_clip_lower = (grad_output * (x < -clip_b).float() * -1.0).sum(dim=(0, 1))
        
        # Component 2: Quantization scale gradient (LSQ-style)
        if use_quant_grad:
            # Recompute quantization (necessary for correct gradient)
            x_q = torch.round(x_clipped / scale_b)
            x_q = torch.clamp(x_q, -qmax, qmax)
            x_dq = x_q * scale_b
            
            mask_in_range = (x.abs() <= clip_b)
            quant_error = x_dq - x_clipped  # Reconstruction error ε
            
            # Use same clipping threshold as original (0.1, not 1e-4)
            clip_safe = torch.clamp(clip_b, min=0.1)
            grad_quant = (grad_output * quant_error * mask_in_range.float() / clip_safe).sum(dim=(0, 1))
            
            grad_alpha = grad_clip_upper + grad_clip_lower + grad_quant
        else:
            # PACT-only: ignore quantization gradient
            grad_alpha = grad_clip_upper + grad_clip_lower
        
        return grad_x, grad_alpha, None, None


class PerTokenActSTEQuant(torch.autograd.Function):
    """
    Per-token activation quantization with per-channel learned clipping.
    
    Forward:
        x: [M, K] or [B, S, K] — already smoothed (x/smooth applied upstream)
        clip_vals: [K] — per-channel learned clip values (PACT)
        num_bits: int (default 4)
        
        Per-token dynamic scale: s_x[t] = max|x_clipped| / qmax, detached.
        Gradient flows only through PACT boundary clip — no quant gradient.
    
    Backward:
        grad_x: STE — pass through for in-range values
        grad_alpha: PACT boundary gradient only, summed over batch/seq dims
    """
    @staticmethod
    def forward(ctx, x, clip_vals, num_bits=4):
        qmax = (1 << (num_bits - 1)) - 1
        clip = clip_vals.abs().clamp(min=1e-4)

        # Broadcast clip for PACT clamping
        if x.dim() == 3:
            clip_b = clip.view(1, 1, -1)
        else:
            clip_b = clip.view(1, -1)

        # Per-channel PACT clipping
        x_clipped = torch.clamp(x, -clip_b, clip_b)

        # Per-token dynamic scale — DETACHED (no gradient through scale computation)
        s_x = x_clipped.abs().amax(dim=-1, keepdim=True) / qmax
        s_x = s_x.detach()

        # Quantize-dequantize with per-token scale
        x_q = torch.round(x_clipped / s_x)
        x_q = torch.clamp(x_q, -qmax, qmax)
        x_dq = x_q * s_x

        ctx.save_for_backward(x, x_clipped, clip)
        ctx.qmax = qmax
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        x, x_clipped, clip = ctx.saved_tensors

        # Broadcast clip for element-wise operations
        if x.dim() == 3:
            clip_b = clip.view(1, 1, -1)
        else:
            clip_b = clip.view(1, -1)

        # Input gradient (STE): only in-range pass through
        mask_inside = (x.abs() <= clip_b).float()
        grad_x = grad_output * mask_inside

        # Clip gradient: PACT boundary only
        # Sum over batch/seq dims — handles both 2D and 3D
        reduce_dims = (0, 1) if x.dim() == 3 else (0,)
        grad_clip_upper = (grad_output * (x > clip_b).float()).sum(dim=reduce_dims)
        grad_clip_lower = (grad_output * (x < -clip_b).float() * -1.0).sum(dim=reduce_dims)
        grad_alpha = grad_clip_upper + grad_clip_lower

        return grad_x, grad_alpha, None


class WeightSTEQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, num_bits):
        # Symmetric per-row(out) scale from row max (detached); zero_point = 0
        qmin = -(1 << (num_bits - 1))
        qmax = (1 << (num_bits - 1)) - 1
        row_max = W.detach().abs().amax(dim=1, keepdim=True)
        scale = (row_max / max(qmax, 1)).clamp(min=1e-8)
        W_q = torch.clamp(torch.round(W / scale), qmin, qmax)
        W_dq = W_q * scale
        return W_dq

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class QATW4A4LinearCore(nn.Module):
    """Core quantization logic without LoRA awareness"""
    
    def __init__(self, original_linear_layer, num_bits=4, init_clip_val=2.0):
        super().__init__()
        self.original_linear = original_linear_layer
        self.num_bits = num_bits
        
        # Determine layer type and dimensions
        if hasattr(original_linear_layer, 'in_features') and hasattr(original_linear_layer, 'out_features'):
            self.in_features = original_linear_layer.in_features
            self.out_features = original_linear_layer.out_features
            self.weight_is_out_in = True
        else:
            # Conv1D case
            weight_shape = original_linear_layer.weight.shape
            self.in_features = weight_shape[0]
            self.out_features = weight_shape[1]
            self.weight_is_out_in = False

        self.device = original_linear_layer.weight.device

        param_dtype = original_linear_layer.weight.dtype
        # Trainable QAT parameters
        self.activation_clip = nn.Parameter(
            torch.ones(self.in_features, device=self.device, dtype=param_dtype) * init_clip_val
        )
        self.smooth_factor = nn.Parameter(
            torch.ones(self.in_features, device=self.device, dtype=param_dtype)
        )

    @property 
    def constrained_clip_val(self):
        """Ensure strictly positive clip values"""
        # return torch.clamp(torch.abs(self.activation_clip), min=1e-6)
        return self.activation_clip.clamp(min=1e-6)

    def get_base_weight(self):
        """Get base weight in correct orientation [out_features, in_features]"""
        base_weight = self.original_linear.weight
        if not self.weight_is_out_in:
            base_weight = base_weight.t()
        return base_weight


class PEQATLoRAModule(nn.Module):
    """
    Complete LoRA module replacement that handles:
    1. Base weights + LoRA merging
    2. SmoothQuant scaling
    3. PACT clipping
    4. W4A4 quantization with STE
    """
    
    def __init__(self, original_lora_module, num_bits=4, init_clip_val=2.0):
        super().__init__()
        
        # Wrap the base layer with QAT core
        self.qat_core = QATW4A4LinearCore(
            original_lora_module.base_layer,
            num_bits=num_bits,
            init_clip_val=init_clip_val
        )
        
        # Copy LoRA parameters from original module
        # These become part of this module's parameters
        # print(vars(original_lora_module))
        # print(original_lora_module.named_modules())
        self.lora_A = original_lora_module.lora_A
        # print(self.lora_A["default"].weight)
        self.lora_B = original_lora_module.lora_B
        # print(self.lora_B["default"].weight)
        self.scaling = original_lora_module.scaling
        
        # Copy LoRA configuration attributes
        self.r = original_lora_module.r
        self.lora_alpha = original_lora_module.lora_alpha
        self.lora_dropout = original_lora_module.lora_dropout
        self.active_adapter = original_lora_module.active_adapter
        
        # print("-------------------------------")
        # print(original_lora_module.active_adapter[0])
        
        # Copy other attributes that PEFT might expect
        if hasattr(original_lora_module, 'fan_in_fan_out'):
            self.fan_in_fan_out = original_lora_module.fan_in_fan_out
        if hasattr(original_lora_module, 'merged'):
            self.merged = original_lora_module.merged
        
        self.num_bits = num_bits
    
    def get_merged_weight(self):
        """Compute base_weight + lora_delta"""
        base_weight = self.qat_core.get_base_weight().detach()
        
        # Compute LoRA delta if not merged
        if self.r['default'] > 0 and not getattr(self, 'merged', False):
            lora_A = self.lora_A[self.active_adapter[0]].weight
            lora_B = self.lora_B[self.active_adapter[0]].weight
            scaling = self.scaling[self.active_adapter[0]]
            
            # LoRA contribution: (B @ A) * scaling
            lora_delta = (lora_B @ lora_A) * scaling
            merged_weight = base_weight + lora_delta
        else:
            merged_weight = base_weight
        
        return merged_weight
    
    def forward(self, x):
        """
        Forward pass with per-token quantization + per-channel learned clipping
        """
        # Get merged weights (base + LoRA)
        merged_weight = self.get_merged_weight()
        
        # Apply SmoothQuant scaling
        if x.dim() == 3:
            smooth_factor = self.qat_core.smooth_factor.view(1, 1, -1)
        else:
            smooth_factor = self.qat_core.smooth_factor.view(1, -1)
        
        scaled_input = x / smooth_factor
        scaled_weights = merged_weight * self.qat_core.smooth_factor.unsqueeze(0)
        
        # CHANGED: Pass activation_clip back in
        # activation_clip acts as learned upper bound for clipping
        clip_vals = self.qat_core.constrained_clip_val
        
        # Per-token quantization with per-channel learned clipping
        quant_input = PerTokenActSTEQuant.apply(scaled_input, clip_vals, self.num_bits)
        
        # Quantize weights
        quant_weight = WeightSTEQuant.apply(scaled_weights, self.num_bits)
        
        # Linear operation
        bias = self.qat_core.original_linear.bias
        output = F.linear(quant_input, quant_weight, bias)
        
        return output

    
    def __repr__(self):
        return (f"PEQATLoRAModule(in_features={self.qat_core.in_features}, "
                f"out_features={self.qat_core.out_features}, "
                f"r={self.r['default']}, num_bits={self.num_bits})")


class NaiveW4A4Linear(nn.Module):
    def __init__(self, original_linear_layer, num_bits=4):
        super().__init__()
        self.original_linear = original_linear_layer
        self.num_bits = num_bits
        # self.fake_quantize = FakeQuantize.apply
        # Support both nn.Linear (weight shape [out, in]) and GPT-2 Conv1D (weight shape [in, out])
        if hasattr(original_linear_layer, 'in_features') and hasattr(original_linear_layer, 'out_features'):
            # nn.Linear-like
            self.in_features = original_linear_layer.in_features
            self.out_features = original_linear_layer.out_features
            self.weight_is_out_in = True
        elif hasattr(original_linear_layer, 'weight'):
            # Conv1D-like (GPT-2)
            # print(original_linear_layer.weight)
            weight_shape = original_linear_layer.weight.shape
            # Conv1D stores weight as [in_features, out_features]
            self.in_features = weight_shape[0]
            self.out_features = weight_shape[1]
            self.weight_is_out_in = False
        

        self.device = next(original_linear_layer.parameters()).device

        W_base = self.original_linear.weight
        W_cpu = W_base.detach().to(torch.float32).cpu()

        self.min_base = W_cpu.min(dim=1).values    
        self.max_base = W_cpu.max(dim=1).values    

        self.min_base = self.min_base.to(device=self.device)
        self.max_base = self.max_base.to(device=self.device)


    @property
    def weight(self):
        # return self._get_merged_weights()
        return self.original_linear.weight if self.weight_is_out_in else self.original_linear.weight.T
        

    def forward(self, input):
        

        quant_input = WeightSTEQuant.apply(input, self.num_bits)
        quant_weight = WeightSTEQuant.apply(self.weight, self.num_bits)

        # If original layer is Conv1D (weights stored [in, out]), ensure quant_weight supplied to F.linear is [out, in]
            
        output = F.linear(quant_input, quant_weight, self.original_linear.bias)

        return output


def naive_quantize_model(model):
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module, (nn.Linear, Conv1D)):
            parent = model
            attrs = name.split('.')
            for attr in attrs[:-1]:
                parent = getattr(parent, attr)
            
            setattr(parent, attrs[-1], NaiveW4A4Linear(module))


def get_trainable_params(model):
    """
    Separate LoRA and QAT parameters for optimizer setup
    
    Returns:
        lora_params: List of LoRA adapter parameters
        qat_params: List of QAT parameters (activation_clip, smooth_factor)
    """
    lora_params = []
    qat_params = []
    
    for name, param in model.named_parameters():
        if any(x in name for x in ['lora_A', 'lora_B']):
            lora_params.append(param)
            param.requires_grad = True
        elif any(x in name for x in ['activation_clip', 'smooth_factor']):
            qat_params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False  # Freeze base model
    
    print(f"Found {len(lora_params)} LoRA parameters")
    print(f"Found {len(qat_params)} QAT parameters")
    
    return lora_params, qat_params


def quantize_peft_gpt2(peft_model, num_bits=4, init_clip_val=2.0):
    """
    Replace all LoRA modules in the model with PEQATLoRAModule
    
    Args:
        peft_model: PEFT model with LoRA adapters
        num_bits: Quantization bits (default 4)
        init_clip_val: Initial activation clip value (default 2.0)
    
    Returns:
        Modified model with quantized LoRA modules
    """
    modules_to_replace = []
    
    # Find all LoRA modules
    for name, module in peft_model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            modules_to_replace.append((name, module))
    
    print(f"Found {len(modules_to_replace)} LoRA modules to quantize")
    
    # Replace each LoRA module
    for name, lora_module in modules_to_replace:
        # print(f"Quantizing LoRA module: {name}")
        
        # Create quantized replacement
        quantized_module = PEQATLoRAModule(
            lora_module,
            num_bits=num_bits,
            init_clip_val=init_clip_val
        )
        
        # Navigate to parent and replace
        parent = peft_model
        attrs = name.split('.')
        for attr in attrs[:-1]:
            parent = getattr(parent, attr)
        setattr(parent, attrs[-1], quantized_module)
    
    return peft_model


# sorts models by model type and passes model on to respective model specific quantization function
def quantize_model(model, num_bits=4, init_clip_val=2.0):
    """Entry point for model quantization"""
    if isinstance(model, PeftModel):
        base = model.base_model.model
    else:
        base = model

    # if getattr(base.config, 'model_type', None) == 'gpt2':
    #     return quantize_peft_gpt2(model, num_bits=num_bits, init_clip_val=init_clip_val)
    # else:
    #     raise ValueError(f"Unsupported model type: {base.config.model_type}")

    return quantize_peft_gpt2(model, num_bits=num_bits, init_clip_val=init_clip_val)
    


def _find_quant_params_path(checkpoint_dir: str) -> str | None:
    cand = os.path.join(checkpoint_dir, "quantization_params.pth")
    if os.path.isfile(cand):
        return cand
    # Search subdirectories (e.g., checkpoint-XXXX)
    if os.path.isdir(checkpoint_dir):
        subdirs = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
        subdirs.sort()
        for sd in reversed(subdirs):
            f = os.path.join(sd, "quantization_params.pth")
            if os.path.isfile(f):
                return f
    return None


def load_peqat_model(adapter_dir: str, base_model_name: str = "gpt2", device: torch.device | None = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    print(f"Loading base model {base_model_name} and PEFT adapters from {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map=None).to(device)

    # 1. Load LoraConfig from adapter_config.json
    adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(adapter_config_path):
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    
    with open(adapter_config_path, 'r') as f:
        lora_config_dict = json.load(f)
    

    # print(lora_config_dict)
    # Create LoraConfig object
    lora_config = LoraConfig(**lora_config_dict)

    # 2. Create PeftModel structure (this injects LoRA layers into the base_model)
    peft_model = get_peft_model(base_model, lora_config)
   
    # safe_tensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    # safe_tensors_dict = load_file(safe_tensors_path)


    quantize_model(peft_model)

    safe_tensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if os.path.exists(safe_tensors_path):
        saved_state_dict = load_file(safe_tensors_path)
        peft_model.load_state_dict(saved_state_dict, strict=False)
        missing_keys, unexpected_keys = peft_model.load_state_dict(saved_state_dict, strict=False)
        
        # print(f"\n{'='*60}")
        # print("KEY REMAPPING")
        # print(f"{'='*60}")

        # Remap keys: add .default before .weight for lora_A and lora_B
        remapped_state_dict = {}
        for saved_key, value in saved_state_dict.items():
            if 'lora_A.weight' in saved_key or 'lora_B.weight' in saved_key:
                # Insert .default before .weight
                # From: base_model.model.transformer.h.0.attn.c_attn.lora_A.weight
                # To:   base_model.model.transformer.h.0.attn.c_attn.lora_A.default.weight
                runtime_key = saved_key.replace('.weight', '.default.weight')
                remapped_state_dict[runtime_key] = value
                # print(f"✓ {saved_key}")
                # print(f"  -> {runtime_key}")
            else:
                # Keep other keys as-is
                remapped_state_dict[saved_key] = value

        # Load the remapped state dict
        missing_keys, unexpected_keys = peft_model.load_state_dict(remapped_state_dict, strict=False)
        
        # Filter to only show LoRA-related issues
        lora_missing = [k for k in missing_keys if 'lora' in k.lower()]
        lora_unexpected = [k for k in unexpected_keys if 'lora' in k.lower()]
        
        print(f"{'='*60}\n")

        if missing_keys:
            print(f"⚠️  WARNING: Missing keys when loading LoRA: {lora_missing}")
        if unexpected_keys:
            print(f"⚠️  WARNING: Unexpected keys: {lora_unexpected[:5]}")
        
        print(f"✓ Loaded {len(saved_state_dict)} parameters from adapter_model.safetensors")
    else:
        raise FileNotFoundError(f"adapter_model.safetensors not found in {adapter_dir}")
    

    peft_model = peft_model.to(device)


    # for name, module in peft_model.named_modules():
    #     # print(name)
    #     if hasattr(module, 'weight') and ('lora_A.default' in name or 'lora_B.default' in name):
    #         print(name)
    #         print(module.weight)
            
    # 5. Load and apply additional quantization parameters from quantization_params.pth
    # This ensures all QAT parameters are correctly loaded, as adapter_model.safetensors might not contain all
    # of them or they might be nested differently. (Matches previous deprecated implementation behavior)
    qp_path = _find_quant_params_path(adapter_dir)
    if qp_path and os.path.isfile(qp_path):
        print(f"Loading quantization parameters from {qp_path}")
        quant_params = torch.load(qp_path, map_location=device)

        with torch.no_grad():
            for name, module in peft_model.named_modules():
                if not isinstance(module, QATW4A4LinearCore):
                    continue
                prefixes = [f"{name}.", f"base_model.model.{name}."]
                for param_name in ["activation_clip", "smooth_factor"]:
                    loaded = None
                    for pref in prefixes:
                        key = pref + param_name
                        if key in quant_params:
                            loaded = quant_params[key].to(device=device, dtype=getattr(module, param_name).dtype)
                            break
                    if loaded is not None:
                        # print(f"Setting {param_name} for {name}")
                        # print(getattr(module, param_name))
                        getattr(module, param_name).data.copy_(loaded.view_as(getattr(module, param_name)))
                        # print(getattr(module, param_name))
                    else:
                        print(f"WARNING: Could not find saved parameter for {name}.{param_name}")
    else:
        print("No quantization_params.pth found, using default initialized QAT values.")

    # 6. Set model to evaluation mode
    peft_model.eval()
    return peft_model, tokenizer




def find_available_dir(base_dir):
    existing_indices = []
    
    # Check if base directory (no suffix) exists
    if os.path.exists(base_dir):
        existing_indices.append(0)
    
    # Find all directories with numeric suffixes
    pattern = f"{base_dir}_*"
    for path in glob.glob(pattern):
        if os.path.isdir(path):
            match = re.match(rf"{re.escape(base_dir)}_(\d+)$", path)
            if match:
                existing_indices.append(int(match.group(1)))
    
    # If no directories exist, return base_dir (no suffix)
    if not existing_indices:
        return base_dir
    
    # Find next index after the largest existing index
    max_index = max(existing_indices)
    next_index = max_index + 1
    
    # Return base_dir for index 1, base_dir_N for N > 1
    if next_index == 1:
        return base_dir
    else:
        return f"{base_dir}_{next_index}"


class SaveQuantParamsCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        try:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.isdir(ckpt_dir):
                quant_params = {}
                for name, module in model.named_modules():
                    if isinstance(module, QATW4A4LinearCore):
                        prefix = f"{name}."
                        quant_params[f"{prefix}activation_clip"] = module.activation_clip.data.detach().cpu()
                        quant_params[f"{prefix}smooth_factor"] = module.smooth_factor.data.detach().cpu()
                torch.save(quant_params, os.path.join(ckpt_dir, "quantization_params.pth"))
                if len(quant_params) == 0:
                    print("WARNING: No quantization parameters found to save!")
                else:
                    torch.save(quant_params, os.path.join(ckpt_dir, "quantization_params.pth"))
                    print(f"✓ Saved {len(quant_params)} quantization parameters to {ckpt_dir}")
        except Exception as e:
            print(f"Warning: failed to save quant params at checkpoint: {e}")

class GradientMonitorCallback(TrainerCallback):
    def __init__(self, log_every_n_steps=10):
        self.log_every_n_steps = log_every_n_steps
        self.last_unclipped_norm = None
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Capture the unclipped gradient norm from Trainer's logs"""
        if logs and 'grad_norm' in logs:
            self.last_unclipped_norm = logs['grad_norm']
    
    def on_pre_optimizer_step(self, args, state, control, model=None, logs=None, **kwargs):
        """Show both unclipped (from logs) and clipped gradients"""
        if state.global_step % self.log_every_n_steps != 0:
            return
        
        # Unwrap model if using DDP or other wrappers
        unwrapped_model = model
        if hasattr(model, 'module'):
            unwrapped_model = model.module
        
        lora_grad_norm = 0.0
        clip_grad_norm = 0.0
        smooth_grad_norm = 0.0
        lora_count = 0
        clip_count = 0
        smooth_count = 0
        
        for name, param in unwrapped_model.named_parameters():
            if not param.requires_grad:
                continue
                
            if param.grad is None:
                continue
            
            grad_norm = param.grad.norm().item()
            
            # Check for LoRA parameters (lora_A or lora_B)
            if 'lora_a' in name.lower() or 'lora_b' in name.lower():
                lora_grad_norm += grad_norm ** 2
                lora_count += 1
            # Check for activation clip parameters
            elif 'activation_clip' in name:
                clip_grad_norm += grad_norm ** 2
                clip_count += 1
            # Check for smooth factor parameters
            elif 'smooth_factor' in name:
                smooth_grad_norm += grad_norm ** 2
                smooth_count += 1
        
        # Compute RMS norms (these are POST-CLIPPING)
        lora_grad_norm = lora_grad_norm ** 0.5 if lora_count > 0 else 0.0
        clip_grad_norm = clip_grad_norm ** 0.5 if clip_count > 0 else 0.0
        smooth_grad_norm = smooth_grad_norm ** 0.5 if smooth_count > 0 else 0.0
        total_clipped = (lora_grad_norm**2 + clip_grad_norm**2 + smooth_grad_norm**2)**0.5
        
        print("=" * 60)
        print(f"Step {state.global_step} - Gradient Norms:")
        print("=" * 60)
        
        # Show UNCLIPPED total (from Trainer's internal calculation)
        if self.last_unclipped_norm is not None:
            print(f"  UNCLIPPED Total:  {self.last_unclipped_norm:8.4f}")
            print(f"  Clipping ratio:   {total_clipped / self.last_unclipped_norm:.4f}")
            print("-" * 60)
        
        # Show POST-CLIPPING breakdown by parameter type
        print(f"  LoRA (clipped):   {lora_grad_norm:8.4f} ({lora_count} params)")
        print(f"  Clips (clipped):  {clip_grad_norm:8.4f} ({clip_count} params)")
        print(f"  Smooth (clipped): {smooth_grad_norm:8.4f} ({smooth_count} params)")
        print(f"  Total (clipped):  {total_clipped:8.4f}")
        print(f"  max_grad_norm:    {args.max_grad_norm}")
        print("=" * 60)



def save_quantization_parameters(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    quant_params = {}
    for name, module in model.named_modules():
        if isinstance(module, QATW4A4LinearCore):
            # Save all quantization parameters
            param_prefix = f"{name}."
            quant_params[f"{param_prefix}activation_clip"] = module.activation_clip.data.detach().cpu()
            quant_params[f"{param_prefix}smooth_factor"] = module.smooth_factor.data.detach().cpu()
    
    # Save quantization parameters
    torch.save(quant_params, os.path.join(save_dir, "quantization_params.pth"))
    print(f"Saved {len(quant_params)} quantization parameters")



# class Evaluation:
#     def __init__(self, tokenizer, dataset, device, num_samples=None):
#         """
#         Proper sliding window evaluation class.
        
#         Args:
#             tokenizer: Tokenizer for the model
#             dataset: Raw dataset (wikitext-2-raw-v1)
#             device: Device to run on
#             num_samples: Maximum number of windows to evaluate (None = all)
#         """
#         self.tokenizer = tokenizer
#         self.dataset = dataset
#         self.device = device
#         self.num_samples = num_samples

#     def evaluate(self, model, seq_len=2048, stride=2048):
#         """
#         Evaluate model using sliding window perplexity.
#         Now uses shared data prep function.
#         """
#         model.eval()
#         nlls = []
        
#         # Use shared concatenation logic
#         input_ids_full = get_concatenated_text(self.dataset, self.tokenizer)
#         seq_len_total = len(input_ids_full)
        
#         prev_end_loc = 0
        
#         with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
#             pbar = tqdm.tqdm(
#                 range(0, seq_len_total, stride), 
#                 desc="Evaluating"
#             )
            
#             for begin_loc in pbar:
#                 end_loc = min(begin_loc + seq_len, seq_len_total)
#                 trg_len = end_loc - prev_end_loc
                
#                 # Extract window
#                 input_ids = input_ids_full[begin_loc:end_loc].unsqueeze(0).to(self.device)
#                 target_ids = input_ids.clone()
                
#                 # Mask context tokens (only evaluate new tokens)
#                 if begin_loc > 0:
#                     target_ids[:, :-trg_len] = -100
                
#                 # Forward pass
#                 outputs = model(input_ids, labels=target_ids)
                
#                 # Accumulate loss
#                 neg_log_likelihood = outputs.loss * trg_len
#                 nlls.append(neg_log_likelihood)
                
#                 prev_end_loc = end_loc
                
#                 if self.num_samples and len(nlls) >= self.num_samples:
#                     break
#                 if end_loc >= seq_len_total:
#                     break
        
#         # Calculate perplexity
#         total_nll = torch.stack(nlls).sum()
#         ppl = torch.exp(total_nll / prev_end_loc)
        
#         print(f"{ppl.item():.4f}")
#         return ppl.item()
class Evaluation:
    def __init__(self, dataset, tokenizer, device, num_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = num_samples

    @torch.no_grad()
    def evaluate(self, model, seq_len=2048, stride=2048):
        model.to(torch.bfloat16)
        model.eval()
        nlls = []
        n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // 2048
        for i in tqdm.tqdm(range(n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (n_samples * 2048))


# ============================================================================
# Hadamard PE-QAT: pure INT4 Tensor Core compatible quantization
# ============================================================================

def fast_hadamard(x):
    """
    Walsh-Hadamard transform along the last dimension.
    Normalized: H @ H^T = I, so applying the same rotation to both
    activations and weights leaves the matmul output unchanged.
    
    x: [M, K] or [B, S, K]
    returns: [M, K] or [B, S, K] rotated, same dtype
    """
    K = x.shape[-1]
    *rest, K = x.shape
    x_flat = x.reshape(-1, K)
    
    h = 1
    while h < K:
        x_flat = x_flat.reshape(-1, h * 2)
        a, b = x_flat[:, :h].clone(), x_flat[:, h:].clone()
        x_flat[:, :h] = a + b
        x_flat[:, h:] = a - b
        h *= 2
    
    x_flat /= math.sqrt(K)
    return x_flat.reshape(*rest, K) if rest else x_flat.reshape(K)


class HadamardPerTokenActSTEQuant(torch.autograd.Function):
    """
    Per-token activation quantization on rotation-uniformized activations.
    
    No learned clip, no boundary gradient, no LSQ term.
    Hadamard rotation diffuses outliers uniformly, so the per-token
    scale works well without any per-channel clipping.
    
    Forward:
        x: [M, K] or [B, S, K] — already rotated by H
        num_bits: int (default 4)
        Per-token dynamic scale: s_x[t] = max|x| / qmax, detached.
    
    Backward:
        Full STE passthrough — no mask, no clip terms.
    """
    @staticmethod
    def forward(ctx, x, num_bits=4):
        qmax = (1 << (num_bits - 1)) - 1
        qmin = -(1 << (num_bits - 1))
        s_x = x.abs().amax(dim=-1, keepdim=True) / qmax
        s_x = s_x.detach()
        x_q = torch.clamp(torch.round(x / s_x), qmin, qmax)
        x_dq = x_q * s_x
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class HadamardPEQATLoRAModule(nn.Module):
    """
    LoRA module with Hadamard rotation + per-token activation quantization.
    
    No learned clip or smooth parameters. The Hadamard rotation diffuses
    outlier channels, making per-token dynamic quantization effective.
    
    Training-forward is numerically identical to full-precision matmul
    because H @ H^T = I (the rotation cancels between activations and weights).
    Quantization noise is the only source of accuracy loss.
    """
    
    def __init__(self, original_lora_module, num_bits=4):
        super().__init__()
        
        self.base_layer = original_lora_module.base_layer
        if hasattr(self.base_layer, 'in_features') and hasattr(self.base_layer, 'out_features'):
            self.in_features = self.base_layer.in_features
            self.out_features = self.base_layer.out_features
        else:
            weight_shape = self.base_layer.weight.shape
            self.in_features = weight_shape[0]
            self.out_features = weight_shape[1]
        
        self.lora_A = original_lora_module.lora_A
        self.lora_B = original_lora_module.lora_B
        self.scaling = original_lora_module.scaling
        
        self.r = original_lora_module.r
        self.lora_alpha = original_lora_module.lora_alpha
        self.lora_dropout = original_lora_module.lora_dropout
        self.active_adapter = original_lora_module.active_adapter
        
        if hasattr(original_lora_module, 'fan_in_fan_out'):
            self.fan_in_fan_out = original_lora_module.fan_in_fan_out
        if hasattr(original_lora_module, 'merged'):
            self.merged = original_lora_module.merged
        
        self.num_bits = num_bits
    
    def get_merged_weight(self):
        """W_merged = W_base + (B @ A) * scaling — gradient flows only through LoRA"""
        base_weight = self.base_layer.weight.detach()
        
        if self.r['default'] > 0 and not getattr(self, 'merged', False):
            lora_A = self.lora_A[self.active_adapter[0]].weight
            lora_B = self.lora_B[self.active_adapter[0]].weight
            scaling = self.scaling[self.active_adapter[0]]
            lora_delta = (lora_B @ lora_A) * scaling
            merged_weight = base_weight + lora_delta
        else:
            merged_weight = base_weight
        
        return merged_weight
    
    def forward(self, x):
        """
        Forward: H(x) → per-token quant → H(W) → per-row quant → F.linear
        
        The H rotation on both inputs (along K) and weights (along K,
        which is the last dim = in_features) cancels exactly in full precision:
            H(x) @ H(W)^T = x @ H @ H^T @ W^T = x @ W^T    ✓
        """
        merged_weight = self.get_merged_weight()
        
        # Hadamard rotate input along in_features
        x_rot = fast_hadamard(x)
        
        # Hadamard rotate merged weight along in_features (last dim)
        # W is [out_features, in_features], last dim = in = K
        W_rot = fast_hadamard(merged_weight)
        
        # Per-token quantization on rotated activations (no clip/smooth)
        quant_input = HadamardPerTokenActSTEQuant.apply(x_rot, self.num_bits)
        
        # Per-row symmetric weight quantization (unchanged)
        quant_weight = WeightSTEQuant.apply(W_rot, self.num_bits)
        
        # Linear with bias
        bias = self.base_layer.bias
        output = F.linear(quant_input, quant_weight, bias)
        
        return output
    
    def __repr__(self):
        return (f"HPEQATLoRAModule(in={self.in_features}, out={self.out_features}, "
                f"r={self.r['default']}, num_bits={self.num_bits})")


def hadamard_quantize_model(model, num_bits=4):
    """
    Replace all LoRA modules with HadamardPEQATLoRAModule.
    Same pattern as quantize_model() but with no clip/smooth params.
    """
    modules_to_replace = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            modules_to_replace.append((name, module))
    
    print(f"Found {len(modules_to_replace)} LoRA modules to quantize (Hadamard PE-QAT)")
    
    for name, lora_module in modules_to_replace:
        quantized_module = HadamardPEQATLoRAModule(
            lora_module,
            num_bits=num_bits,
        )
        
        parent = model
        attrs = name.split('.')
        for attr in attrs[:-1]:
            parent = getattr(parent, attr)
        setattr(parent, attrs[-1], quantized_module)
    
    return model