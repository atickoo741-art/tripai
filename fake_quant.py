import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, Conv1D
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

class ActSTEQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, clip_vals, num_bits):
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
        
        # Save EVERYTHING we need for the full gradient
        ctx.save_for_backward(x, x_clipped, x_dq, clip, scale)
        ctx.qmax = qmax
        return x_dq
    
    @staticmethod
    def backward(ctx, grad_output):
        x, x_clipped, x_dq, clip, scale = ctx.saved_tensors
        qmax = ctx.qmax
        
        # Broadcasting
        if x.dim() == 3:
            clip_b = clip.view(1, 1, -1)
        else:
            clip_b = clip.view(1, -1)
        
        # 1. Gradient to Input (Standard STE)
        mask_inside = (x.abs() <= clip_b)
        grad_x = grad_output * mask_inside.float()
        
        # 2. Gradient to Clip (THE OPTIMAL FORMULA)
        # This accounts for BOTH clipping and quantization effects
        
        grad_alpha = torch.zeros_like(clip)
        
        # Component A: Gradient from CLIPPED values (Outliers)
        # If x > α, output = α, so ∂output/∂α = 1
        grad_clip_upper = (grad_output * (x > clip_b).float()).sum(dim=(0, 1))
        grad_clip_lower = (grad_output * (x < -clip_b).float() * -1.0).sum(dim=(0, 1))
        
        # Component B: Gradient from QUANTIZATION ERROR (All in-range values)
        # The quantization error is (x_dq - x_clipped)
        # This error is a function of the scale (α/qmax)
        # Derivation: x_dq = round(x/s)*s where s=α/qmax
        # For small changes in α, the change in x_dq is proportional to (x_dq - x)/α
        # This captures: "If I increase α (bigger bins), does my reconstruction error get better or worse?"
        
        mask_in_range = (x.abs() <= clip_b)
        quant_error = x_dq - x_clipped  # The reconstruction error
        
        # The gradient contribution from quantization error:
        # ∂(x_dq)/∂α ≈ (x_dq - x_clipped) / α
        # This tells α: "Adjust to minimize the reconstruction error"
        grad_quant = (grad_output * quant_error * mask_in_range.float() / (clip_b + 1e-8)).sum(dim=(0, 1))
        
        # COMBINE: Total gradient is clipping loss + quantization loss
        grad_alpha = grad_clip_upper + grad_clip_lower + grad_quant
        
        return grad_x, grad_alpha, None


class WeightSTEQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, num_bits):
        # Symmetric per-row(out) scale from row max (detached); zero_point = 0
        qmin = -(1 << (num_bits - 1))
        qmax = (1 << (num_bits - 1)) - 1
        row_max = W.detach().abs().amax(dim=0, keepdim=True)
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
        
        # Trainable QAT parameters
        self.activation_clip = nn.Parameter(
            torch.ones(self.in_features, device=self.device) * init_clip_val
        )
        self.smooth_factor = nn.Parameter(
            torch.ones(self.in_features, device=self.device)
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
        quant_input = ActSTEQuant.apply(scaled_input, clip_vals, self.num_bits)
        
        # Quantize weights
        quant_weight = WeightSTEQuant.apply(scaled_weights, self.num_bits)

        # ensure dtype match
        quant_weight = quant_weight.to(quant_input.dtype)

        # Linear operation
        bias = self.qat_core.original_linear.bias
        if bias is not None:
            bias = bias.to(quant_input.dtype)
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
            print(original_linear_layer.weight)
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

    if getattr(base.config, 'model_type', None) == 'gpt2':
        return quantize_peft_gpt2(model, num_bits=num_bits, init_clip_val=init_clip_val)
    else:
        raise ValueError(f"Unsupported model type: {base.config.model_type}")
    


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

def get_dtype():
    if torch.cuda.is_available():
        supports_bf16 = torch.cuda.is_bf16_supported()
        use_bf16 = supports_bf16  
        use_fp16 = not supports_bf16  
        use_tf32 = True
        model_dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        use_bf16 = use_fp16 = use_tf32 = False
        model_dtype = torch.float32
    # return model_dtype, use_bf16,use_fp16,use_tf32
    return use_bf16,use_fp16,use_tf32

def load_peqat_model(adapter_dir: str, base_model_name: str = "gpt2", device: torch.device | None = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    print(f"Loading base model {base_model_name} and PEFT adapters from {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype, use_bf16, use_fp16, use_tf32 = get_dtype()
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=model_dtype, device_map=None).to(device)

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
    """Find the next available directory with an index to prevent overwrites."""
    index = 1
    while True:
        dir_name = f"{base_dir}_{index}" if index > 1 else base_dir
        if not os.path.exists(dir_name):
            return dir_name
        index += 1


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
    def __init__(self, log_every_n_steps=50):  # Remove clip thresholds
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
    
    def on_step_end(self, args, state, control, **kwargs):  # Changed from on_optimizer_step
        """Monitor gradients BEFORE they're used"""
        self.step_count += 1
        
        if self.step_count % self.log_every_n_steps != 0:
            return
        
        model = kwargs.get("model")
        
        # Just MONITOR, don't clip
        lora_grads = []
        qat_grads = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            
            grad_norm = param.grad.norm().item()
            
            if 'lora' in name.lower():
                lora_grads.append((name, grad_norm))
            elif 'activation_clip' in name or 'smooth_factor' in name:
                qat_grads.append((name, grad_norm))
        
        # Compute group norms
        if lora_grads:
            lora_total_norm = sum(n**2 for _, n in lora_grads) ** 0.5
            print(f"\nLoRA total norm: {lora_total_norm:.4f}")
            print(f"  Max individual: {max(n for _, n in lora_grads):.4f}")
            print(f"  Mean individual: {sum(n for _, n in lora_grads)/len(lora_grads):.4f}")
        
        if qat_grads:
            qat_total_norm = sum(n**2 for _, n in qat_grads) ** 0.5
            print(f"\nQAT total norm: {qat_total_norm:.4f}")
            print(f"  Max individual: {max(n for _, n in qat_grads):.4f}")
            print(f"  Mean individual: {sum(n for _, n in qat_grads)/len(qat_grads):.4f}")


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



class Evaluation:
    def __init__(self, tokenizer, dataset, device, num_samples=None, block_size=1024):
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = block_size
        self.num_samples = num_samples # Max number of samples (batches) to evaluate

        # Tokenize and group texts
        tokenized_datasets = dataset.map(
            self._tokenize_function,
            batched=True,
            num_proc=4, # Use multiple processes for faster tokenization
            remove_columns=["text"],
        )

        # Group texts into chunks of block_size
        self.lm_dataset = tokenized_datasets.map(
            self._group_texts,
            batched=True,
            batch_size=1000,
            num_proc=4,
        )
        # Set format for PyTorch
        self.lm_dataset.set_format(type="torch", columns=["input_ids", "labels"])

        # Data collator for language modeling will handle padding
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def _tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    def _group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def evaluate(self, model):
        model.eval()
        total_loss = 0.0
        num_batches = 0

        eval_dataloader = torch.utils.data.DataLoader(
            self.lm_dataset, 
            batch_size=8,  # Match training eval batch size
            collate_fn=self.data_collator
        )

        if self.num_samples is not None:
            eval_dataloader = itertools.islice(eval_dataloader, self.num_samples)

        for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating..."):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=labels)
                # outputs.loss is already the mean loss per token in this batch
                total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss))

        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Perplexity: {perplexity.item():.2f}")
        return perplexity.item()
