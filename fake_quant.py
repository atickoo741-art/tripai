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

class ActSTEQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, clip_vals, num_bits):
        # Symmetric per-channel: scale = clip/qmax, zero_point = 0
        qmin = -(1 << (num_bits - 1))
        qmax = (1 << (num_bits - 1)) - 1
        
        # Ensure qmax is a float if clip_vals is float to avoid type errors in division
        if isinstance(qmax, int):
            qmax = float(qmax)

        scale = (clip_vals / qmax).clamp(min=1e-8)

        # Right-align broadcast to x rank
        target_shape = [1] * len(x.shape)
        target_shape[-1] = scale.shape[-1] 
        broadcastable_scale = scale.view(target_shape)

        x_q = torch.clamp(torch.round(x / broadcastable_scale), qmin, qmax)
        x_dq = x_q * broadcastable_scale
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        # STE: gradients only to x
        return grad_output, None, None

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


class QATW4A4Linear(nn.Module):
    def __init__(self, original_linear_layer, num_bits=4, init_clip_val=2.0):
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
        else:
            # Conv1D-like (GPT-2)
            weight_shape = original_linear_layer.weight.shape
            # Conv1D stores weight as [in_features, out_features]
            self.in_features = weight_shape[0]
            self.out_features = weight_shape[1]
            self.weight_is_out_in = False
        

        self.device = next(original_linear_layer.parameters()).device
        

        self.activation_clip = nn.Parameter(
            torch.ones(self.in_features, device=self.device) * init_clip_val
        )
        
        # Initialize per-channel smooth factors to 1.0
        self.smooth_factor = nn.Parameter(torch.ones(self.in_features, device=self.device))

        # Freeze base weights/bias of the wrapped linear layer
        if hasattr(self.original_linear, 'weight'):
            self.original_linear.weight.requires_grad_(False)
        if hasattr(self.original_linear, 'bias') and self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad_(False)
        # Train only QAT per-channel parameters
        self.activation_clip.requires_grad_(True)
        # self.raw_alpha.requires_grad_(True)
        self.smooth_factor.requires_grad_(True)

        W_base = self.original_linear.weight
        W_cpu = W_base.detach().to(torch.float32).cpu()

        self.min_base = W_cpu.min(dim=1).values    
        self.max_base = W_cpu.max(dim=1).values    

        self.min_base = self.min_base.to(device=self.device)
        self.max_base = self.max_base.to(device=self.device)

        

    @property 
    def constrained_clip_val(self):
        # Ensure strictly-positive per-channel clip to avoid degenerate scales
        # Keep no hard upper cap; training/calibration sets magnitude
        return torch.clamp(torch.abs(self.activation_clip), min=1e-6)


    @property
    def weight(self):
        # return self._get_merged_weights()
        return self.original_linear.weight

    # @weight.setter  
    # def weight(self, value):
    #     """Set the original linear weight for PEFT compatibility"""
    #     self.original_linear.weight = nn.Parameter(value.detach().clone())
        

    def _get_merged_weights(self):
        """Merge base weights with LoRA deltas if they exist."""
        # Cache during eval mode
        if not self.training and hasattr(self, '_cached_merged_weights'):
            return self._cached_merged_weights
            
        base_weight = self.original_linear.weight
        # Normalize to [out_features, in_features] for internal computation
        if not self.weight_is_out_in:
            base_weight = base_weight.t()
        
        # Check if this is a LoRA layer by checking if the original linear has LoRA attributes
        if hasattr(self.original_linear, 'lora_A') and hasattr(self.original_linear, 'lora_B'):
            lora_A = self.original_linear.lora_A.default.weight
            lora_B = self.original_linear.lora_B.default.weight
            lora_scaling = self.original_linear.scaling['default']
            
            lora_delta = lora_B @ lora_A * lora_scaling
            
            if lora_delta.shape != base_weight.shape:
                delta = torch.zeros_like(base_weight)
                delta[:lora_delta.shape[0], :lora_delta.shape[1]] = lora_delta
                lora_delta = delta
                
            merged = base_weight + lora_delta
        else:
            # For LoRA layers that were converted to QATW4A4Linear, the weight is already the LoRA weight
            # The original_linear.weight in this case contains the actual trainable LoRA weight, not base weight
            merged = base_weight
        
        # Cache for eval mode
        if not self.training:
            self._cached_merged_weights = merged
            
        return merged


    def forward(self, input):
        # input shape: [batch_size, seq_len, in_features] or [batch_size, in_features]
        # Ensure input_absmax is the right shape [in_features]
        
        # Reshape smooth_factor for broadcasting: [in_features] -> [1, 1, in_features] or [1, in_features]
        if input.dim() == 3:  # [batch, seq_len, in_features]
            smooth_factor = self.smooth_factor.view(1, 1, -1)
        else:  # [batch, in_features]
            smooth_factor = self.smooth_factor.view(1, -1)
            
        # Apply per-channel scaling to input: x = x / s_j
        scaled_input = input / smooth_factor
        # merged_weights = self._get_merged_weights()

        merged_weights = self.weight.T if not self.weight_is_out_in else self.weight
        # print(self.weight)

        scaled_weights = merged_weights * self.smooth_factor.unsqueeze(0) 

        # PACT: per-channel clipping for activations
        clip_vals = self.constrained_clip_val  # [in_features]
        clipped_input = torch.clamp(scaled_input, -clip_vals, clip_vals)
        

        # quant_input = quantize_weight_per_channel_absmax(clipped_input, self.num_bits)
        # quant_weight = quantize_weight_per_channel_absmax(scaled_weights, self.num_bits)
        # quant_input = STEQuant.apply(clipped_input, self.num_bits)
        # quant_weight = STEQuant.apply(scaled_weights, self.num_bits)

        quant_input = ActSTEQuant.apply(clipped_input, self.constrained_clip_val, self.num_bits)
        quant_weight = WeightSTEQuant.apply(scaled_weights, self.num_bits)

        # If original layer is Conv1D (weights stored [in, out]), ensure quant_weight supplied to F.linear is [out, in]
            
        output = F.linear(quant_input, quant_weight, self.original_linear.bias)

        return output

    def check_gradients(self):
        """Debug method to check gradient flow"""
        print(f"activation_clip grad: {self.activation_clip.grad}")
        if hasattr(self, 'smooth_factor'):
            grad = self.smooth_factor.grad
            grad_info = 'None' if grad is None else f"mean={grad.mean().item():.6f}"
            print(f"smooth_factor grad: {grad_info}")
        if hasattr(self.original_linear, 'lora_A'):
            print(f"lora_A grad: {self.original_linear.lora_A.default.weight.grad is not None}")
            print(f"lora_B grad: {self.original_linear.lora_B.default.weight.grad is not None}")






def quantize_peft_gpt2(peft_model):
    """Apply QATW4A4Linear to all linear layers within the peft_model's base_model."""
    base_model_to_quantize = peft_model.base_model.model

    for name, module in base_model_to_quantize.named_modules():
        if isinstance(module, (nn.Linear, Conv1D)):
            # This check is crucial to ensure we don't re-wrap if somehow already done
            if hasattr(module, 'original_linear'): # This is how your QATW4A4Linear signals it's a wrapper
                continue # Skip if it's already your custom QAT module

            print(f"Applying quantization to {name}")

            parent = base_model_to_quantize
            attrs = name.split('.')
            for attr in attrs[:-1]:
                parent = getattr(parent, attr)
            
            # Replace the original (now LoRA-enabled) linear layer with your QATW4A4Linear wrapper
            setattr(parent, attrs[-1], QATW4A4Linear(module))


# sorts models by model type and passes model on to respective model specific quantization function
def quantize_model(model):
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block

    if isinstance(model, PeftModel):
        base = model.base_model.model
    else:
        base = model


    if getattr(base.config, 'model_type', None) == 'gpt2':
       return quantize_peft_gpt2(model)
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


def load_peqat_model(adapter_dir: str, base_model_name: str = "gpt2", device: torch.device | None = None):
    """
    Load a base HF model and PEFT LoRA adapters from adapter_dir, optionally merge them,
    apply QATW4A4Linear wrappers to GPT-2 linear layers, and load saved quantization
    parameters (activation_clip, smooth_factor) from quantization_params.pth.

    Returns: (model, tokenizer)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    # Load base and adapters
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32, device_map=None).to(device)
    peft_model = PeftModel.from_pretrained(base, adapter_dir, torch_dtype=torch.float32).to(device)


    quantize_model(peft_model)
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    evaluation = Evaluation(tokenizer, dataset, device)
    evaluation.evaluate(peft_model, seq_len=1024, n_samples=40)
    
    # Load quant params
    qp_path = _find_quant_params_path(adapter_dir)
    if qp_path and os.path.isfile(qp_path):
        quant_params = torch.load(qp_path, map_location=device)
        with torch.no_grad():
            for name, module in peft_model.named_modules():
                if isinstance(module, QATW4A4Linear):
                    # Keys may have been saved with 'base_model.model.' prefix
                    prefixes = [f"{name}.", f"base_model.model.{name}."]
                    for param_name in ["activation_clip", "smooth_factor"]:
                        loaded = None
                        for pref in prefixes:
                            key = pref + param_name
                            if key in quant_params:
                                loaded = quant_params[key].to(device=device, dtype=module.activation_clip.dtype)
                                break
                        if loaded is not None:
                            getattr(module, param_name).data.copy_(loaded.view_as(getattr(module, param_name)))
    else:
        # No saved params; leave defaults
        pass

    peft_model.eval()
    return peft_model, tokenizer

import os
from transformers import TrainerCallback
from fake_quant import QATW4A4Linear
import torch

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
                    if isinstance(module, QATW4A4Linear):
                        prefix = f"{name}."
                        quant_params[f"{prefix}activation_clip"] = module.activation_clip.data.detach().cpu()
                        if hasattr(module, 'smooth_factor'):
                            quant_params[f"{prefix}smooth_factor"] = module.smooth_factor.data.detach().cpu()
                        if hasattr(module, 'input_min') and hasattr(module, 'input_max'):
                            quant_params[f"{prefix}input_min"] = module.input_min.data.detach().cpu()
                            quant_params[f"{prefix}input_max"] = module.input_max.data.detach().cpu()
                        quant_params[f"{prefix}weight_scale"] = module.weight_scale.data.detach().cpu()
                        quant_params[f"{prefix}weight_zero_point"] = module.weight_zero_point.data.detach().cpu()
                        quant_params[f"{prefix}act_scale"] = module.act_scale.data.detach().cpu()
                        quant_params[f"{prefix}act_zero_point"] = module.act_zero_point.data.detach().cpu()
                torch.save(quant_params, os.path.join(ckpt_dir, "quantization_params.pth"))
        except Exception as e:
            print(f"Warning: failed to save quant params at checkpoint: {e}")

def save_quantization_parameters(model, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        quant_params = {}
        i = 0
        for name, module in model.named_modules():
            i += 1
            print(i)
            if isinstance(module, QATW4A4Linear):
                # Save all quantization parameters
                param_prefix = f"{name}."
                quant_params[f"{param_prefix}activation_clip"] = module.activation_clip.data
                quant_params[f"{param_prefix}smooth_factor"] = module.smooth_factor.data
                quant_params[f"{param_prefix}input_min"] = module.input_min.data
                quant_params[f"{param_prefix}input_max"] = module.input_max.data
                quant_params[f"{param_prefix}weight_scale"] = module.weight_scale.data
                quant_params[f"{param_prefix}weight_zero_point"] = module.weight_zero_point.data
                quant_params[f"{param_prefix}act_scale"] = module.act_scale.data
                quant_params[f"{param_prefix}act_zero_point"] = module.act_zero_point.data
        
        # Save quantization parameters
        torch.save(quant_params, os.path.join(save_dir, "quantization_params.pth"))
        print(f"Saved {len(quant_params)} quantization parameters")
class Evaluation:
    def __init__(self, tokenizer, dataset, device, num_samples = 40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
    
    def evaluate(self, model, seq_len=1024, n_samples=None):
        input_ids = self.tokenizer("\n\n".join(self.dataset["text"]), return_tensors="pt").input_ids.to(self.device)
        n_samples = n_samples if n_samples is not None else (input_ids.size(1) // seq_len)

        model.eval()

        nlls = []
        with torch.no_grad():
            for i in range(n_samples):
                start = i * seq_len
                end = (i + 1) * seq_len
                batch = input_ids[:, start:end]
                logits = model(batch).logits
                shift_logits = logits[:, :-1, :].contiguous().float()
                shift_labels = batch[:, 1:]
                loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss.float() * (seq_len - 1))
        denom = n_samples * (seq_len - 1)
        ppl = torch.exp(torch.stack(nlls).sum() / denom)
        print(f"Perplexity: {ppl.item():.2f}")
