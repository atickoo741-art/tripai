import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, Conv1D
from peft.peft_model import PeftModel

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=4):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w

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
        """Return the merged weights for PEFT compatibility"""
        return self._get_merged_weights()

    @weight.setter  
    def weight(self, value):
        """Set the original linear weight for PEFT compatibility"""
        self.original_linear.weight = nn.Parameter(value.detach().clone())
        

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
        
        # Get merged weights and apply per-channel scaling: W = W * s_j
        merged_weights = self._get_merged_weights()  # [out_features, in_features]
        scaled_weights = merged_weights * self.smooth_factor.unsqueeze(0)  # [out_features, in_features]
        
        # PACT: per-channel clipping for activations
        clip_vals = self.constrained_clip_val  # [in_features]
        # Reshape for broadcasting: [in_features] -> [1, 1, in_features] or [1, in_features]
        
        # Apply symmetric per-channel clipping to [-a, a]
        clipped_input = torch.clamp(scaled_input, -clip_vals, clip_vals)
        

        quant_input = quantize_weight_per_channel_absmax(clipped_input, self.num_bits)

        quant_weight = quantize_weight_per_channel_absmax(scaled_weights, self.num_bits)

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
    "apply QATW4A4Linear to all nn.Linear and Conv1D layers in a PEFT GPT-2 model"
    model = peft_model.base_model.model

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, Conv1D)):
            # Skip if already quantized
            if hasattr(module, 'original_linear'):
                continue
                
            # Get parent module and attribute name
            parent = model
            attrs = name.split('.')
            for attr in attrs[:-1]:
                parent = getattr(parent, attr)
            
            # Replace with quantized version
            setattr(parent, attrs[-1], QATW4A4Linear(module))
    return model


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
    

