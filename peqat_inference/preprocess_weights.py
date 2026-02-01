# preprocess_weights_from_peqat.py
"""
Load trained PE-QAT model and convert to INT4 inference format
Compatible with PE-QAT training codebase
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import struct

class PEQATModelConverter:
    """Convert trained PE-QAT model to INT4 inference format"""
    
    def __init__(self, base_model_name, adapter_dir, device='cuda'):
        self.device = torch.device(device)
        self.qmax = 7
        self.qmin = -8
        
        print(f"[1/4] Loading base model: {base_model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=None
        ).to(self.device)
        
        print(f"[2/4] Loading PEFT adapters from: {adapter_dir}")
        self.peft_model = PeftModel.from_pretrained(
            self.base_model,
            adapter_dir,
            device_map=None
        ).to(self.device)
        
        print(f"[3/4] Loading quantization parameters")
        quant_params_path = os.path.join(adapter_dir, "quantization_params.pth")
        if not os.path.exists(quant_params_path):
            raise FileNotFoundError(f"quantization_params.pth not found in {adapter_dir}")
        
        self.quant_params = torch.load(quant_params_path, map_location=self.device)
        print(f"  Found {len(self.quant_params)} quantization parameters")
        
        # DEBUG: Print sample keys to understand format
        print("\n[DEBUG] Sample quantization parameter keys:")
        sample_keys = list(self.quant_params.keys())[:10]
        for key in sample_keys:
            print(f"  {key}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    
    def get_layer_names(self):
        """
        Extract layer names from quantization parameters
        Returns list of layer names (without .qat_core.activation_clip/.smooth_factor suffix)
        """
        layer_names = set()
        
        for key in self.quant_params.keys():
            # Keys are like: base_model.model.model.layers.0.self_attn.q_proj.qat_core.activation_clip
            # Remove the .qat_core.activation_clip or .qat_core.smooth_factor suffix
            if key.endswith('.qat_core.activation_clip'):
                layer_name = key[:-len('.qat_core.activation_clip')]
                layer_names.add(layer_name)
            elif key.endswith('.qat_core.smooth_factor'):
                layer_name = key[:-len('.qat_core.smooth_factor')]
                layer_names.add(layer_name)
        
        return sorted(list(layer_names))
    
    def get_module_from_name(self, module_name):
        """Navigate to module by name"""
        parts = module_name.split('.')
        module = self.peft_model
        
        for part in parts:
            module = getattr(module, part)
        
        return module
    
    def get_merged_weight(self, module_name):
        """
        Get merged base + LoRA weight for a specific module
        Handles both nn.Linear and Conv1D weight orientations
        """
        try:
            module = self.get_module_from_name(module_name)
        except AttributeError as e:
            print(f"  ERROR: Could not find module {module_name}")
            raise
        
        # Get base layer (unwrap PEFT/QAT wrappers)
        if hasattr(module, 'qat_core'):
            # PEQATLoRAModule wrapper
            base_layer = module.qat_core.original_linear
        elif hasattr(module, 'base_layer'):
            # Direct PEFT module
            base_layer = module.base_layer
        elif hasattr(module, 'original_linear'):
            # QATW4A4LinearCore
            base_layer = module.original_linear
        else:
            base_layer = module
        
        base_weight = base_layer.weight.data.detach()
        
        # Check orientation (Conv1D stores as in×out, Linear as out×in)
        weight_is_out_in = hasattr(base_layer, 'in_features') and \
                          hasattr(base_layer, 'out_features')
        
        if not weight_is_out_in:
            # Conv1D case: transpose to get out×in
            base_weight = base_weight.T
        
        # Merge LoRA if present
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            try:
                lora_A = module.lora_A['default'].weight.data
                lora_B = module.lora_B['default'].weight.data
                scaling = module.scaling['default']
                
                lora_delta = (lora_B @ lora_A) * scaling
                merged_weight = base_weight + lora_delta
                print(f"  ✓ Merged LoRA (scaling={scaling:.4f})")
            except Exception as e:
                print(f"  ⚠ Could not merge LoRA: {e}")
                merged_weight = base_weight
        else:
            merged_weight = base_weight
            print(f"  ℹ No LoRA adapters found")
        
        return merged_weight, weight_is_out_in
    
    def process_layer(self, module_name):
        """
        Process one layer: merge LoRA, apply smoothing, quantize weights
        
        Returns:
            weight_int4_packed: (d, k//2) packed INT4
            weight_scales: (d,) FP32 per output channel
            smooth_factors: (k,) FP32 per input channel
            act_clip_thresholds: (k,) FP32 per input channel
        """
        print(f"\n[Processing] {module_name}")
        
        # Step 1: Get merged weight
        merged_weight, weight_is_out_in = self.get_merged_weight(module_name)
        d, k = merged_weight.shape
        print(f"  Weight shape: ({d}, {k})")
        
        # Step 2: Load quantization parameters
        smooth_key = f"{module_name}.qat_core.smooth_factor"
        clip_key = f"{module_name}.qat_core.activation_clip"
        
        if smooth_key not in self.quant_params:
            raise KeyError(f"Smoothing factors not found: {smooth_key}\nAvailable keys: {list(self.quant_params.keys())[:5]}")
        if clip_key not in self.quant_params:
            raise KeyError(f"Clipping thresholds not found: {clip_key}")
        
        # FIX: Convert BFloat16 -> Float32 before .numpy()
        smooth_factors = self.quant_params[smooth_key].cpu().float().numpy()
        act_clip_thresholds = self.quant_params[clip_key].cpu().float().numpy()
        
        print(f"  Smoothing range: [{smooth_factors.min():.4f}, {smooth_factors.max():.4f}]")
        print(f"  Clipping range: [{act_clip_thresholds.min():.4f}, {act_clip_thresholds.max():.4f}]")
        
        # Step 3: Apply smoothing to weights (multiply)
        # PE-QAT: scaled_weights = merged_weight * smoothfactor
        merged_weight_np = merged_weight.cpu().float().numpy()
        smoothed_weights = merged_weight_np * smooth_factors[None, :]
        
        # Step 4: Quantize weights per OUTPUT channel (per row)
        weight_scales = np.zeros(d, dtype=np.float32)
        W_q = np.zeros((d, k), dtype=np.int8)
        
        for j in range(d):
            # Scale: max(|W[j,:]|) / qmax
            row_max = np.max(np.abs(smoothed_weights[j, :]))
            weight_scales[j] = row_max / self.qmax if row_max > 0 else 1e-8
            
            W_q[j, :] = np.clip(
                np.round(smoothed_weights[j, :] / weight_scales[j]),
                self.qmin, self.qmax
            ).astype(np.int8)
        
        print(f"  Weight scales range: [{weight_scales.min():.6f}, {weight_scales.max():.6f}]")
        print(f"  W_q range: [{W_q.min()}, {W_q.max()}]")
        
        # Step 5: Pack INT4
        weight_int4_packed = self._pack_int4(W_q)
        
        return weight_int4_packed, weight_scales, smooth_factors, act_clip_thresholds

    
    def _pack_int4(self, W_int8):
        """Pack INT8 array into INT4 (2 values per byte)"""
        d, k = W_int8.shape
        assert k % 2 == 0, f"k must be even, got {k}"
        
        W_packed = np.zeros((d, k // 2), dtype=np.uint8)
        
        for j in range(d):
            for c in range(0, k, 2):
                val0 = W_int8[j, c] & 0x0F
                val1 = W_int8[j, c+1] & 0x0F
                W_packed[j, c//2] = val0 | (val1 << 4)
        
        savings = 100 * (1 - (d*k//2) / (d*k*4))
        print(f"  Packed: {d}×{k} INT8 -> {d}×{k//2} bytes ({savings:.1f}% memory savings)")
        return W_packed
    
    def save_layer(self, module_name, output_file):
        """Process and save one layer"""
        weight_packed, weight_scales, smooth_factors, act_clip_thresholds = \
            self.process_layer(module_name)
        
        d, k_half = weight_packed.shape
        k = k_half * 2
        
        with open(output_file, 'wb') as f:
            # Header: d, k, rank (placeholder)
            f.write(struct.pack('III', d, k, 32))
            
            # Packed weights
            f.write(weight_packed.tobytes())
            
            # Weight scales
            f.write(weight_scales.astype(np.float32).tobytes())
            
            # Smoothing factors
            f.write(smooth_factors.astype(np.float32).tobytes())
            
            # Activation clip thresholds
            f.write(act_clip_thresholds.astype(np.float32).tobytes())
        
        total_size = d*k_half + (d + 2*k)*4
        print(f"  ✓ Saved to {output_file}")
        print(f"  Total size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--adapter_dir", type=str, required=True,
                       help="Path to trained PE-QAT adapter directory")
    parser.add_argument("--output_dir", type=str, default="./peqat_int4_weights")
    parser.add_argument("--layers", type=str, nargs='+',
                       help="Layer names to convert (e.g., model.layers.0.self_attn.q_proj)")
    parser.add_argument("--max_layers", type=int, default=5,
                       help="Maximum number of layers to convert (default: 5, use -1 for all)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    converter = PEQATModelConverter(
        base_model_name=args.base_model,
        adapter_dir=args.adapter_dir
    )
    
    # Auto-detect layers if not specified
    if args.layers is None:
        print("\n[4/4] Auto-detecting quantized layers...")
        layers = converter.get_layer_names()
        print(f"  ✓ Found {len(layers)} quantized layers")
        
        if len(layers) > 0:
            print("\nSample layer names:")
            for i, layer in enumerate(layers[:5]):
                print(f"  [{i}] {layer}")
    else:
        layers = args.layers
    
    if len(layers) == 0:
        print("\n❌ ERROR: No layers found!")
        print("This might mean:")
        print("  1. quantization_params.pth is empty or corrupted")
        print("  2. Layer naming format has changed")
        print("\nPlease check the quantization_params.pth file")
        exit(1)
    
    # Process layers
    print(f"\n{'='*60}")
    print("Converting PE-QAT Model to INT4 Inference Format")
    print(f"{'='*60}")
    
    num_to_process = len(layers) if args.max_layers == -1 else min(args.max_layers, len(layers))
    
    for i, layer_name in enumerate(layers[:num_to_process]):
        print(f"\n[{i+1}/{num_to_process}] Converting: {layer_name}")
        
        # Create safe filename
        safe_name = layer_name.replace('.', '_').replace('/', '_')
        output_file = os.path.join(args.output_dir, f"{safe_name}.peqat")
        
        try:
            converter.save_layer(layer_name, output_file)
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"✓ Conversion complete!")
    print(f"  Processed: {num_to_process}/{len(layers)} layers")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*60}")
