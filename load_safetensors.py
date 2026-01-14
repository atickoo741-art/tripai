from safetensors import safe_open
import os
import torch
# file_path = "/Users/shresthmishra/Documents/Code/w4a4peqat/qat_lora_model_6/adapter_model.safetensors"
file_path = "/home/shresth/peqat-checkpoints/output_qat_lora_old_clip_backward_co_5f71_001/adapter_model.safetensors"
from safetensors.torch import load_file


with safe_open(file_path, framework="pt") as f: # "pt" for PyTorch
    size = 0
    # # Print the metadata
    # print("Metadata:")
    # metadata = f.metadata()
    # if metadata:
    #     for key, value in metadata.items():
    #         print(f"  - {key}: {value}")
    # else:
    #     print("  No metadata found.")

    # # Print the tensor keys
    # print("\nTensor Keys:")
    # for key in f.keys():
    #     print(f"  - {key}")

    safe_tensors_dict = load_file(file_path)

    for key, tensor in safe_tensors_dict.items():
        print(f"Tensor name: {key}")
        # print(tensor)
        if 'lora_A' in key or 'lora_B' in key:
            size += tensor.numel()
            print(f"Tensor name: {key}")
            print(tensor)

print("total params", size)

# quant_params = torch.load("qat_lora_model_16/quantization_params.pth")
# # for key in quant_params:
#     # print(f"Key: {key}")
#     # print(f"Value: {value}")
# print(quant_params)








# """
# Check what base model LoRA adapters were trained on
# """

# import json
# import os
# from safetensors import safe_open
# from pathlib import Path


# def check_lora_base_model(checkpoint_path):
#     """
#     Determine the base model from PEFT LoRA checkpoint
    
#     Args:
#         checkpoint_path: Path to folder containing adapter_model.safetensors
#     """
#     checkpoint_path = Path(checkpoint_path)
    
#     print("="*70)
#     print("LORA BASE MODEL CHECKER")
#     print("="*70)
#     print(f"Checkpoint: {checkpoint_path}\n")
    
#     # Method 1: Check adapter_config.json (most reliable)
#     config_file = checkpoint_path / "adapter_config.json"
    
#     if config_file.exists():
#         print("[✓] Found adapter_config.json")
#         with open(config_file, 'r') as f:
#             config = json.load(f)
        
#         print("\n--- Adapter Configuration ---")
#         print(f"Base Model:        {config.get('base_model_name_or_path', 'NOT FOUND')}")
#         print(f"PEFT Type:         {config.get('peft_type', 'N/A')}")
#         print(f"Task Type:         {config.get('task_type', 'N/A')}")
#         print(f"LoRA Rank:         {config.get('r', 'N/A')}")
#         print(f"LoRA Alpha:        {config.get('lora_alpha', 'N/A')}")
#         print(f"LoRA Dropout:      {config.get('lora_dropout', 'N/A')}")
#         print(f"Target Modules:    {config.get('target_modules', 'N/A')}")
        
#         if 'modules_to_save' in config:
#             print(f"Modules to Save:   {config['modules_to_save']}")
        
#         return config.get('base_model_name_or_path')
    
#     else:
#         print("[✗] adapter_config.json NOT FOUND")
#         print("    Attempting to infer from tensor shapes...\n")
    
#     # Method 2: Inspect safetensors file
#     safetensors_file = checkpoint_path / "adapter_model.safetensors"
    
#     if not safetensors_file.exists():
#         print("[✗] adapter_model.safetensors NOT FOUND")
#         print(f"    Checked: {safetensors_file}")
#         return None
    
#     print("[✓] Found adapter_model.safetensors")
    
#     # Load tensor metadata
#     with safe_open(safetensors_file, framework="pt", device="cpu") as f:
#         keys = list(f.keys())
    
#     print(f"\n--- Tensor Analysis ---")
#     print(f"Total tensors: {len(keys)}")
    
#     # Group by layer type
#     layer_info = {}
#     for key in keys:
#         # Parse key like: base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
#         parts = key.split('.')
        
#         if 'lora_A' in parts or 'lora_B' in parts:
#             # Extract layer number and module name
#             try:
#                 layer_idx = int([p for i, p in enumerate(parts) if parts[i-1] == 'layers'][0])
#                 module = parts[-4]  # e.g., q_proj, v_proj
                
#                 if layer_idx not in layer_info:
#                     layer_info[layer_idx] = set()
#                 layer_info[layer_idx].add(module)
#             except:
#                 pass
    
#     if layer_info:
#         num_layers = max(layer_info.keys()) + 1
#         print(f"Number of layers: {num_layers}")
#         print(f"LoRA modules per layer: {sorted(layer_info[0]) if 0 in layer_info else 'N/A'}")
        
#         # Infer model based on layer count (rough heuristic)
#         model_guess = infer_model_from_layers(num_layers)
#         print(f"\n[Inference] Likely base model: {model_guess}")
#         print("    (Based on layer count - not guaranteed)")
        
#         return model_guess
    
#     print("\n[✗] Could not determine base model from checkpoint")
#     print("    Missing both adapter_config.json and recognizable tensor structure")
#     return None


# def infer_model_from_layers(num_layers):
#     """Guess model family from number of transformer layers"""
#     layer_map = {
#         32: "Llama-2-7B / Mistral-7B / Llama-3-8B",
#         40: "Llama-2-13B / Llama-3-70B",
#         48: "GPT-3 (6.7B)",
#         80: "Llama-2-70B",
#         24: "GPT-2 Large / OPT-6.7B",
#         12: "BERT-base / GPT-2 Small"
#     }
#     return layer_map.get(num_layers, f"Unknown ({num_layers} layers)")


# def check_quantization_params(checkpoint_path):
#     """Check if PE-QAT quantization parameters exist"""
#     checkpoint_path = Path(checkpoint_path)
#     quant_file = checkpoint_path / "quantization_params.pth"
    
#     if quant_file.exists():
#         print("\n--- PE-QAT Quantization Info ---")
#         print("[✓] Found quantization_params.pth")
        
#         import torch
#         params = torch.load(quant_file, map_location='cpu')
        
#         print(f"Saved parameters: {list(params.keys())}")
        
#         # Sample one module's params
#         if params:
#             sample_key = list(params.keys())[0]
#             sample = params[sample_key]
#             print(f"\nSample module: {sample_key}")
#             for k, v in sample.items():
#                 if torch.is_tensor(v):
#                     print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
#     else:
#         print("\n[✗] No quantization_params.pth found (not a PE-QAT checkpoint)")


# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) < 2:
#         print("Usage: python check_lora_base.py <path_to_checkpoint>")
#         print("\nExample:")
#         print("  python check_lora_base.py /path/to/peqat-checkpoints/output_qat_lora")
#         sys.exit(1)
    
#     checkpoint_path = sys.argv[1]
    
#     base_model = check_lora_base_model(checkpoint_path)
#     check_quantization_params(checkpoint_path)
    
#     print("\n" + "="*70)
#     if base_model:
#         print(f"✓ Base Model: {base_model}")
#     else:
#         print("✗ Could not determine base model")
#     print("="*70)
