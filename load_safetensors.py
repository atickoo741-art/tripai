from safetensors import safe_open
import os
import torch
# file_path = "/Users/shresthmishra/Documents/Code/w4a4peqat/qat_lora_model_6/adapter_model.safetensors"
from safetensors.torch import load_file



# with safe_open(file_path, framework="pt") as f: # "pt" for PyTorch
#     # # Print the metadata
#     # print("Metadata:")
#     # metadata = f.metadata()
#     # if metadata:
#     #     for key, value in metadata.items():
#     #         print(f"  - {key}: {value}")
#     # else:
#     #     print("  No metadata found.")

#     # # Print the tensor keys
#     # print("\nTensor Keys:")
#     # for key in f.keys():
#     #     print(f"  - {key}")

#     safe_tensors_dict = load_file(file_path)

#     for key, tensor in safe_tensors_dict.items():
#         print(f"Tensor name: {key}")
#         print(tensor)
#         if key.endswith('lora_A.original_linear.weight') or key.endswith('lora_B.original_linear.weight'):
#             print(f"Tensor name: {key}")
#             print(tensor)


# quant_params = torch.load("output_qat_lora_21/checkpoint-2190/quantization_params.pth")
# quant_params = torch.load("qat_lora_model_19/quantization_params.pth")
quant_params = torch.load("qat_lora_model_22/quantization_params.pth")
# for key in quant_params:
    # print(f"Key: {key}")
    # print(f"Value: {value}")
print(quant_params)