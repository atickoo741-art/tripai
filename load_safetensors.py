from safetensors import safe_open

file_path = "/Users/shresthmishra/Documents/Code/w4a4peqat/qat_lora_model_5/adapter_model.safetensors"

with safe_open(file_path, framework="pt") as f: # "pt" for PyTorch
    # Print the metadata
    print("Metadata:")
    metadata = f.metadata()
    if metadata:
        for key, value in metadata.items():
            print(f"  - {key}: {value}")
    else:
        print("  No metadata found.")

    # Print the tensor keys
    print("\nTensor Keys:")
    for key in f.keys():
        print(f"  - {key}")