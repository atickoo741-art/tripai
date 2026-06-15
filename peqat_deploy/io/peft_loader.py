import json
import os

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file


def load_adapter(model, adapter_dir: str):
    adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(adapter_config_path):
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")

    with open(adapter_config_path) as f:
        lora_config_dict = json.load(f)

    lora_config = LoraConfig(**lora_config_dict)
    peft_model = get_peft_model(model, lora_config)

    weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"adapter_model.safetensors not found in {adapter_dir}")

    saved_state_dict = load_file(weights_path)
    remapped = {}
    for saved_key, value in saved_state_dict.items():
        if "lora_A.weight" in saved_key or "lora_B.weight" in saved_key:
            runtime_key = saved_key.replace(".weight", ".default.weight")
            remapped[runtime_key] = value
        else:
            remapped[saved_key] = value

    missing, unexpected = peft_model.load_state_dict(remapped, strict=False)
    lora_missing = [k for k in missing if "lora" in k.lower()]
    lora_unexpected = [k for k in unexpected if "lora" in k.lower()]
    if lora_missing:
        print(f"  Warning: missing LoRA keys: {lora_missing}")
    if lora_unexpected:
        print(f"  Warning: unexpected LoRA keys: {lora_unexpected}")

    peft_model.eval()
    return peft_model
