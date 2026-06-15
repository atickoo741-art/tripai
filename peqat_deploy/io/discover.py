import os
import glob
from pathlib import Path


def discover_source_dir(source_dir: str, checkpoint: str | None = None) -> dict:
    source_dir = os.path.abspath(source_dir)
    source_basename = os.path.basename(source_dir)

    result = {
        "source_dir": source_dir,
        "source_basename": source_basename,
        "selected_checkpoint": None,
        "adapter_config": None,
        "adapter_weights": None,
        "quantization_params": None,
        "training_config": None,
        "tokenizer_files": [],
        "model_config": None,
        "checkpoints": [],
    }

    checkpoint_dirs = sorted(
        glob.glob(os.path.join(source_dir, "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1]),
    )
    result["checkpoints"] = checkpoint_dirs

    if checkpoint:
        candidate = os.path.join(source_dir, checkpoint)
        if os.path.isdir(candidate):
            result["selected_checkpoint"] = candidate
        else:
            raise FileNotFoundError(
                f"Specified checkpoint '{checkpoint}' not found in {source_dir}"
            )
    elif checkpoint_dirs:
        result["selected_checkpoint"] = checkpoint_dirs[-1]

    search_dir = result["selected_checkpoint"] or source_dir

    adapter_config_path = os.path.join(search_dir, "adapter_config.json")
    if os.path.isfile(adapter_config_path):
        result["adapter_config"] = adapter_config_path

    adapter_weights_path = os.path.join(search_dir, "adapter_model.safetensors")
    if os.path.isfile(adapter_weights_path):
        result["adapter_weights"] = adapter_weights_path

    quant_params_path = os.path.join(search_dir, "quantization_params.pth")
    if os.path.isfile(quant_params_path):
        result["quantization_params"] = quant_params_path

    for fname in ("config.yaml", "alpaca_config.yaml"):
        candidate = os.path.join(source_dir, fname)
        if os.path.isfile(candidate):
            result["training_config"] = candidate
            break
    if not result["training_config"]:
        candidate = os.path.join(search_dir, "training_config.yaml")
        if os.path.isfile(candidate):
            result["training_config"] = candidate

    tokenizer_extensions = (
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
    )
    for ext in tokenizer_extensions:
        candidate = os.path.join(search_dir, ext)
        if os.path.isfile(candidate):
            result["tokenizer_files"].append(candidate)

    if not result["tokenizer_files"]:
        for ext in tokenizer_extensions:
            candidate = os.path.join(source_dir, ext)
            if os.path.isfile(candidate):
                result["tokenizer_files"].append(candidate)

    config_json = os.path.join(search_dir, "config.json")
    if os.path.isfile(config_json):
        result["model_config"] = config_json

    return result
