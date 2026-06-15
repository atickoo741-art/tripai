import json
import os
import yaml

from peqat_deploy.config.schema import ExportJob


def load_export_job(discovered: dict, **overrides) -> ExportJob:
    source_dir = discovered["source_dir"]
    source_basename = discovered["source_basename"]
    parent_dir = os.path.dirname(source_dir)

    base_model_name = "meta-llama/Llama-2-7b-hf"
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lora_rank = 32
    lora_alpha = 64
    weight_bits = 4
    activation_bits = 4
    block_size = 2048
    calibration_percentile = 99.5
    learn_smooth = True
    calibrate_smooth = True

    if discovered["training_config"]:
        with open(discovered["training_config"]) as f:
            cfg = yaml.safe_load(f)
        if cfg:
            model_cfg = cfg.get("model", {})
            base_model_name = model_cfg.get("name", base_model_name)
            weight_bits = model_cfg.get("num_bits_weight", weight_bits)
            activation_bits = model_cfg.get("num_bits_activation", activation_bits)
            block_size = model_cfg.get("block_size", block_size)

            lora_cfg = cfg.get("lora", {})
            lora_rank = lora_cfg.get("rank", lora_rank)
            lora_alpha = lora_cfg.get("alpha", lora_alpha)
            target_modules = lora_cfg.get("target_modules", target_modules)

            cal_cfg = cfg.get("calibration", {})
            calibration_percentile = cal_cfg.get("percentile", calibration_percentile)
            learn_smooth = cal_cfg.get("learn_smooth", learn_smooth)
            calibrate_smooth = cal_cfg.get("calibrate_smooth", calibrate_smooth)

    if discovered["adapter_config"]:
        with open(discovered["adapter_config"]) as f:
            adapter_cfg = json.load(f)
        adapter_base = adapter_cfg.get("base_model_name_or_path")
        if adapter_base and adapter_base != "gpt2":
            base_model_name = adapter_base

    export_dir = os.path.join(parent_dir, source_basename + "_export")

    job = ExportJob(
        source_dir=source_dir,
        source_basename=source_basename,
        selected_checkpoint=discovered.get("selected_checkpoint"),
        export_dir=export_dir,
        base_model_name=base_model_name,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        block_size=block_size,
        calibration_percentile=calibration_percentile,
        learn_smooth=learn_smooth,
        calibrate_smooth=calibrate_smooth,
    )

    for key, value in overrides.items():
        if hasattr(job, key) and value is not None:
            setattr(job, key, value)

    return job
