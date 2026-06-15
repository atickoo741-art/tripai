import json
import os
import shutil

import torch
from safetensors.torch import save_file as safetensors_save


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_artifacts(
    export_dir: str,
    export_job,
    discovered: dict,
    merged_weights: dict,
    quant_params: dict,
    quantized: dict,
    packed: dict,
    error_stats: dict,
    layer_map: dict,
    manifest: dict,
    overwrite: bool = False,
):
    if os.path.exists(export_dir):
        if overwrite:
            shutil.rmtree(export_dir)
            print(f"  Removed existing export dir: {export_dir}")
        else:
            raise FileExistsError(
                f"Export dir already exists: {export_dir}. Use --overwrite to replace."
            )

    print(f"  Creating export directory: {export_dir}")
    _ensure_dir(export_dir)

    _ensure_dir(os.path.join(export_dir, "metadata"))
    _ensure_dir(os.path.join(export_dir, "tokenizer"))
    _ensure_dir(os.path.join(export_dir, "activation"))
    _ensure_dir(os.path.join(export_dir, "weights"))
    _ensure_dir(os.path.join(export_dir, "validation"))

    # --- manifest.json ---
    manifest_path = os.path.join(export_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote manifest.json")

    # --- export_config.json ---
    config_path = os.path.join(export_dir, "export_config.json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "source_dir": export_job.source_dir,
                "source_basename": export_job.source_basename,
                "selected_checkpoint": export_job.selected_checkpoint,
                "export_dir": export_job.export_dir,
                "base_model_name": export_job.base_model_name,
                "weight_bits": export_job.weight_bits,
                "activation_bits": export_job.activation_bits,
                "lora_rank": export_job.lora_rank,
                "lora_alpha": export_job.lora_alpha,
                "target_modules": export_job.target_modules,
                "dtype": export_job.dtype,
            },
            f,
            indent=2,
        )
    print(f"  Wrote export_config.json")

    # --- model_config.json ---
    if discovered.get("model_config"):
        with open(discovered["model_config"]) as f:
            model_cfg = json.load(f)
        model_config_path = os.path.join(export_dir, "model_config.json")
        with open(model_config_path, "w") as f:
            json.dump(model_cfg, f, indent=2)
        print(f"  Wrote model_config.json")

    # --- metadata/ ---
    if discovered.get("training_config"):
        shutil.copy2(
            discovered["training_config"],
            os.path.join(export_dir, "metadata", "source_training_config.yaml"),
        )
        print(f"  Copied training config")

    if discovered.get("adapter_config"):
        shutil.copy2(
            discovered["adapter_config"],
            os.path.join(export_dir, "metadata", "adapter_config.json"),
        )
        print(f"  Copied adapter config")

    export_log = {
        "timestamp": manifest["timestamp"],
        "git_commit": manifest.get("git_commit"),
        "source_dir": export_job.source_dir,
        "selected_checkpoint": export_job.selected_checkpoint,
        "num_layers_exported": len(layer_map),
        "weight_bits": export_job.weight_bits,
        "activation_bits": export_job.activation_bits,
        "error_stats_summary": {
            "mean_max_abs_error": sum(
                e["max_abs_error"] for e in error_stats.values()
            )
            / max(len(error_stats), 1),
            "mean_mean_abs_error": sum(
                e["mean_abs_error"] for e in error_stats.values()
            )
            / max(len(error_stats), 1),
            "mean_relative_frobenius_error": sum(
                e["relative_frobenius_error"] for e in error_stats.values()
            )
            / max(len(error_stats), 1),
        },
    }
    export_log_path = os.path.join(export_dir, "metadata", "export_log.json")
    with open(export_log_path, "w") as f:
        json.dump(export_log, f, indent=2)
    print(f"  Wrote metadata/export_log.json")

    layer_map_path = os.path.join(export_dir, "metadata", "layer_map.json")
    with open(layer_map_path, "w") as f:
        json.dump(layer_map, f, indent=2)
    print(f"  Wrote metadata/layer_map.json")

    selected_cp = {
        "selected_checkpoint": export_job.selected_checkpoint,
    }
    cp_path = os.path.join(export_dir, "metadata", "selected_checkpoint.json")
    with open(cp_path, "w") as f:
        json.dump(selected_cp, f, indent=2)
    print(f"  Wrote metadata/selected_checkpoint.json")

    # --- tokenizer/ ---
    tokenizer_files = discovered.get("tokenizer_files", [])
    for tf in tokenizer_files:
        shutil.copy2(tf, os.path.join(export_dir, "tokenizer"))
    if tokenizer_files:
        print(f"  Copied {len(tokenizer_files)} tokenizer files")

    # --- activation/ ---
    smooth_dict = {}
    clip_dict = {}
    for module_name in quant_params:
        smooth_dict[module_name] = quant_params[module_name]["smooth"].contiguous()
        clip_dict[module_name] = quant_params[module_name]["clip"].contiguous()

    smooth_path = os.path.join(export_dir, "activation", "smooth.safetensors")
    safetensors_save(smooth_dict, smooth_path)
    print(f"  Wrote activation/smooth.safetensors ({len(smooth_dict)} entries)")

    clip_path = os.path.join(export_dir, "activation", "clip.safetensors")
    safetensors_save(clip_dict, clip_path)
    print(f"  Wrote activation/clip.safetensors ({len(clip_dict)} entries)")

    activation_meta = {
        "num_modules": len(quant_params),
        "module_names": list(quant_params.keys()),
        "smooth_file": "smooth.safetensors",
        "clip_file": "clip.safetensors",
        "weight_bits": export_job.weight_bits,
        "activation_bits": export_job.activation_bits,
    }
    activation_meta_path = os.path.join(export_dir, "activation", "activation_meta.json")
    with open(activation_meta_path, "w") as f:
        json.dump(activation_meta, f, indent=2)
    print(f"  Wrote activation/activation_meta.json")

    # --- weights/ -- per-layer ---
    for module_name in packed:
        safe_name = module_name.replace(".", "_")
        layer_dir = os.path.join(export_dir, "weights", safe_name)
        _ensure_dir(layer_dir)

        layer_data = packed[module_name]
        weight_int4_path = os.path.join(layer_dir, "weight_int4.bin")
        with open(weight_int4_path, "wb") as f:
            f.write(layer_data["weight_int4_bytes"])
        print(f"  Wrote weights/{safe_name}/weight_int4.bin")

        weight_scale_path = os.path.join(layer_dir, "weight_scale.fp16")
        with open(weight_scale_path, "wb") as f:
            f.write(layer_data["weight_scale_bytes"])
        print(f"  Wrote weights/{safe_name}/weight_scale.fp16")

        shape_path = os.path.join(layer_dir, "shape.json")
        with open(shape_path, "w") as f:
            json.dump({"rows": layer_data["shape"][0], "cols": layer_data["shape"][1]}, f)
        print(f"  Wrote weights/{safe_name}/shape.json")

        pack_meta_path = os.path.join(layer_dir, "pack_meta.json")
        with open(pack_meta_path, "w") as f:
            json.dump(layer_data["pack_meta"], f, indent=2)
        print(f"  Wrote weights/{safe_name}/pack_meta.json")

    # --- validation/ ---
    error_stats_path = os.path.join(export_dir, "validation", "layer_error_stats.json")
    with open(error_stats_path, "w") as f:
        json.dump(error_stats, f, indent=2)
    print(f"  Wrote validation/layer_error_stats.json")

    export_summary = {
        "export_dir": export_dir,
        "manifest": manifest_path,
        "num_modules_exported": len(packed),
        "module_names": list(packed.keys()),
        "error_stats": error_stats_path,
    }
    summary_path = os.path.join(export_dir, "validation", "export_summary.json")
    with open(summary_path, "w") as f:
        json.dump(export_summary, f, indent=2)
    print(f"  Wrote validation/export_summary.json")

    print(f"\n  Export complete: {export_dir}")
