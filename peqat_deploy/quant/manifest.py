import json
import os
import subprocess
from datetime import datetime


def build_manifest(
    export_job,
    layer_map: dict,
    quant_params: dict,
    packed: dict,
    artifact_paths: dict,
    packed_folded: dict | None = None,
) -> dict:
    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            .decode()
            .strip()
        )
    except Exception:
        git_commit = None

    activation_file_paths = {
        "smooth": artifact_paths.get("smooth"),
        "clip": artifact_paths.get("clip"),
        "activation_meta": artifact_paths.get("activation_meta"),
    }

    weight_artifact_paths = {}
    for module_name in layer_map:
        safe_name = module_name.replace(".", "_")
        entry = {
            "weight_int4": f"weights/{safe_name}/weight_int4.bin",
            "weight_scale": f"weights/{safe_name}/weight_scale.fp16",
            "shape": f"weights/{safe_name}/shape.json",
            "pack_meta": f"weights/{safe_name}/pack_meta.json",
        }
        if packed_folded and module_name in packed_folded:
            entry["weight_int4_folded"] = f"weights/{safe_name}/weight_int4_folded.bin"
            entry["weight_scale_folded"] = f"weights/{safe_name}/weight_scale_folded.bf16"
        weight_artifact_paths[module_name] = entry

    manifest = {
        "manifest_version": 1,
        "timestamp": datetime.utcnow().isoformat(),
        "git_commit": git_commit,
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
        "num_exported_layers": len(layer_map),
        "activation_file_paths": activation_file_paths,
        "weight_artifact_paths": weight_artifact_paths,
        "tokenizer_path": "tokenizer",
        "runtime_layout_version": packed.get("_layout_version", 1)
        if isinstance(packed, dict)
        else 1,
    }

    return manifest


def build_hadamard_manifest(
    export_job,
    layer_map: dict,
    packed: dict,
) -> dict:
    """
    Build manifest for Hadamard PE-QAT export.
    No activation clip/smooth entries — H is fixed, no learned QAT params.
    """
    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            .decode()
            .strip()
        )
    except Exception:
        git_commit = None

    weight_artifact_paths = {}
    for module_name in layer_map:
        safe_name = module_name.replace(".", "_")
        entry = {
            "weight_int4": f"weights/{safe_name}/weight_int4.bin",
            "weight_scale": f"weights/{safe_name}/weight_scale.fp16",
            "shape": f"weights/{safe_name}/shape.json",
            "pack_meta": f"weights/{safe_name}/pack_meta.json",
        }
        weight_artifact_paths[module_name] = entry

    manifest = {
        "manifest_version": 2,
        "format": "hadamard",
        "timestamp": datetime.utcnow().isoformat(),
        "git_commit": git_commit,
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
        "num_exported_layers": len(layer_map),
        "hadamard_rotation": True,
        "note": "No clip/smooth params. Input H applied at runtime in preprocess kernel.",
        "weight_artifact_paths": weight_artifact_paths,
        "tokenizer_path": "tokenizer",
        "runtime_layout_version": packed.get("_layout_version", 1)
        if isinstance(packed, dict)
        else 1,
    }

    return manifest
