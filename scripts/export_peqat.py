import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from peqat_deploy.config.loader import load_export_job
from peqat_deploy.io.discover import discover_source_dir
from peqat_deploy.io.hf_loader import load_base_model
from peqat_deploy.io.peft_loader import load_adapter
from peqat_deploy.io.save_artifacts import save_artifacts
from peqat_deploy.quant.extract_quant_params import extract_quant_params
from peqat_deploy.quant.fold_smoothing import fold_smoothing
from peqat_deploy.quant.manifest import build_manifest
from peqat_deploy.quant.merge_lora import build_layer_map, merge_lora_weights
from peqat_deploy.quant.pack_int4 import pack_int4
from peqat_deploy.quant.quantize_weights import quantize_weights
from peqat_deploy.quant.fold_act_scale import fold_act_scale_into_weights


def main():
    parser = argparse.ArgumentParser(
        description="Export a trained PE-QAT checkpoint into packed INT4 inference artifacts"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Path to PE-QAT training output directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint subdirectory (e.g. checkpoint-1320)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Export directory (default: <source_basename>_export)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Dtype for export computation",
    )
    parser.add_argument(
        "--save-validation",
        action="store_true",
        help="Run and save validation outputs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing export directory",
    )
    parser.add_argument(
        "--fold-weights",
        action="store_true",
        help="Fold activation scale s_x into weights for INT4 Tensor Core runtime",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for model loading and merging",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PE-QAT Export Pipeline")
    print("=" * 60)

    num_main_steps = "8" if not args.fold_weights else "9"

    # Step 1: Discover source
    print(f"\n[1/{num_main_steps}] Discovering source directory...")
    discovered = discover_source_dir(args.source_dir, args.checkpoint)
    print(f"  Source: {discovered['source_dir']}")
    if discovered["selected_checkpoint"]:
        print(f"  Checkpoint: {os.path.basename(discovered['selected_checkpoint'])}")
    else:
        print(f"  Using final model at source dir")

    # Step 2: Load config
    print(f"\n[2/{num_main_steps}] Loading export configuration...")
    overrides = {}
    if args.output_dir:
        overrides["export_dir"] = args.output_dir
    if args.dtype:
        overrides["dtype"] = args.dtype
    job = load_export_job(discovered, **overrides)
    print(f"  Base model: {job.base_model_name}")
    print(f"  Target modules: {job.target_modules}")
    print(f"  Export dir: {job.export_dir}")

    # Step 3: Load base model and adapter
    print(f"\n[3/{num_main_steps}] Loading base model and adapter...")
    device = args.device
    base_model, tokenizer = load_base_model(job.base_model_name, device=device)
    print(f"  Base model loaded: {type(base_model).__name__}")

    adapter_path = discovered["selected_checkpoint"] or discovered["source_dir"]
    peft_model = load_adapter(base_model, adapter_path)
    print(f"  Adapter loaded")

    # Step 4: Merge LoRA weights
    print(f"\n[4/{num_main_steps}] Merging LoRA weights...")
    merged_weights = merge_lora_weights(
        peft_model,
        job.target_modules,
        job.lora_rank,
        job.lora_alpha,
    )
    print(f"  Merged {len(merged_weights)} target modules")

    del base_model
    del peft_model
    gc.collect()
    print("  Freed base model and adapter from memory")

    # Step 5: Extract quantization parameters
    print(f"\n[5/{num_main_steps}] Extracting quantization parameters...")
    if not discovered["quantization_params"]:
        raise FileNotFoundError(
            "quantization_params.pth not found in checkpoint directory"
        )
    quant_params = extract_quant_params(
        discovered["quantization_params"], merged_weights
    )
    print(f"  Extracted params for {len(quant_params)} modules")

    # Step 6: Fold smoothing onto weights
    print(f"\n[6/{num_main_steps}] Folding smoothing onto weights...")
    smoothed_weights = fold_smoothing(merged_weights, quant_params)
    print(f"  Smoothed {len(smoothed_weights)} weight tensors")

    # Step 7: Quantize and pack
    print(f"\n[7/{num_main_steps}] Quantizing and packing weights...")
    quantized, error_stats = quantize_weights(smoothed_weights)
    print(f"  Quantized {len(quantized)} modules")

    packed = pack_int4(quantized, smoothing_folded=True)
    print(f"  Packed {len(packed)} modules")

    # Step 8: Fold activation scales into weights for INT4 TC (optional)
    packed_folded = None
    next_step = 8
    if args.fold_weights:
        print(f"\n[{next_step}/{num_main_steps}] Folding activation scales into weights for INT4 Tensor Core...")
        next_step += 1
        clip_dict = {k: v["clip"] for k, v in quant_params.items()}
        packed_folded = fold_act_scale_into_weights(packed, clip_dict)
        print(f"  Folded {len(packed_folded)} modules")

    # Step 9 (or 8): Save artifacts
    print(f"\n[{next_step}/{num_main_steps}] Saving export artifacts...")
    layer_map = build_layer_map(merged_weights, job.export_dir)

    artifact_paths = {
        "smooth": "activation/smooth.safetensors",
        "clip": "activation/clip.safetensors",
        "activation_meta": "activation/activation_meta.json",
    }

    manifest = build_manifest(job, layer_map, quant_params, packed, artifact_paths, packed_folded=packed_folded)

    save_artifacts(
        export_dir=job.export_dir,
        export_job=job,
        discovered=discovered,
        merged_weights=merged_weights,
        quant_params=quant_params,
        quantized=quantized,
        packed=packed,
        packed_folded=packed_folded,
        error_stats=error_stats,
        layer_map=layer_map,
        manifest=manifest,
        overwrite=job.overwrite or args.overwrite,
    )

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
