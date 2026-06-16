import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from peqat_deploy.eval.harness import run_suite, measure_memory, measure_prefill
from peqat_deploy.runtime.loader import build_cuda_model


def build_fp16_model(base_model_name: str, device: str = "cuda"):
    print(f"Loading FP16 model {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def main():
    export_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if not export_dir or not os.path.isdir(export_dir):
        print(f"Usage: python {sys.argv[0]} <export_dir>")
        return 1

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 1

    with open(os.path.join(export_dir, "manifest.json")) as f:
        manifest = json.load(f)
    base_model_name = manifest["base_model_name"]

    output_dir = os.path.join(
        os.path.dirname(export_dir),
        os.path.basename(export_dir).replace("_export", "_benchmarks"),
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")

    seq_lens = [128, 512, 2048]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --- FP16 baseline ---
    model_fp16, tokenizer = build_fp16_model(base_model_name)
    fp16_mem = measure_memory(model_fp16)
    run_suite(model_fp16, tokenizer, "fp16_baseline", os.path.join(output_dir, "fp16"), seq_lens)
    del model_fp16
    torch.cuda.empty_cache()

    # --- CUDA quantized ---
    model_cuda, tokenizer = build_cuda_model(export_dir, base_model_name)
    cuda_mem = measure_memory(model_cuda)
    run_suite(model_cuda, tokenizer, "cuda_peqat", os.path.join(output_dir, "cuda_peqat"), seq_lens)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'FP16':>12} {'CUDA PEQAT':>12}")
    print(f"{'':-<30} {'':->12} {'':->12}")
    print(f"{'GPU memory (GB)':<30} {fp16_mem['gpu_allocated_gb']:>12.1f} {cuda_mem['gpu_allocated_gb']:>12.1f}")

    summary = {
        "fp16": {"memory_gb": fp16_mem["gpu_allocated_gb"]},
        "cuda_peqat": {"memory_gb": cuda_mem["gpu_allocated_gb"]},
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults in: {output_dir}")


if __name__ == "__main__":
    main()
