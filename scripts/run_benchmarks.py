import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from peqat_deploy.eval.harness import run_suite, measure_memory
from peqat_deploy.runtime.loader import build_cuda_model


def main():
    export_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if not export_dir or not os.path.isdir(export_dir):
        print(f"Usage: python {sys.argv[0]} <export_dir>")
        return 1
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 1

    with open(os.path.join(export_dir, "manifest.json")) as f:
        base_model_name = json.load(f)["base_model_name"]

    output_dir = os.path.join(
        os.path.dirname(export_dir),
        os.path.basename(export_dir).replace("_export", "_benchmarks"),
    )
    os.makedirs(output_dir, exist_ok=True)

    # --- FP16 baseline (quick: seq_len 128 only) ---
    print("Loading FP16 model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.float16, low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()

    fp16_mem = measure_memory(model)
    run_suite(model, tokenizer, "fp16_baseline", os.path.join(output_dir, "fp16"), seq_lens=[128])

    del model
    torch.cuda.empty_cache()

    # --- CUDA quantized ---
    model, tokenizer = build_cuda_model(export_dir, base_model_name)
    cuda_mem = measure_memory(model)
    run_suite(model, tokenizer, "cuda_peqat", output_dir, seq_lens=[128, 512, 2048])

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
