import gc
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from peqat_deploy.eval.harness import run_suite, measure_memory
from peqat_deploy.runtime.cpu_model import build_packed_model


def force_free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    args = sys.argv[1:]
    device = "cuda"
    hadamard = False
    export_dir = None
    i = 0
    while i < len(args):
        if args[i] == "--device" and i + 1 < len(args):
            device = args[i + 1]
            i += 2
        elif args[i] == "--hadamard":
            hadamard = True
            i += 1
        else:
            export_dir = args[i]
            i += 1
    if not export_dir or not os.path.isdir(export_dir):
        print(f"Usage: python {sys.argv[0]} [--device cpu|cuda] [--hadamard] <export_dir>")
        return 1

    with open(os.path.join(export_dir, "manifest.json")) as f:
        manifest = json.load(f)
    base_model_name = manifest["base_model_name"]

    base_output = os.path.join(
        os.path.dirname(export_dir),
        os.path.basename(export_dir).replace("_export", "_benchmarks"),
    )
    output_dir = base_output
    n = 1
    while os.path.exists(output_dir):
        output_dir = f"{base_output}_{n}"
        n += 1
    os.makedirs(output_dir)

    force_free()

    if device == "cuda":
        from peqat_deploy.runtime.loader import build_cuda_model
        if hadamard:
            from peqat_deploy.runtime.loader import build_hadamard_cuda_model

        # FP16 baseline
        print("Loading FP16 model on GPU...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, dtype=torch.float16, low_cpu_mem_usage=True,
        ).to("cuda")
        model.eval()
        fp16_mem = measure_memory(model)
        run_suite(model, tokenizer, "fp16_baseline", os.path.join(output_dir, "fp16"), seq_lens=[128])
        del model, tokenizer
        force_free()

        # CUDA PEQAT (standard or Hadamard)
        label = "hadamard_peqat" if hadamard else "cuda_peqat"
        if hadamard:
            model, tokenizer = build_hadamard_cuda_model(export_dir, base_model_name)
        else:
            model, tokenizer = build_cuda_model(export_dir, base_model_name)
        cuda_mem = measure_memory(model)
        run_suite(model, tokenizer, label, output_dir, seq_lens=[128, 512])

        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"{'Metric':<30} {'FP16':>12} {label.upper():>12}")
        print(f"{'':-<30} {'':->12} {'':->12}")
        print(f"{'GPU memory (GB)':<30} {fp16_mem['gpu_allocated_gb']:>12.1f} {cuda_mem['gpu_allocated_gb']:>12.1f}")
    else:
        # CPU PEQAT (no GPU contention)
        print(f"Loading model on CPU...")
        model, tokenizer = build_packed_model(export_dir, base_model_name, device="cpu")
        run_suite(model, tokenizer, "cpu_peqat", output_dir, seq_lens=[128, 512])

    print(f"\nResults in: {output_dir}")


if __name__ == "__main__":
    main()
