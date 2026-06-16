import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
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
        manifest = json.load(f)
    base_model_name = manifest["base_model_name"]

    print("Building CUDA PE-QAT model...")
    t0 = time.time()
    model, tokenizer = build_cuda_model(export_dir, base_model_name, device="cuda")
    build_time = time.time() - t0
    print(f"  Build time: {build_time:.1f}s")

    prompt = "The meaning of life is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print(f"\nRunning forward pass...")
    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    forward_time = time.time() - t0
    print(f"  Forward time: {forward_time:.3f}s")

    next_token_id = outputs.logits[0, -1].argmax().item()
    next_token = tokenizer.decode(next_token_id, skip_special_tokens=True)
    print(f"  Logits shape: {outputs.logits.shape}")
    print(f"  Predicted next token: '{next_token}' (id={next_token_id})")

    print(f"\nGenerating 10 tokens...")
    input_ids = inputs["input_ids"]
    t0 = time.time()
    for i in range(10):
        logits = model(input_ids).logits
        next_id = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_id], dim=-1)
    gen_time = time.time() - t0
    generated = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"  Generated: {generated}")
    print(f"  Generation time: {gen_time:.3f}s ({gen_time/10*1000:.1f}ms/token)")
    print(f"  Output length: {input_ids.shape[1]} tokens")

    print("\nPASS: CUDA runtime end-to-end forward pass and generation successful")
    return 0


if __name__ == "__main__":
    sys.exit(main())
