import csv
import json
import os
import time

import torch


def measure_prefill(model, input_ids, warmup=3, repeats=10):
    for _ in range(warmup):
        model(input_ids)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeats):
        model(input_ids)
    torch.cuda.synchronize()
    avg = (time.time() - t0) / repeats
    tokens = input_ids.shape[1]
    return {"ms": avg * 1000, "tokens": tokens, "ms_per_token": avg * 1000 / tokens}


def measure_decode(model, prompt_len=128, decode_tokens=50, warmup=3, repeats=5):
    input_ids = torch.randint(0, 100, (1, prompt_len), device="cuda")
    for _ in range(warmup):
        ids = input_ids.clone()
        for _ in range(5):
            logits = model(ids).logits
            ids = torch.cat([ids, logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)], dim=-1)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        ids = input_ids.clone()
        t0 = time.time()
        for _ in range(decode_tokens):
            logits = model(ids).logits
            ids = torch.cat([ids, logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)], dim=-1)
        torch.cuda.synchronize()
        times.append((time.time() - t0) / decode_tokens)

    avg_s = sum(times) / len(times)
    return {
        "ms_per_token": avg_s * 1000,
        "tokens_per_sec": 1.0 / avg_s,
    }


def measure_memory(model):
    if not torch.cuda.is_available():
        return {"gpu_mem_gb": 0}
    allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
    torch.cuda.reset_peak_memory_stats()
    return {"gpu_allocated_gb": round(allocated, 2), "gpu_reserved_gb": round(reserved, 2)}


def run_suite(model, tokenizer, suite_name: str, output_dir: str, seq_lens=None):
    if seq_lens is None:
        seq_lens = [128, 512, 2048]

    os.makedirs(output_dir, exist_ok=True)

    results = {"suite": suite_name}

    print(f"\n{'='*60}")
    print(f"Benchmark Suite: {suite_name}")
    print(f"{'='*60}")

    print("\n--- Prefill ---")
    prefill_rows = []
    for sl in seq_lens:
        input_ids = torch.randint(0, 100, (1, sl), device="cuda")
        r = measure_prefill(model, input_ids)
        r["seq_len"] = sl
        prefill_rows.append(r)
        print(f"  seq_len={sl:>5}: {r['ms']:8.1f}ms ({r['ms_per_token']:.2f}ms/tok)")
    results["prefill"] = prefill_rows

    print("\n--- Decode ---")
    decode_rows = []
    for sl in seq_lens:
        r = measure_decode(model, prompt_len=sl)
        r["prompt_len"] = sl
        decode_rows.append(r)
        print(f"  prompt_len={sl:>5}: {r['ms_per_token']:.1f}ms/tok ({r['tokens_per_sec']:.1f} tok/s)")
    results["decode"] = decode_rows

    print("\n--- Memory ---")
    mem = measure_memory(model)
    results["memory"] = mem
    print(f"  GPU allocated: {mem['gpu_allocated_gb']} GB")
    print(f"  GPU reserved:  {mem['gpu_reserved_gb']} GB")

    def write_csv(filename, fieldnames, rows):
        path = os.path.join(output_dir, filename)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"  Wrote {path}")

    write_csv("prefill.csv", ["seq_len", "ms", "tokens", "ms_per_token"], prefill_rows)
    write_csv("decode.csv", ["prompt_len", "ms_per_token", "tokens_per_sec"], decode_rows)

    return results
