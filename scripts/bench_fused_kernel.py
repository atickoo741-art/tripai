"""Microbenchmark: fused kernel vs old unpack+cuBLAS path.
Single-layer only — no model loading, ~100MB GPU memory max."""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "csrc"))


def pack_weights(w_q):
    rows, cols = w_q.shape
    if cols % 2 != 0:
        w_q = torch.cat([w_q, torch.zeros(rows, 1, dtype=w_q.dtype, device=w_q.device)], dim=1)
        cols = cols + 1
    w_q_flat = w_q.flatten().int()
    w_q_clamped = torch.clamp(w_q_flat, -8, 7).reshape(rows, cols)
    low = (w_q_clamped[:, ::2] & 0xF).to(torch.uint8)
    high = ((w_q_clamped[:, 1::2] & 0xF) << 4).to(torch.uint8)
    return (low | high).contiguous()


def bench_layer(D, K, N, warmup=10, repeats=100):
    w_f32 = torch.randn(D, K, device="cuda", dtype=torch.float32)
    qmax = 7
    w_scale = (w_f32.abs().amax(dim=1) / qmax).clamp(min=1e-8)
    w_q = torch.clamp(torch.round(w_f32 / w_scale.unsqueeze(1)), -8, 7).to(torch.int8)
    packed = pack_weights(w_q)
    w_scale_f16 = w_scale.to(torch.float16)

    x_dq = torch.randn(N, K, device="cuda", dtype=torch.float32)

    import peqat_cuda

    for _ in range(warmup):
        _ = peqat_cuda.linear_forward(x_dq, packed, w_scale_f16, D, K)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        _ = peqat_cuda.linear_forward(x_dq, packed, w_scale_f16, D, K)
    end.record()
    torch.cuda.synchronize()
    old_ms = start.elapsed_time(end) / repeats

    for _ in range(warmup):
        _ = peqat_cuda.fused_linear_forward(x_dq, packed, w_scale_f16, D, K)
    torch.cuda.synchronize()

    start.record()
    for _ in range(repeats):
        _ = peqat_cuda.fused_linear_forward(x_dq, packed, w_scale_f16, D, K)
    end.record()
    torch.cuda.synchronize()
    fused_ms = start.elapsed_time(end) / repeats

    speedup = old_ms / fused_ms if fused_ms > 0 else 0
    mem = packed.numel() + w_scale_f16.numel() * 2 + x_dq.numel() * 4
    return old_ms, fused_ms, speedup, mem


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1

    try:
        import peqat_cuda
    except ImportError:
        print("peqat_cuda not built. Run: cd csrc && python3 setup.py build_ext --inplace")
        return 1

    print(f"  {'shape':<20} {'N':>4} {'old(ms)':>10} {'fused(ms)':>10} {'speedup':>8} {'mem(MB)':>8}")
    print("  " + "-" * 64)

    shapes = [
        (4096, 4096, [1, 4, 16]),
        (14336, 4096, [1, 4]),
        (4096, 14336, [1, 4]),
        (1024, 1024, [1, 16, 64]),
    ]

    for D, K, Ns in shapes:
        for N in Ns:
            old_ms, fused_ms, speedup, mem_bytes = bench_layer(D, K, N)
            mem_mb = mem_bytes / (1024 * 1024)
            shape_str = f"{D}x{K}"
            print(f"  {shape_str:<20} {N:>4} {old_ms:>9.3f} {fused_ms:>9.3f} {speedup:>7.2f}x {mem_mb:>7.1f}")


if __name__ == "__main__":
    main()
