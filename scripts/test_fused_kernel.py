import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def pack_weights(w_q: torch.Tensor) -> torch.Tensor:
    rows, cols = w_q.shape
    if cols % 2 != 0:
        w_q = torch.cat([w_q, torch.zeros(rows, 1, dtype=w_q.dtype, device=w_q.device)], dim=1)
    w_q_flat = w_q.flatten().int()
    w_q_clamped = torch.clamp(w_q_flat, -8, 7).reshape(rows, cols)
    low = (w_q_clamped[:, ::2] & 0xF).to(torch.uint8)
    high = ((w_q_clamped[:, 1::2] & 0xF) << 4).to(torch.uint8)
    return (low | high).contiguous()


def reference_linear(x_dq, w_q, w_scale):
    w_dq = w_q.float() * w_scale.unsqueeze(1).float()
    return x_dq.float() @ w_dq.T


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 1

    try:
        import peqat_cuda
    except ImportError:
        print("FAIL: peqat_cuda extension not built")
        return 1

    torch.manual_seed(42)
    D, K = 4096, 4096
    qmax = 7

    print("=== Correctness ===")
    for N in [1, 2, 4, 8]:
        w_f32 = torch.randn(D, K, device="cuda", dtype=torch.float32)
        w_scale = (w_f32.abs().amax(dim=1) / qmax).clamp(min=1e-8)
        w_q = torch.clamp(torch.round(w_f32 / w_scale.unsqueeze(1)), -8, 7).to(torch.int8)
        packed = pack_weights(w_q)
        x_dq = torch.randn(N, K, device="cuda", dtype=torch.float32)

        y_ref = reference_linear(x_dq, w_q, w_scale)
        y_fused = peqat_cuda.linear_forward_fused(x_dq, packed, w_scale.to(torch.float16), D, K)

        max_err = (y_ref - y_fused).abs().max().item()
        ok = torch.allclose(y_ref, y_fused, atol=1e-4, rtol=1e-3)
        print(f"  N={N:>2d}: allclose={str(ok):>5}  max_err={max_err:.2e}")
        if not ok:
            print("FAIL")
            return 1

    print("\n=== Decode speed (N=1) ===")
    w_f32 = torch.randn(D, K, device="cuda", dtype=torch.float32)
    w_scale = (w_f32.abs().amax(dim=1) / qmax).clamp(min=1e-8)
    w_q = torch.clamp(torch.round(w_f32 / w_scale.unsqueeze(1)), -8, 7).to(torch.int8)
    packed = pack_weights(w_q)

    x_dq = torch.randn(1, K, device="cuda", dtype=torch.float32)

    # Warmup
    for _ in range(10):
        peqat_cuda.linear_forward_fused(x_dq, packed, w_scale.to(torch.float16), D, K)
        peqat_cuda.linear_forward(x_dq, packed, w_scale.to(torch.float16), D, K)
    torch.cuda.synchronize()

    # Benchmark fused
    t0 = time.time()
    for _ in range(100):
        peqat_cuda.linear_forward_fused(x_dq, packed, w_scale.to(torch.float16), D, K)
    torch.cuda.synchronize()
    fused_ms = (time.time() - t0) / 100 * 1000

    # Benchmark old path
    t0 = time.time()
    for _ in range(100):
        peqat_cuda.linear_forward(x_dq, packed, w_scale.to(torch.float16), D, K)
    torch.cuda.synchronize()
    old_ms = (time.time() - t0) / 100 * 1000

    print(f"  Old unpack+cuBLAS:  {old_ms:.1f}ms")
    print(f"  Fused decode:       {fused_ms:.1f}ms")
    print(f"  Speedup:            {old_ms/fused_ms:.1f}x")

    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
