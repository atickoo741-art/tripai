"""Test the per-token preprocess CUDA kernel against a PyTorch reference.

Usage:
    python scripts/test_preprocess_per_token_kernel.py

Tests:
    1. Random small shapes (for quick validation)
    2. Llama-2-7B scale shapes (M=1..128, K=4096)
    3. Edge cases (uniform inputs, zeros, extremes)
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import peqat_cuda
except ImportError:
    print("peqat_cuda not found. Build it first:")
    print("  cd csrc && python setup.py build_ext --inplace")
    sys.exit(1)


def reference_preprocess_per_token(x, smooth, clip):
    """Pure PyTorch reference matching the CUDA kernel logic."""
    qmax = 7.0
    qmin = -8.0

    M, K = x.shape
    x_f = x.float()
    s_f = smooth.float()
    c_f = clip.float()

    x_s = x_f / s_f.unsqueeze(0)
    x_c = torch.clamp(x_s, -c_f, c_f)
    max_abs = x_c.abs().amax(dim=1)
    s_x = (max_abs / qmax).clamp(min=1e-8)

    x_scaled = x_c / s_x.unsqueeze(1)
    x_q = torch.clamp(torch.round(x_scaled), qmin, qmax).to(torch.int32)

    n_packed = K // 4
    x_packed = torch.zeros(M, n_packed, dtype=torch.int32)
    for i in range(4):
        x_packed |= ((x_q[:, i::4] & 0xF) << (i * 4))

    return x_packed, s_x


def unpack_int4(x_packed, M, K):
    """Unpack int32[M, K/4] to individual INT4 values as int8[M, K]."""
    n_packed = K // 4
    x_q = torch.zeros(M, K, dtype=torch.int32, device=x_packed.device)
    for i in range(4):
        x_q[:, i::4] = (x_packed[:, :n_packed] >> (i * 4)) & 0xF
    x_q = torch.where(x_q > 7, x_q - 16, x_q)
    return x_q.to(torch.int8)


def test_shapes():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  SKIP: no CUDA device available")
        return

    tests = [
        ("small 2x16", 2, 16),
        ("small 4x32", 4, 32),
        ("single token 1x4096", 1, 4096),
        ("batch 8x4096", 8, 4096),
        ("prefill 128x4096", 128, 4096),
        ("decode 1x4096", 1, 4096),
    ]

    for name, M, K in tests:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 3.0
        smooth = torch.rand(K, dtype=torch.bfloat16, device=device).abs() + 0.1
        clip = torch.rand(K, dtype=torch.bfloat16, device=device).abs() * 5.0 + 0.01

        ref_packed, ref_s_x = reference_preprocess_per_token(
            x.cpu().float(), smooth.cpu().float(), clip.cpu().float()
        )
        cuda_packed, cuda_s_x = peqat_cuda.preprocess_per_token_forward(x, smooth, clip)

        s_x_ok = torch.allclose(cuda_s_x.cpu(), ref_s_x, atol=1e-5)

        ref_q = unpack_int4(ref_packed, M, K)
        cuda_q = unpack_int4(cuda_packed.cpu(), M, K)
        q_match = (cuda_q == ref_q).all().item()

        status = "PASS" if (s_x_ok and q_match) else "FAIL"
        s_x_max_diff = (cuda_s_x.cpu() - ref_s_x).abs().max().item()
        mismatch_pct = ((cuda_q != ref_q).float().mean() * 100).item()

        print(f"  [{status}] {name}: s_x diff={s_x_max_diff:.6e}, quant mismatch={mismatch_pct:.2f}%")


def test_edge_cases():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return

    M, K = 4, 4096

    cases = [
        ("all zeros", torch.zeros(M, K, dtype=torch.bfloat16, device=device)),
        ("all ones", torch.ones(M, K, dtype=torch.bfloat16, device=device)),
        ("all positive", torch.rand(M, K, dtype=torch.bfloat16, device=device) * 10.0),
        ("all negative", torch.rand(M, K, dtype=torch.bfloat16, device=device) * -10.0),
        ("extreme 100x", torch.randn(M, K, dtype=torch.bfloat16, device=device) * 100.0),
    ]

    smooth = torch.rand(K, dtype=torch.bfloat16, device=device).abs() + 0.1
    clip = torch.rand(K, dtype=torch.bfloat16, device=device).abs() * 5.0 + 0.01

    for name, x in cases:
        ref_packed, ref_s_x = reference_preprocess_per_token(
            x.cpu().float(), smooth.cpu().float(), clip.cpu().float()
        )
        cuda_packed, cuda_s_x = peqat_cuda.preprocess_per_token_forward(x, smooth, clip)

        s_x_ok = torch.allclose(cuda_s_x.cpu(), ref_s_x, atol=1e-5)
        ref_q = unpack_int4(ref_packed, M, K)
        cuda_q = unpack_int4(cuda_packed.cpu(), M, K)
        q_match = (cuda_q == ref_q).all().item()

        status = "PASS" if (s_x_ok and q_match) else "FAIL"
        print(f"  [{status}] {name}")


if __name__ == "__main__":
    print("=== Per-Token Preprocess Kernel Test ===")
    print(f"CUDA available: {torch.cuda.is_available()}")

    print("\n--- Shape tests ---")
    test_shapes()

    print("\n--- Edge case tests ---")
    test_edge_cases()

    print("\nDone.")
