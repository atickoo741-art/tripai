"""Test INT4 Tensor Core GEMM (Phase C) on SM 12.1.

End-to-end test:
  1. preprocess_per_token_forward (M tokens, K features)
  2. repack_4_to_8_forward
  3. fused_int4_tc_per_token_forward
  4. Compare against bf16 reference matmul
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import peqat_cuda

torch.manual_seed(42)


def reference_forward(x_bf16, smooth, clip, w_q, w_scale, s_x=None):
    """Compute reference matching kernel: float32 arithmetic throughout.

    If s_x is provided (from kernel preprocess), uses it directly to
    avoid FP-non-associativity in the per-token max reduction.
    """
    M, K = x_bf16.shape
    N = w_q.shape[0]

    # Convert to float32 and divide (matches kernel: xv / sv in float32)
    x_f32 = x_bf16.float()
    smooth_f32 = smooth.float()
    clip_f32 = clip.float()

    x_scaled = x_f32 / smooth_f32
    x_clamped = x_scaled.clamp(-clip_f32, clip_f32)

    # Per-token max scale (use kernel's s_x if provided)
    if s_x is None:
        x_abs_max = x_clamped.abs().amax(dim=1)
        s_x = (x_abs_max / 7.0).clamp(min=1e-8)

    # Quantize
    x_q = (x_clamped / s_x[:, None]).round().clamp(-8, 7).to(torch.int8)

    # Dequantize
    x_dq = x_q.float() * s_x[:, None]

    # Matmul with dequantized weights
    w_dq = w_q.float() * w_scale[:, None]
    y_ref = x_dq @ w_dq.T
    return y_ref.bfloat16()


def pack_int4_low_high(w):
    """Pack int8[N, K] to uint8 low_high format."""
    w = torch.clamp(w, -8, 7).int()
    low = (w[:, ::2] & 0xF).to(torch.uint8)
    high = ((w[:, 1::2] & 0xF) << 4).to(torch.uint8)
    return (low | high).contiguous()


def repack_for_tc(packed_uint8, rows, cols):
    """Pack 8 INT4 values per int32 (same as loader._repack_for_tc)."""
    K_packed = packed_uint8.shape[1]
    K_eff = K_packed * 2
    lo = (packed_uint8 & 0xF).to(torch.int8)
    hi = ((packed_uint8 >> 4) & 0xF).to(torch.int8)
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)
    dev = packed_uint8.device
    w_q = torch.empty(rows, K_eff, dtype=torch.int8, device=dev)
    w_q[:, ::2] = lo
    w_q[:, 1::2] = hi
    if K_eff > cols:
        w_q = w_q[:, :cols]
    w_i32 = w_q.to(torch.int32) & 0xF
    K = cols
    K8 = K // 8
    w_groups = w_i32[:, :K8 * 8].reshape(rows, K8, 8)
    packed = (
        w_groups[:, :, 0]
        | (w_groups[:, :, 1] << 4)
        | (w_groups[:, :, 2] << 8)
        | (w_groups[:, :, 3] << 12)
        | (w_groups[:, :, 4] << 16)
        | (w_groups[:, :, 5] << 20)
        | (w_groups[:, :, 6] << 24)
        | (w_groups[:, :, 7] << 28)
    )
    return packed


def test_tc_gemm():
    shapes = [
        ("tiny 1x64", 1, 64, 64),
        ("small 1x4096", 1, 4096, 4096),
        ("small 4x4096", 4, 4096, 4096),
        ("prefill 16x4096", 16, 4096, 4096),
        ("gate 1x11008", 1, 4096, 11008),
        ("down 1x4096", 1, 11008, 4096),
    ]

    for name, M, K, N in shapes:
        if K % 64 != 0:
            K = (K + 63) // 64 * 64
        if N % 64 != 0:
            N = (N + 63) // 64 * 64

        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        smooth = torch.randn(K, dtype=torch.bfloat16, device="cuda").abs() + 0.5
        clip = torch.randn(K, dtype=torch.bfloat16, device="cuda").abs() + 0.5

        w = torch.randint(-8, 7, (N, K), dtype=torch.int8, device="cuda")
        w_scale = torch.randn(N, dtype=torch.float32, device="cuda").abs() + 0.1

        packed_u8 = pack_int4_low_high(w)
        w_packed_tc = repack_for_tc(packed_u8, N, K).to(device="cuda")

        # TC preprocess first (to get kernel's s_x, avoiding FP-non-associativity)
        x_packed, s_x = peqat_cuda.preprocess_per_token_forward(x, smooth, clip)

        # Compute reference using kernel's s_x
        y_ref = reference_forward(x, smooth, clip, w, w_scale, s_x=s_x)
        x_packed_8 = peqat_cuda.repack_4_to_8_forward(x_packed)

        s_x_padded = s_x
        x_8_padded = x_packed_8
        if M < 16:
            pad = 16 - M
            x_8_padded = torch.cat([
                x_packed_8,
                torch.zeros(pad, K // 8, dtype=torch.int32, device="cuda"),
            ], dim=0)
            s_x_padded = torch.cat([
                s_x,
                torch.ones(pad, device="cuda"),
            ], dim=0)

        y_tc = peqat_cuda.fused_int4_tc_per_token_forward(
            x_8_padded, w_packed_tc, w_scale.float(), s_x_padded,
            N, K,
        )
        y_tc = y_tc[:M, :N].contiguous()

        cos = torch.nn.CosineSimilarity(dim=-1)
        sim = cos(y_ref.float().flatten(), y_tc.float().flatten()).item()
        max_err = (y_ref.float() - y_tc.float()).abs().max().item()

        status = "PASS" if sim > 0.95 and max_err < 1.0 else "FAIL"
        print(f"  [{status}] {name}: M={M} K={K} N={N} cos_sim={sim:.6f} max_err={max_err:.4f}")


def test_repack_4_to_8():
    """Verify repack_4_to_8 kernel produces correct output."""
    M, K = 4, 64
    x_packed = torch.randint(0, 256, (M, K // 4), dtype=torch.int32, device="cuda")
    x_8 = peqat_cuda.repack_4_to_8_forward(x_packed)

    # Manual verification
    for m in range(M):
        for i in range(K // 8):
            lo = x_packed[m, i * 2 + 0].item()
            hi = x_packed[m, i * 2 + 1].item()
            expected = lo | (hi << 16)
            actual = x_8[m, i].item()
            assert actual == expected, f"Row {m}, i={i}: {actual:08x} != {expected:08x}"

    print(f"  [PASS] repack_4_to_8: {M}x{K/4} → {M}x{K/8}")


if __name__ == "__main__":
    print("=== INT4 TC GEMM Test (SM 12.1) ===")
    print("\n--- Repack 4→8 ---")
    test_repack_4_to_8()
    print("\n--- TC GEMM ---")
    test_tc_gemm()
    print("\nDone.")
