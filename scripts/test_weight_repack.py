"""Test weight repack from uint8 low_high to int32 TC-compatible format (8 s4 per int32).

Tests:
  1. Round-trip: pack → repack → unpack matches original
  2. Multiple shapes matching Llama-2-7B projections
  3. Edge cases (odd K, single row, etc.)
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def pack_int4_low_high(w_q):
    """PyTorch reference: pack int8[N, K] to uint8 low_high format."""
    rows, cols = w_q.shape
    w_clamped = torch.clamp(w_q, -8, 7).int()
    if cols % 2 != 0:
        w_clamped = torch.cat([w_clamped, torch.zeros(rows, 1, dtype=torch.int)], dim=1)
        cols_padded = cols + 1
    else:
        cols_padded = cols
    low = (w_clamped[:, ::2] & 0xF).to(torch.uint8)
    high = ((w_clamped[:, 1::2] & 0xF) << 4).to(torch.uint8)
    return (low | high).contiguous()


def repack_for_tc(packed_uint8, rows, cols):
    """Pack 8 INT4 values per int32 (dense) for MMA consumption.

    Each output int32:
      bits [0,4,8,12,16,20,24,28] = 8 consecutive INT4 values
    """
    K_packed = packed_uint8.shape[1]
    K_eff = K_packed * 2

    lo = (packed_uint8 & 0xF).to(torch.int8)
    hi = ((packed_uint8 >> 4) & 0xF).to(torch.int8)
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)

    w_q = torch.empty(rows, K_eff, dtype=torch.int8)
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


def unpack_int4_tc(w_packed, rows):
    """Unpack int32[N, K/8] (8 s4 per word) back to individual INT4 values."""
    n_packed = w_packed.shape[1]
    K = n_packed * 8
    w_q = torch.zeros(rows, K, dtype=torch.int32)
    for i in range(8):
        w_q[:, i::8] = (w_packed[:, :] >> (i * 4)) & 0xF
    w_q = torch.where(w_q > 7, w_q - 16, w_q)
    return w_q.to(torch.int8)


def test_shapes():
    test_cases = [
        ("small 4x16", 4, 16),
        ("q_proj 4096x4096", 4096, 4096),
        ("v_proj 4096x4096", 4096, 4096),
        ("gate_proj 4096x11008", 4096, 11008),
        ("down_proj 11008x4096", 11008, 4096),
        ("odd K 4x18", 4, 18),
        ("single row 1x4096", 1, 4096),
    ]

    for name, rows, cols in test_cases:
        w_orig = torch.randint(-8, 7, (rows, cols), dtype=torch.int8)
        packed_u8 = pack_int4_low_high(w_orig)
        packed_i32 = repack_for_tc(packed_u8, rows, cols)
        w_unpacked = unpack_int4_tc(packed_i32, rows)

        K_round = cols // 8 * 8
        if K_round > 0:
            w_cmp = w_unpacked[:, :K_round]
            w_ref = w_orig[:, :K_round]
        else:
            w_cmp = w_unpacked
            w_ref = w_orig
        match = (w_cmp == w_ref).all().item()
        status = "PASS" if match else "FAIL"
        mismatch = ((w_cmp != w_ref).float().mean() * 100).item() if not match else 0.0
        expected_shape = (rows, cols // 8) if cols >= 8 else (rows, 1)
        print(f"  [{status}] {name}: shape={tuple(packed_i32.shape)} expected={expected_shape} mismatch={mismatch:.2f}%")


def test_round_trip_integrity():
    """Test that repack → unpack is identity within INT4 range [-8, 7]."""
    rows, cols = 256, 4096
    w = torch.randint(-8, 7, (rows, cols), dtype=torch.int8)
    packed_u8 = pack_int4_low_high(w)
    packed_i32 = repack_for_tc(packed_u8, rows, cols)
    w_back = unpack_int4_tc(packed_i32, rows)
    assert (w == w_back).all(), "Round-trip failed!"
    print(f"  [PASS] round-trip integrity: {rows}x{cols}")


def test_export_format_parity():
    """Verify repack produces correct int32 bit pattern (8 s4 per word)."""
    rows, cols = 4, 16
    w = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8],
        [7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, -2, -3, -4, -5, -6, -7, -8, 0, 1, 2, 3, 4, 5, 6, 7],
    ], dtype=torch.int8)

    packed_u8 = pack_int4_low_high(w)
    packed_i32 = repack_for_tc(packed_u8, rows, cols)

    # Manual verification: first row [0,1,2,3,4,5,6,7]
    # Should pack as 0 | 1<<4 | 2<<8 | 3<<12 | 4<<16 | 5<<20 | 6<<24 | 7<<28
    expected_first = 0 | 1 << 4 | 2 << 8 | 3 << 12 | 4 << 16 | 5 << 20 | 6 << 24 | 7 << 28
    assert packed_i32[0, 0].item() == expected_first, (
        f"First word mismatch: {packed_i32[0,0].item():08x} vs {expected_first:08x}"
    )

    w_back = unpack_int4_tc(packed_i32, rows)
    match = (w == w_back).all().item()
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] export format parity")


if __name__ == "__main__":
    print("=== Weight Repack Test (uint8 → int32 TC, 8 s4/word) ===")
    print("\n--- Shape tests ---")
    test_shapes()
    print("\n--- Round-trip integrity ---")
    test_round_trip_integrity()
    print("\n--- Export format parity ---")
    test_export_format_parity()
    print("\nDone.")
