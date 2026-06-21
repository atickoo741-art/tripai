"""Direct MMA test: feed known INT4 data, compare element-by-element."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import peqat_cuda

torch.manual_seed(42)

def pack_8_per_int32(w):
    """Pack [N, K] int8 values (in [-8,7]) into [N, K/8] int32 (8 per word)."""
    N, K = w.shape
    w_i32 = (w.to(torch.int32) & 0xF)
    K8 = K // 8
    w_groups = w_i32[:, :K8 * 8].reshape(N, K8, 8)
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

M, K, N = 1, 64, 64

x_q = torch.randint(-8, 7, (M, K), dtype=torch.int8, device="cuda")
w_q = torch.randint(-8, 7, (N, K), dtype=torch.int8, device="cuda")

# Pack x and w for TC
x_packed = pack_8_per_int32(x_q)  # [1, 8]
w_packed = pack_8_per_int32(w_q)  # [64, 8]

# Pad M to 16
x_packed = torch.cat([x_packed, torch.zeros(15, K//8, dtype=torch.int32, device="cuda")], dim=0)

# Scales (unit scale for this test)
s_x = torch.ones(16, dtype=torch.float32, device="cuda")
w_scale = torch.ones(N, dtype=torch.float32, device="cuda")

# Run TC kernel
y_tc = peqat_cuda.fused_int4_tc_per_token_forward(
    x_packed, w_packed, w_scale, s_x, N, K,
)
y_tc = y_tc[:M, :N].contiguous()

# Reference: INT4 dot product (no scaling)
y_ref = (x_q.float() @ w_q.float().T).bfloat16()

# Compare element by element
print(f"M={M}, K={K}, N={N}")
print(f"\nFirst 8 elements of first row:")
print(f"  Ref: {y_ref[0, :8].tolist()}")
print(f"  TC:  {y_tc[0, :8].tolist()}")

diff = (y_ref.float() - y_tc.float()).abs()
print(f"\n  Max diff: {diff.max().item():.4f}")
print(f"  Mean diff: {diff.mean().item():.4f}")

cos = torch.nn.CosineSimilarity(dim=-1)
sim = cos(y_ref.float().flatten(), y_tc.float().flatten()).item()
print(f"  Cos sim: {sim:.6f}")

# Check individual elements
print(f"\nElement-by-element comparison (row 0):")
for n in range(min(N, 16)):
    ref_val = y_ref[0, n].item()
    tc_val = y_tc[0, n].item()
    expected_dot = (x_q[0].float() * w_q[n].float()).sum().item()
    match = "✓" if abs(ref_val - tc_val) < 1.0 else "✗"
    print(f"  [{n:2d}] ref={ref_val:6.0f} tc={tc_val:6.0f} dot={expected_dot:6.0f} {match}")
