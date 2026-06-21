"""Find exact K-quad mapping: set a single K position to 1, check which column sees it."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import peqat_cuda

def pack_8_per_int32(w):
    N, K = w.shape
    w_i32 = (w.to(torch.int32) & 0xF)
    K8 = K // 8
    w_groups = w_i32[:, :K8 * 8].reshape(N, K8, 8)
    packed = (
        w_groups[:, :, 0] | (w_groups[:, :, 1] << 4)
        | (w_groups[:, :, 2] << 8) | (w_groups[:, :, 3] << 12)
        | (w_groups[:, :, 4] << 16) | (w_groups[:, :, 5] << 20)
        | (w_groups[:, :, 6] << 24) | (w_groups[:, :, 7] << 28)
    )
    return packed

K = 64

# x_q: all 0 except K=1 is 1 (so only K=1 contributes)
x_q = torch.zeros(1, K, dtype=torch.int8, device="cuda")
x_q[0, 1] = 1
print(f"x_q: K=1 set to 1, rest 0")

# w_q: column n has weight at K = offset[n]
# We'll test multiple offsets to map the K-quad assignment
offsets = [0, 8, 16, 24, 32, 40, 48, 56]
N_test = len(offsets)

print(f"\nTest: each column n has weight at K={offsets}")
for n, kpos in enumerate(offsets):
    print(f"  Col {n}: w[K={kpos}] = 1, rest 0")

# Build w_q: [N_test, K] with one weight per column
w_q = torch.zeros(N_test, K, dtype=torch.int8, device="cuda")
for n, kpos in enumerate(offsets):
    w_q[n, kpos] = 1

# Pack
x_packed = pack_8_per_int32(x_q)  # [1, 8]
w_packed = pack_8_per_int32(w_q)  # [N_test, 8]

print(f"\nx_packed hex: {[hex(v.item()) for v in x_packed[0]]}")
print(f"w_packed hex per row:")
for n in range(N_test):
    print(f"  col {n}: {[hex(v.item()) for v in w_packed[n]]}")

# Pad M
x_packed = torch.cat([x_packed, torch.zeros(15, K//8, dtype=torch.int32, device="cuda")], dim=0)

# Run TC
s_x = torch.ones(16, dtype=torch.float32, device="cuda")
w_scale = torch.ones(N_test, dtype=torch.float32, device="cuda")

y_tc = peqat_cuda.fused_int4_tc_per_token_forward(
    x_packed, w_packed, w_scale, s_x, N_test, K,
)
y_tc = y_tc[:1, :N_test].contiguous()

print(f"\nTC output (should be 1 where K position matches column):")
for n in range(N_test):
    expected = 1 if x_q[0, offsets[n]].item() == 1 else 0
    actual = y_tc[0, n].item()
    mark = "✓" if expected == actual else "✗"
    print(f"  Col {n} (K={offsets[n]}): expected={expected}, got={actual} {mark}")

# Now try with x_q having K=16 set to 1
print(f"\n\n=== Second test: x_q[K=16] = 1 ===")
x_q2 = torch.zeros(1, K, dtype=torch.int8, device="cuda")
x_q2[0, 16] = 1
x_packed2 = pack_8_per_int32(x_q2)
x_packed2 = torch.cat([x_packed2, torch.zeros(15, K//8, dtype=torch.int32, device="cuda")], dim=0)
y_tc2 = peqat_cuda.fused_int4_tc_per_token_forward(
    x_packed2, w_packed, w_scale, s_x, N_test, K,
)
y_tc2 = y_tc2[:1, :N_test].contiguous()
print(f"TC output (should be 1 at the column whose K group contains K=16):")
for n in range(N_test):
    actual = y_tc2[0, n].item()
    mark = "✓" if actual != 0 else " "
    print(f"  Col {n} (K={offsets[n]}): got={actual} {mark}")

# Third test: x_q[K=32] = 1
print(f"\n\n=== Third test: x_q[K=32] = 1 ===")
x_q3 = torch.zeros(1, K, dtype=torch.int8, device="cuda")
x_q3[0, 32] = 1
x_packed3 = pack_8_per_int32(x_q3)
x_packed3 = torch.cat([x_packed3, torch.zeros(15, K//8, dtype=torch.int32, device="cuda")], dim=0)
y_tc3 = peqat_cuda.fused_int4_tc_per_token_forward(
    x_packed3, w_packed, w_scale, s_x, N_test, K,
)
y_tc3 = y_tc3[:1, :N_test].contiguous()
print(f"TC output (should be 1 at the column whose K group contains K=32):")
for n in range(N_test):
    actual = y_tc3[0, n].item()
    mark = "✓" if actual != 0 else " "
    print(f"  Col {n} (K={offsets[n]}): got={actual} {mark}")
