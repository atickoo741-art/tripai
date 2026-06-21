"""Debug MMA: print packed data, check register values, step through K loop."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import peqat_cuda

torch.manual_seed(42)

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

# Tiny test: K=64, M=1, N=4 (just 4 output cols to keep it simple)
K = 64
M = 1
N = 4

# Fixed known values for easier debugging
x_q = torch.tensor([[1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1, 0,
                     0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
                     1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1, 0,
                     0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1]],
    dtype=torch.int8, device="cuda")

w_q = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    dtype=torch.int8, device="cuda")

print(f"x_q shape: {x_q.shape}, first 16 values: {x_q[0, :16].tolist()}")
print(f"w_q shape: {w_q.shape}")

# Expected dot products
print(f"\nExpected dot products (ref = x @ w.T):")
for n in range(N):
    dot = (x_q[0].float() * w_q[n].float()).sum().item()
    print(f"  y[0,{n}] = sum(x * w[{n}]) = {dot}")

# Pack for TC
x_packed = pack_8_per_int32(x_q)  # [1, 8]
w_packed = pack_8_per_int32(w_q)  # [4, 8]

print(f"\nx_packed shape: {x_packed.shape}")
print(f"x_packed hex: {[hex(v.item()) for v in x_packed[0]]}")

print(f"\nw_packed hex:")
for n in range(N):
    print(f"  row {n}: {[hex(v.item()) for v in w_packed[n]]}")

# Verify packing: unpack first word of x_packed
x0 = x_packed[0, 0].item()
print(f"\nVerifying x_packed[0,0] = 0x{x0:08X}:")
for i in range(8):
    nibble = (x0 >> (i*4)) & 0xF
    s4 = nibble if nibble < 8 else nibble - 16
    print(f"  nibble {i} (K{i}: {nibble:1X} -> s4={s4:3d})")

# Pad M
x_packed = torch.cat([x_packed, torch.zeros(15, K//8, dtype=torch.int32, device="cuda")], dim=0)

# Run TC kernel
s_x = torch.ones(16, dtype=torch.float32, device="cuda")
w_scale = torch.ones(N, dtype=torch.float32, device="cuda")

y_tc = peqat_cuda.fused_int4_tc_per_token_forward(
    x_packed, w_packed, w_scale, s_x, N, K,
)
y_tc = y_tc[:M, :N].contiguous()

print(f"\nTC output:")
for n in range(N):
    print(f"  y[0,{n}] = {y_tc[0, n].item()}")
