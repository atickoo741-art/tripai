"""Map exact nibble positions: place 1 at specific K offset, see which column sees it."""

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

# Test: for each K position 0..63, set x_q[K] = 1, and check which 
# column gets value=1 in the output. 8 columns, each with weight at a 
# different word boundary.
print("Mapping A-fragment K positions to B-fragment columns...")
print("(8 columns, each with weight at K = word_idx * 8)")
print()

word_offsets = list(range(0, 64, 8))  # [0, 8, 16, 24, 32, 40, 48, 56]
N_test = 8

# Weight matrix: each column c has weight=1 at K = word_offsets[c]
w_q = torch.zeros(N_test, K, dtype=torch.int8, device="cuda")
for c, kpos in enumerate(word_offsets):
    w_q[c, kpos] = 1
w_packed = pack_8_per_int32(w_q)

s_x = torch.ones(16, dtype=torch.float32, device="cuda")
w_scale = torch.ones(N_test, dtype=torch.float32, device="cuda")

print("K pos -> column that gets value 1 (expect column = K // 8):")
print("  K  | cols with value 1 (expect one)")
for kpos in range(64):
    x_q = torch.zeros(1, K, dtype=torch.int8, device="cuda")
    x_q[0, kpos] = 1
    x_packed = pack_8_per_int32(x_q)
    x_packed = torch.cat([x_packed, torch.zeros(15, K//8, dtype=torch.int32, device="cuda")], dim=0)
    
    y_tc = peqat_cuda.fused_int4_tc_per_token_forward(
        x_packed, w_packed, w_scale, s_x, N_test, K,
    )
    y_tc = y_tc[:1, :N_test].contiguous()
    
    active_cols = [c for c in range(N_test) if abs(y_tc[0, c].item()) > 0.5]
    expected = kpos // 8
    if active_cols != [expected]:
        print(f"  K={kpos:2d}: expected col {expected}, got cols {active_cols}")
    else:
        pass  # skip correct matches for brevity

print()
print("All mismatches printed above. Correct matches suppressed.")
