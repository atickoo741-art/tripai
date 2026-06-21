"""Diagnostic v2: probe exact m16n8k64 fragment layouts + accumulation."""

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

// Test 1: D fragment layout with column-specific B
// B col0=1 (lanes 0-3), col1=3 (lanes 4-7), col4=2 (lanes 16-19), col5=4 (lanes 20-23)
// D[row][0]=64, D[row][1]=192 (64*3), D[row][4]=128, D[row][5]=256 (64*4)
__global__ void diag_d_layout(int32_t* d_out) {
    int lane = threadIdx.x;
    uint32_t a[4], b[2];
    a[0] = a[1] = a[2] = a[3] = 0x11111111;

    b[0] = b[1] = 0;
    if (lane >= 0 && lane < 4)  { b[0] = 0x11111111; b[1] = 0x11111111; }
    if (lane >= 4 && lane < 8)  { b[0] = 0x33333333; b[1] = 0x33333333; }
    if (lane >= 16 && lane < 20) { b[0] = 0x22222222; b[1] = 0x22222222; }
    if (lane >= 20 && lane < 24) { b[0] = 0x44444444; b[1] = 0x44444444; }

    int32_t d[4] = {0, 0, 0, 0};
    asm("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );

    d_out[lane * 4 + 0] = d[0];
    d_out[lane * 4 + 1] = d[1];
    d_out[lane * 4 + 2] = d[2];
    d_out[lane * 4 + 3] = d[3];
}

// Test 2: A fragment layout
// Each thread puts its lane# in A's s4 positions.
// B col0=1 to get non-zero output only in column 0.
// D[row][0] = sum over K of (A_value[row][k] * 1) for each row.
__global__ void diag_a_layout(int32_t* d_out) {
    int lane = threadIdx.x;
    uint32_t a[4];
    uint32_t val = 0;
    for (int i = 0; i < 8; i++) val |= ((lane & 0xF) << (i * 4));
    a[0] = a[1] = a[2] = a[3] = val;

    uint32_t b[2] = {0, 0};
    if (lane < 4) { b[0] = 0x11111111; b[1] = 0x11111111; }

    int32_t d[4] = {0, 0, 0, 0};
    asm("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );

    d_out[lane * 4 + 0] = d[0];
    d_out[lane * 4 + 1] = d[1];
    d_out[lane * 4 + 2] = d[2];
    d_out[lane * 4 + 3] = d[3];
}

// Test 3: Accumulation test
// Run MMA twice with same A=1, B=1. Result should be double.
__global__ void diag_accum_layout(int32_t* d_out) {
    int lane = threadIdx.x;
    uint32_t a[4], b[2];
    a[0] = a[1] = a[2] = a[3] = 0x11111111;
    b[0] = b[1] = 0;
    if (lane < 4) { b[0] = 0x11111111; b[1] = 0x11111111; }

    int32_t d[4] = {0, 0, 0, 0};

    // First MMA
    asm("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );

    // Second MMA with same values (should accumulate)
    asm("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );

    d_out[lane * 4 + 0] = d[0];
    d_out[lane * 4 + 1] = d[1];
    d_out[lane * 4 + 2] = d[2];
    d_out[lane * 4 + 3] = d[3];
}

torch::Tensor diag_d() {
    auto d_out = torch::zeros({32, 4}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    diag_d_layout<<<1, 32>>>(d_out.data_ptr<int32_t>());
    return d_out;
}
torch::Tensor diag_a() {
    auto d_out = torch::zeros({32, 4}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    diag_a_layout<<<1, 32>>>(d_out.data_ptr<int32_t>());
    return d_out;
}
torch::Tensor diag_accum() {
    auto d_out = torch::zeros({32, 4}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    diag_accum_layout<<<1, 32>>>(d_out.data_ptr<int32_t>());
    return d_out;
}
"""

cpp_source = "torch::Tensor diag_d(); torch::Tensor diag_a(); torch::Tensor diag_accum();"

print("Compiling diagnostic v2...")
mod = load_inline(
    name="mma_diag2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["diag_d", "diag_a", "diag_accum"],
    extra_cuda_cflags=["-arch=sm_121"],
    verbose=False,
)

# === Test 1: D layout ===
print("\n=== Test 1: D layout (col0=1, col1=3, col4=2, col5=4) ===")
d_out = mod.diag_d()
print("Raw D (lanes x registers):")
for lane in range(32):
    d = d_out[lane].tolist()
    print(f"  L{lane:02d}: [{d[0]:6d}, {d[1]:6d}, {d[2]:6d}, {d[3]:6d}]")

# Reconstruct D matrix under PAIR mapping
print("\n  Reconstructed D (pair mapping):")
D = [[0 for _ in range(8)] for _ in range(16)]
for lane in range(32):
    d = d_out[lane].tolist()
    r = lane // 4
    c = (lane % 4) * 2
    D[r][c] = d[0]
    D[r][c + 1] = d[1]
    D[r + 8][c] = d[2]
    D[r + 8][c + 1] = d[3]

for r in range(16):
    print(f"    row {r:2d}: " + " ".join(f"{D[r][c]:6d}" for c in range(8)))

# Check expected:
# D[*][0]=64, D[*][1]=192, D[*][4]=128, D[*][5]=256
ok = True
for r in range(16):
    for c, exp in [(0, 64), (1, 192), (4, 128), (5, 256)]:
        if D[r][c] != exp:
            print(f"    MISMATCH D[{r}][{c}]={D[r][c]} != {exp}")
            ok = False
if ok:
    print("  => PAIR mapping VERIFIED!")

# === Test 2: A layout ===
print("\n=== Test 2: A layout (thread->row mapping) ===")
d_out = mod.diag_a()
print("Raw D (only lanes with non-zero D[*,0] shown):")
for lane in range(32):
    d = d_out[lane].tolist()
    if d[0] != 0 or d[2] != 0:
        # This lane contributes to some D[*,0] position
        r_a = lane // 4  # which row group (0-based, for first 8 rows)
        # d[0] = D[r][0], d[2] = D[r+8][0]
        # With A encoding lane&0xF, the value in D[*,0] = sum over K of (lane_val * 1)
        # = 64 * (lane & 0xF)
        expected_val = 64 * (lane & 0xF)
        print(f"  L{lane:02d}: d=[{d[0]:6d},{d[1]:6d},{d[2]:6d},{d[3]:6d}]  (expect {expected_val})")

# === Test 3: Accumulation ===
print("\n=== Test 3: Accumulation (2x MMA with same values) ===")
d_out = mod.diag_accum()
print("Raw D (should be 2x single-MMA values):")
for lane in range(32):
    d = d_out[lane].tolist()
    if d[0] != 0 or d[2] != 0:
        print(f"  L{lane:02d}: [{d[0]:6d}, {d[1]:6d}, {d[2]:6d}, {d[3]:6d}]  (expect 128 for col0)")
        if d[0] == 128:
            print(f"    => Accumulation VERIFIED!")
