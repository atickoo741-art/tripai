"""Diagnostic: probe mma.sync fragment layout with known values.

Tests what (row, col) each thread's D registers map to by using
A=all-ones, B=select-one-column, then reading back D.
"""

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

// Set A to all 1s in the s4 packed format.
// Each int32 packs 8 s4 values = 0x11111111 (8 copies of s4=1).
// B sets a single column to a known value, then we read back D
// to map which lane/register = which output element.

__global__ void diagnostic_a_all_ones(
    int32_t* d_out  // [32, 4] output: each lane stores its 4 D registers
) {
    int lane = threadIdx.x;

    // A fragment: all s4 = 1 → packed = 0x11111111 per uint32
    uint32_t a[4] = {0x11111111, 0x11111111, 0x11111111, 0x11111111};

    // B fragment: all s4 = 1 in column 0 (lanes 0-3), 0 elsewhere
    // Each uint32 packs 8 s4 values
    uint32_t b[2] = {0, 0};

    // Only lanes 0-3 (column 0 of B) get non-zero values
    // These correspond to different K-quarters:
    // lane 0: column 0, K 0-15
    // lane 1: column 0, K 16-31
    // lane 2: column 0, K 32-47
    // lane 3: column 0, K 48-63
    if (lane < 4) {
        b[0] = 0x11111111;  // first 8 K positions = s4=1
        b[1] = 0x11111111;  // next 8 K positions = s4=1
    }

    int32_t d[4] = {0, 0, 0, 0};

    asm("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );

    // If A=1, B=1 for column 0 (4 lanes × 16 K each = 64 K total):
    // D[row][0] should be 64 for all rows.
    // D[row][col>0] should be 0.

    d_out[lane * 4 + 0] = d[0];
    d_out[lane * 4 + 1] = d[1];
    d_out[lane * 4 + 2] = d[2];
    d_out[lane * 4 + 3] = d[3];
}

torch::Tensor diagnostic_all_ones() {
    auto d_out = torch::zeros({32, 4}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    diagnostic_a_all_ones<<<1, 32>>>(d_out.data_ptr<int32_t>());
    return d_out;
}
"""

cpp_source = "torch::Tensor diagnostic_all_ones();"

print("Compiling diagnostic kernel...")
mod = load_inline(
    name="mma_diag",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["diagnostic_all_ones"],
    extra_cuda_cflags=["-arch=sm_121"],
    verbose=False,
)

print("Running diagnostic (A=1, B=col0=1, rest=0)...")
d_out = mod.diagnostic_all_ones()
print(f"d_out shape: {d_out.shape}")
print(f"d_out dtype: {d_out.dtype}")
print()

# Try to determine the layout
for lane in range(32):
    d0, d1, d2, d3 = d_out[lane].tolist()
    if d0 != 0 or d1 != 0 or d2 != 0 or d3 != 0:
        print(f"  Lane {lane:2d}: d=[{d0:3d}, {d1:3d}, {d2:3d}, {d3:3d}]")

print()
# Create a 16x8 output grid showing which lane mapped to each (row,col)
d_map = [["----" for _ in range(8)] for _ in range(16)]
for lane in range(32):
    d0, d1, d2, d3 = d_out[lane].tolist()
    for reg_idx, val in enumerate([d0, d1, d2, d3]):
        if abs(val) > 0:
            d_map[0][0] = f"L{lane:02d}r{reg_idx}"

# Instead, print the raw output as a matrix for interpretation
print("Raw D output (32 lanes × 4 registers):")
for lane in range(32):
    vals = d_out[lane].tolist()
    print(f"  L{lane:02d}: {vals}")

# Summarize which rows got non-zero output
nonzero_rows = {}
for lane in range(32):
    vals = d_out[lane].tolist()
    for ri, v in enumerate(vals):
        if v != 0:
            # This lane, register ri has value v for some (row, col) output
            # row = lane % 16 (hypothesis), col = 0 (from B col 0)
            row_hyp = lane % 16
            col_group = lane // 16
            col_hyp_0 = col_group * 4 + ri
            # Also try: col = (lane // 16) * ... 
            key = (row_hyp, col_hyp_0)
            if key not in nonzero_rows:
                nonzero_rows[key] = []
            nonzero_rows[key].append(v)

print(f"\nNon-zero entries by (row_hyp=lane%16, col_hyp=col_group*4+ri):")
for (r, c), vals in sorted(nonzero_rows.items()):
    print(f"  D[{r},{c}] = {vals} {'✓' if abs(vals[0]) >= 60 else '?'} (expected ~64 for col 0, 0 for cols>0)")
