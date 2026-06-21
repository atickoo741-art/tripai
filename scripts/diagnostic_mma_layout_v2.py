"""Diagnostic v2: probe exact m16n8k64 fragment layouts.

Tests A, B, and D fragment layouts independently using known patterns.
Outputs a full 16x8 D matrix so we can reverse-engineer the thread->element mapping.
"""

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

// Diagnostic: D fragment layout test with column-specific B.
// Set A=1 all rows, B=col0=1 (lanes 0-3), col4=2 (lanes 16-19), else 0.
// D[row][0] = 64 (sum over 64 K of 1*1), D[row][4] = 128 (sum of 1*2).
__global__ void diag_d_layout(int32_t* d_out) {
    int lane = threadIdx.x;
    uint32_t a[4], b[2];

    a[0] = a[1] = a[2] = a[3] = 0x11111111;

    b[0] = b[1] = 0;
    if (lane < 4) { b[0] = 0x11111111; b[1] = 0x11111111; }
    if (lane >= 16 && lane < 20) { b[0] = 0x22222222; b[1] = 0x22222222; }

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

torch::Tensor diag_d() {
    auto d_out = torch::zeros({32, 4}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    diag_d_layout<<<1, 32>>>(d_out.data_ptr<int32_t>());
    return d_out;
}
"""

cpp_source = "torch::Tensor diag_d();"

print("Compiling diagnostic v2...")
mod = load_inline(
    name="mma_diag2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["diag_d"],
    extra_cuda_cflags=["-arch=sm_121"],
    verbose=False,
)

print("\n=== D layout diagnostic (A=1, col0=1, col4=2) ===")
d_out = mod.diag_d()

print("\nRaw D output (32 lanes x 4 registers):")
for lane in range(32):
    d = d_out[lane].tolist()
    print(f"  L{lane:02d}: [{d[0]:6d}, {d[1]:6d}, {d[2]:6d}, {d[3]:6d}]")

# Build 16x8 D matrix under BOTH mapping hypotheses
print("\n--- Hypothesis 1: column PAIR mapping (INT8 pattern) ---")
D_pair = [[0 for _ in range(8)] for _ in range(16)]
for lane in range(32):
    d = d_out[lane].tolist()
    row_a = lane // 4
    col_base = (lane % 4) * 2
    D_pair[row_a][col_base] = d[0].item()
    D_pair[row_a][col_base + 1] = d[1].item()
    D_pair[row_a + 8][col_base] = d[2].item()
    D_pair[row_a + 8][col_base + 1] = d[3].item()

for r in range(16):
    print(f"  row {r:2d}: " + " ".join(f"{D_pair[r][c]:6d}" for c in range(8)))

errors_pair = 0
for r in range(16):
    for c in [0, 4]:
        expected = 64 if c == 0 else 128
        if D_pair[r][c] != expected:
            errors_pair += 1
print(f"  Mismatches: {errors_pair}/32 (expect D[*][0]=64, D[*][4]=128)")

print("\n--- Hypothesis 2: single-column mapping ---")
D_single = [[0 for _ in range(8)] for _ in range(16)]
for lane in range(32):
    d = d_out[lane].tolist()
    row_a = lane // 4
    col0 = lane % 4
    D_single[row_a][col0] = d[0].item()
    D_single[row_a][col0 + 4] = d[1].item()
    D_single[row_a + 8][col0] = d[2].item()
    D_single[row_a + 8][col0 + 4] = d[3].item()

for r in range(16):
    print(f"  row {r:2d}: " + " ".join(f"{D_single[r][c]:6d}" for c in range(8)))

errors_single = 0
for r in range(16):
    for c in [0, 4]:
        expected = 64 if c == 0 else 128
        if D_single[r][c] != expected:
            errors_single += 1
print(f"  Mismatches: {errors_single}/32 (expect D[*][0]=64, D[*][4]=128)")

print(f"\nConclusion: pair mapping errors={errors_pair}, single-col errors={errors_single}")
if errors_pair == 0 and errors_single > 0:
    print("  => Use PAIR mapping in kernel")
elif errors_single == 0 and errors_pair > 0:
    print("  => Use SINGLE-COL mapping in kernel")
elif errors_pair == 0 and errors_single == 0:
    print("  => BOTH mappings match (ambiguous test)")
else:
    print("  => NEITHER mapping matches! Wrong assumptions about row/col assignment.")
