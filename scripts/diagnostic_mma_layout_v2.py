"""Diagnostic v2: probe exact m16n8k64 fragment layouts."""

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

// Probe A mapping: set B col0=1, and set A encoding to (lane_index << 4) | lane_index
// in each s4 position. Then compute D. From D[*,0], we can deduce which lanes'
// A values contribute to each row.
__global__ void diag_a_probe(int32_t* d_out) {
    int lane = threadIdx.x;
    // A: each nibble = (lane & 0xFF) — unique per lane
    uint32_t v = 0;
    for (int i = 0; i < 8; i++) v |= ((lane & 0xFF) << (i * 4));
    uint32_t a[4] = {v, v, v, v};

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

// Probe B mapping: B col0 with unique per-lane values, A=all-ones.
// D[*,0] should show which lane's B values affect which output elements.
// But since B only affects per-column, and we only set col0, all D[*][0]
// should be affected equally by all 4 B-quad lanes. This tests B K-range mapping.
__global__ void diag_b_probe(int32_t* d_out) {
    int lane = threadIdx.x;
    uint32_t a[4] = {0x11111111, 0x11111111, 0x11111111, 0x11111111};

    // B: each lane with lane<4 puts unique value per quad of col0
    // Lane 0: value=1 (K0-15), Lane 1: value=2 (K16-31)
    // Lane 2: value=3 (K32-47), Lane 3: value=4 (K48-63)
    uint32_t b[2] = {0, 0};
    int vals[] = {1, 2, 3, 4};
    if (lane < 4) {
        int v = vals[lane];
        uint32_t bv = 0;
        for (int i = 0; i < 8; i++) bv |= ((v & 0xF) << (i * 4));
        b[0] = b[1] = bv;
    }

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

// Probe: single unique column with single unique quad.
// Set only lane L to B=col0, others B=0. A=all-ones.
// Deduce which D row gets contribution from each lane's A.
__global__ void diag_row_probe(int32_t* d_out, int active_lane) {
    int lane = threadIdx.x;
    uint32_t a[4] = {0x11111111, 0x11111111, 0x11111111, 0x11111111};

    uint32_t b[2] = {0, 0};
    uint32_t bv = 0;
    for (int i = 0; i < 8; i++) bv |= (1 << (i * 4));
    b[0] = b[1] = bv;
    // Only active_lane has B=col0=1

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

// Run row probe for all 4 B-quad lanes to map which threads' A contribute to which D rows
__global__ void diag_row_probe_all(int32_t* d_out) {
    int lane = threadIdx.x;
    uint32_t a[4] = {0x11111111, 0x11111111, 0x11111111, 0x11111111};

    // Each of lanes 0-3 puts B=1 for col0 at their specific quad
    // Lane i (0-3): quad i has value i+1, all other quads 0
    uint32_t b[2] = {0, 0};
    if (lane < 4) {
        uint32_t bv = 0;
        for (int i = 0; i < 8; i++) bv |= ((lane + 1) << (i * 4));
        b[0] = b[1] = bv;
    }

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

torch::Tensor diag_a() {
    auto d_out = torch::zeros({32, 4}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    diag_a_probe<<<1, 32>>>(d_out.data_ptr<int32_t>());
    return d_out;
}
torch::Tensor diag_b() {
    auto d_out = torch::zeros({32, 4}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    diag_b_probe<<<1, 32>>>(d_out.data_ptr<int32_t>());
    return d_out;
}
torch::Tensor diag_row() {
    auto d_out = torch::zeros({32, 4}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    diag_row_probe_all<<<1, 32>>>(d_out.data_ptr<int32_t>());
    return d_out;
}
"""

cpp_source = "torch::Tensor diag_a(); torch::Tensor diag_b(); torch::Tensor diag_row();"

print("Compiling...")
mod = load_inline(
    name="mma_diag3",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["diag_a", "diag_b", "diag_row"],
    extra_cuda_cflags=["-arch=sm_121"],
    verbose=False,
)

# === A probe: decode which lanes' A contributes to each row ===
print("\n=== A probe: lane-unique A, B=col0=1 ===")
d_out = mod.diag_a().cpu()

# Build D matrix
D = [[0]*8 for _ in range(16)]
for lane in range(32):
    d = d_out[lane].tolist()
    r = lane // 4
    c = (lane % 4) * 2
    D[r][c] = d[0]
    D[r][c+1] = d[1]
    D[r+8][c] = d[2]
    D[r+8][c+1] = d[3]

print("  D[*][0] values (should reveal A mapping):")
for r in range(16):
    print(f"    row {r:2d}: D[{r}][0]={D[r][0]:6d}")

# D[r][0] = sum_k A[r][k] = 2 * (sum of 32 A values from each half-thread)
# With A encoding = (lane & 0xFF) in all 4 registers = 32 identical s4 values per half-thread
# So D[r][0] = 32 * s4_val_thread_a + 32 * s4_val_thread_b = 32 * (a + b)
# where s4_val_thread_X = (lane & 0xFF) but truncated to s4 (4-bit signed)

# Actually wait, lane & 0xFF can be > 15, which as s4 would wrap.
# Let me use just lane & 0xF in the earlier test. But in THIS test,
# I used lane & 0xFF. Let me reinterpret:
# lane & 0xFF returns values 0-255. But these are packed into s4 (4-bit signed).
# So the actual MMA value = (lane & 0xF) if lane&0xF < 8, else (lane&0xF - 16)
# = the s4 interpretation of the lower 4 bits.

# Wait, I used (lane & 0xFF) not (lane & 0xF). Masking with 0xFF gives
# the full byte (0-255). But only the lower 4 bits matter for s4.
# So effectively it's still (lane & 0xF).

# Let me just print and analyze
print("\n  D[*][4] values (should be half of D[*][0] since col4 B=0):")
for r in range(16):
    print(f"    row {r:2d}: D[{r}][4]={D[r][4]:6d}")

# === B probe ===
print("\n=== B probe: A=all-1, B col0 with quads 0-3 having values 1-4 ===")
d_out = mod.diag_b().cpu()
D = [[0]*8 for _ in range(16)]
for lane in range(32):
    d = d_out[lane].tolist()
    r = lane // 4
    c = (lane % 4) * 2
    D[r][c] = d[0]
    D[r][c+1] = d[1]
    D[r+8][c] = d[2]
    D[r+8][c+1] = d[3]

# With A=all-1 and B=val per quad: D[*,0] = sum over quads of 16 * 1 * quad_val
# = 16*1 + 16*2 + 16*3 + 16*4 = 16+32+48+64 = 160
expected = 16 + 32 + 48 + 64
print(f"  Expected D[*][0] = {expected} (sum of 16*{1}+16*{2}+16*{3}+16*{4})")
for r in range(16):
    print(f"    row {r:2d}: D[{r}][0]={D[r][0]:6d}" + (" OK" if D[r][0] == expected else " MISMATCH"))

# If mismatch, deduce which quads contributed:
# D[*][0] = 16*1*(b_active?) + 16*2*(b_active?) + ...
# We can back-calculate which K quads had B=1

# === Row probe ===
print("\n=== Row probe: A=all-1, each B quad has unique value ===")
d_out = mod.diag_row().cpu()
D = [[0]*8 for _ in range(16)]
for lane in range(32):
    d = d_out[lane].tolist()
    r = lane // 4
    c = (lane % 4) * 2
    D[r][c] = d[0]
    D[r][c+1] = d[1]
    D[r+8][c] = d[2]
    D[r+8][c+1] = d[3]

print("\n  Full D matrix:")
for r in range(16):
    print(f"    row {r:2d}: " + " ".join(f"{D[r][c]:6d}" for c in range(8)))

# With A=all-1 and B quad having unique value:
# Quad 0 (lane 0): B=1 → contribution  16*1 = 16
# Quad 1 (lane 1): B=2 → contribution  16*2 = 32
# Quad 2 (lane 2): B=3 → contribution  16*3 = 48
# Quad 3 (lane 3): B=4 → contribution  16*4 = 64
# D[*][0] should be 16+32+48+64=160 if all quads active
print(f"  Expected D[*][0] = 160 (=16+32+48+64) if all 4 quads active")
