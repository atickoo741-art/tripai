# Fused INT4 GEMM Kernel — Design Document

## 1. Problem

Current `linear_forward` does two separate GPU operations:
1. `unpack_dequant_kernel`: reads packed INT4 → writes full FP32 weight matrix [N, K] (~8GB for Llama-8B)
2. `torch::matmul(x_dq, w_dq.t())`: cuBLAS GEMM on the materialized FP32 weights

This wastes 4× memory bandwidth on weights (FP32 vs INT4) and allocates a huge intermediate buffer.

## 2. Algorithm

We compute the same thing as before, but fused:

```
# x_dq: [M, K] FP32 — preprocessed, dequantized activations
# packed: [P] uint8 — low_high nibble order
# w_scale: [N] FP16 — per-row scales

For each threadblock output tile [Mt, Nt]:
  Load A_tile = x_dq[mt : mt+Mt, :] in K-chunks of size Kt
  Load B_tile = packed_w[nt : nt+Nt, :] in same K-chunks, UNPACK + DEQUANT on the fly
  Accumulate partial C_tile in registers
  Write C_tile[mt:mt+Mt, nt:nt+Nt] to global memory
```

## 3. Data Layout (exact)

**weight_int4.bin:** uint8 row-major, low_high nibble order
- Row `r`, columns `c` and `c+1` packed into byte `(r*K + c) / 2`
- Byte bits: `[val_{c+1}_nibble (bits 7:4) | val_c_nibble (bits 3:0)]`
- Signed conversion: `q = nibble > 7 ? nibble - 16 : nibble` → range [-8, 7]
- Padded to even K, then 16-byte aligned

**weight_scale.fp16:** flat FP16, one per row, length N

**Shape:** W_dq ∈ ℝ^[N, K], activation x_dq ∈ ℝ^[M, K], output C ∈ ℝ^[M, N]

## 4. Kernel Design

### 4.1 Launch Configuration

```cuda
fused_int4_gemm_kernel<<<grid, block, smem_size, stream>>>(
    x_dq,       // [M, K] FP32
    packed_w,   // [P] uint8
    w_scale,    // [N] FP16
    output,     // [M, N] FP32
    M, N, K
);
```

- **Block:** 256 threads = 8 warps × 32
- **Grid:** (M / BM, N / BN) — ceil division

### 4.2 Tile Sizes

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| BM (M tile) | 128 | Good for prefill (M=128+, large M) and decode (M=1, 1×128 slice) |
| BN (N tile) | 128 | Matches 128-wide output, good for weight fetch |
| BK (K tile) | 32 | Balances shared memory pressure vs. K-iteration overhead |

Alternative tile config for decode-heavy workloads: BM=8, BN=128, BK=64 — to be benchmarked.

### 4.3 Shared Memory Layout

```cuda
__shared__ float s_A[BM * BK];          // Activation tile [BM, BK] row-major
__shared__ float s_B[BK * BN];          // Weight tile [BK, BN] row-major (unpacked on the fly)
```

SMEM cost: `(BM*BK + BK*BN) * 4 bytes = (128*32 + 32*128) * 4 = 32,768 bytes` — well under the 48KB limit.

### 4.4 Weight Tile Load + Unpack

Each block loads a BK×BN tile of weights. This tile spans:
- K dimension: `k_start` to `k_start + BK`
- N dimension: `n_start` to `n_start + BN`

The physical storage is packed row-major: `packed_w[row_idx * ((K+1)/2) + col_pair_idx]`.

For the tile `[K_offset : K_offset+BK, N_offset : N_offset+BN]`, the threads cooperatively load the packed bytes and unpack.

**Thread mapping:** 256 threads load `BK * BN / 2 = 32*128/2 = 2048` packed bytes
- Each thread loads `2048 / 256 = 8` packed bytes
- For each byte: unpack 2 nibbles → convert to signed → multiply by `w_scale[row]` → store FP32 to `s_B[k*BN + n]`

**Pseudo-code per thread:**

```cuda
int tid = threadIdx.x;
int num_packed = (BK * BN) / 2;
int items_per_thread = num_packed / blockDim.x;
int remaining = num_packed % blockDim.x;
int start = tid * items_per_thread + min(tid, remaining);
int count = items_per_thread + (tid < remaining ? 1 : 0);

for (int i = 0; i < count; i++) {
    int flat_idx = start + i;                   // flat index into packed tile
    int k_in_tile = (flat_idx * 2) / BN;        // which K row within tile
    int n_base = (flat_idx * 2) % BN;           // which N column (first of pair)

    int global_k = k_start + k_in_tile;
    int global_n1 = n_start + n_base;
    int global_n2 = n_start + n_base + 1;

    // Packed byte index in global buffer
    int packed_byte_idx = global_n1 * ((K + 1) / 2) + global_k / 2;
    uint8_t byte = packed_w[packed_byte_idx];
    // BUT: this assumes global_n1 is even (low nibble). Need to check.

    ...
}
```

Wait — the packing is row-major: `byte = packed_w[row * ((K+1)/2) + col/2]` where col/2 accounts for pairs in the K dimension. So a given `(global_n, global_k)` maps to:

```
byte_idx = global_n * ((K + 1) / 2) + global_k / 2
nibble = (global_k % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF)
```

This means: within a row, consecutive K values are paired. The tile in K dimension maps to contiguous bytes within rows.

**Corrected thread mapping for weight tile load:**

Each thread loads a contiguous chunk of the BK×BN packed tile. The tile in memory is NOT contiguous as a 2D block because row strides differ.

Better approach: each thread handles a set of (k_in_tile, n_in_tile) coordinates within the tile:

```cuda
for each assigned (k_in_tile, n_in_tile):
    global_n = n_start + n_in_tile
    global_k = k_start + k_in_tile
    byte_idx = global_n * ((K + 1) / 2) + global_k / 2
    byte = packed_w[byte_idx]
    nibble = (global_k % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF)
    q = nibble > 7 ? nibble - 16 : nibble
    s_B[k_in_tile * BN + n_in_tile] = (float)q * __half2float(w_scale[global_n])
```

Each thread handles `(BK * BN) / 256` elements. For BK=32, BN=128: 16 elements per thread.

### 4.5 Activation Tile Load

x_dq is [M, K] in FP32 row-major. The tile [m_start : m_start+BM, k_start : k_start+BK] is contiguous in each row but strided across rows (stride = K). Each thread loads a chunk:

```cuda
for each assigned (m_in_tile, k_in_tile):
    s_A[m_in_tile * BK + k_in_tile] = x_dq[(m_start + m_in_tile) * K + (k_start + k_in_tile)]
```

### 4.6 Compute Phase

After loading A and B tiles into shared memory and syncing, each thread computes partial products for its assigned output elements.

**Approach A: Manual FP32 loop (portable, baseline)**

Each thread computes a `THR_M × THR_N` output fragment:

```cuda
float accum[THR_M][THR_N] = {0};

for (int kk = 0; kk < BK; kk++) {
    for (int tm = 0; tm < THR_M; tm++) {
        for (int tn = 0; tn < THR_N; tn++) {
            accum[tm][tn] += s_A[my_m[tm] * BK + kk] * s_B[kk * BN + my_n[tn]];
        }
    }
}
```

Where `my_m[tm]` and `my_n[tn]` are the thread's assigned output indices within the tile.

For 256 threads and BM×BN = 128×128 = 16384 outputs, each thread gets `16384 / 256 = 64` elements. With THR_M=8, THR_N=8: 64 elements, 8×8×32 = 2048 FMAs per thread per K tile.

**Approach B: WMMA with FP16 tensor cores (Blackwell)**

```cuda
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;

// Per K-tile iteration:
nvcuda::wmma::load_matrix_sync(a_frag, s_A + warp_m * 16 * BK + kk, BK);  // BK stride
nvcuda::wmma::load_matrix_sync(b_frag, s_B + kk * BN + warp_n * 16, BN);  // BN stride
nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

Approach A first for correctness; Approach B as follow-up for tensor core speed.

### 4.7 Global Memory Write

After iterating all K-tiles, each thread writes its accumulated fragment to `output[m_start + my_m][n_start + my_n]`.

## 5. Handling Small M (Decode, M=1)

For decode with M=1, the BM=128 tile wastes threads. Add a specialization:
- **Kernel variant 2:** `fused_int4_gemm_kernel_smallm` with BM=8 (or dynamic)
- When M < BM/2, launch with BM=M and adjust block/grid
- Grid: `(ceil(M/BM), ceil(N/BN))` — no extra blocks

For M=1, BM=8: each threadblock covers 1×128 output, 8 warps compute in parallel over K.

## 6. Binding Interface

```cuda
// New binding in bindings.cpp
torch::Tensor fused_linear_forward(
    torch::Tensor x_dq,       // [M, K] float32
    torch::Tensor packed_w,   // [P] uint8
    torch::Tensor w_scale,    // [N] float16
    int64_t rows,             // N
    int64_t cols              // K
);
```

Replaces the old `linear_forward` in `CUDAQuantizedLinear.forward()`. Same Python API, no loader changes.

## 7. Expected Speedup

| Phase | Weight Read | Matmul | Total (est) |
|-------|------------|--------|-------------|
| Current (Phase B) | 256GB/s × 8GB = 31ms | cuBLAS ~40ms | ~71ms/tok decode |
| Fused (Phase F) | 256GB/s × 2GB = 8ms | tiled ~45ms | ~53ms/tok decode |
| Target | — | — | ~40-50ms/tok |

Primary savings: eliminating the FP32 weight materialization (8GB write + 8GB read → 2GB INT4 read). Secondary: better cache reuse from tiling.

## 8. Key Optimizations Implemented (v2)

### 8.1 Byte-pair loading (halves global weight reads)

Each thread loads 8 packed bytes instead of 16 individual values. Each byte is loaded once from global memory, then both nibbles are unpacked and stored as two FP32 values in `s_B`. This halves the global memory traffic for weights.

### 8.2 w_scale cached in shared memory

`s_scale[BN]` in shared memory loaded once per K-tile iteration (512 bytes). Used during B-tile unpack instead of per-element `w_scale[g_row]` loads.

### 8.3 Bank-conflict mitigation

Both `s_A` and `s_B` use stride `BK_STRIDE = BK + 1` (=33) instead of `BK` (=32). This changes the shared-memory bank mapping from `(row*BK + col) % 32 = col % 32` (all threads at same col → same bank) to `((row*33 + col) % 32) = (row + col) % 32`, distributing accesses across banks.

### 8.4 Separate decode kernel for M ≤ 8

`fused_int4_gemm_smallm_kernel` with tile BM=8, BN=128, BK=32. Each thread computes 1×4 output elements. No wasted threads: all 256 threads produce valid output for the full 8×128 tile.

### 8.5 Float4 vectorized A-tile load

Activation tile loaded via `float4` when all 4 elements are within bounds, reducing instruction count for the A-side global memory reads.

## 9. Implementation Plan

1. **`csrc/peqat_fused_gemm.cu`** — prefill + decode kernels with all optimizations
2. **`csrc/bindings.cpp`** — dispatch to decode kernel when M ≤ 8
3. **`csrc/setup.py`** — add `peqat_fused_gemm.cu` to sources
4. **`peqat_deploy/runtime/loader.py`** — update `linear_forward` call in `CUDAQuantizedLinear`
5. **`scripts/bench_fused_kernel.py`** — microbenchmark comparing old vs fused
6. **WMMA variant** — follow-up when baseline correct
