#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 128
#define BN 128
#define BK 64

using namespace nvcuda::wmma;

// Fused GEMM kernel using wmma FP16 tensor cores.
// B tile is stored as K×N (s_b[BK][BN]) so wmma matrix_b row_major
// reads K-dim along rows, matching A's K-dim alignment.
__global__ void fused_gemm_wmma_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const half* __restrict__ w_scale,
    half* __restrict__ output,
    int M, int N, int K
) {
    __shared__ half s_a[BM][BK];
    __shared__ half s_b[BK][BN];
    __shared__ half s_ws[BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int m_start = bx * BM;
    int n_start = by * BN;
    int m_end = min(m_start + BM, M);
    int n_end = min(n_start + BN, N);
    int bm_eff = m_end - m_start;
    int bn_eff = n_end - n_start;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int row_group = warp_id;
    int row_base = row_group * WMMA_M;

    int K_packed = (K + 1) / 2;

    if (tid < BN) {
        s_ws[tid] = (n_start + tid < N) ? __ldg(&w_scale[n_start + tid]) : __float2half(0.0f);
    }
    __syncthreads();

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        fill_fragment(acc[i], __float2half(0.0f));
    }

    for (int k_start = 0; k_start < K; k_start += BK) {
        int bk_eff = min(K - k_start, BK);

        // Load A tile: BM×BK, each thread loads 32 half values
        int a_elems = (BM * BK) / 256;
        for (int i = 0; i < a_elems; i++) {
            int idx = tid * a_elems + i;
            int row = idx / BK;
            int col = idx % BK;
            if (row < bm_eff && col < bk_eff) {
                s_a[row][col] = __ldg(&x[(m_start + row) * K + (k_start + col)]);
            } else {
                s_a[row][col] = __float2half(0.0f);
            }
        }

        // Load B tile: unpack from packed INT4 → s_b[BK][BN] (K×N layout)
        // Each thread loads (BN*BK/2)/256 = 16 packed bytes → 32 FP16 values
        int b_packed_elems = (BN * BK / 2) / 256;
        for (int j = 0; j < b_packed_elems; j++) {
            int byte_idx_in_tile = tid * b_packed_elems + j;
            int nibble_idx = byte_idx_in_tile * 2;
            int n_row = nibble_idx / BK;     // N-dim row in original N×K layout
            int k_col = nibble_idx % BK;     // K-dim col in original N×K layout

            if (n_row < bn_eff && k_col < bk_eff - 1) {
                int g_row = n_start + n_row;
                int byte_global = g_row * K_packed + (k_start + k_col) / 2;
                uint8_t byte = packed[byte_global];
                uint8_t lo = byte & 0xF;
                uint8_t hi = (byte >> 4) & 0xF;
                int8_t q_lo = (lo > 7) ? (int8_t)(lo - 16) : (int8_t)lo;
                int8_t q_hi = (hi > 7) ? (int8_t)(hi - 16) : (int8_t)hi;
                half scale = s_ws[n_row];
                s_b[k_col][n_row]     = __hmul(__int2half_rn(q_lo), scale);
                s_b[k_col + 1][n_row] = __hmul(__int2half_rn(q_hi), scale);
            } else if (n_row < bn_eff && k_col < bk_eff) {
                int g_row = n_start + n_row;
                int byte_global = g_row * K_packed + (k_start + k_col) / 2;
                uint8_t byte = packed[byte_global];
                uint8_t lo = byte & 0xF;
                int8_t q_lo = (lo > 7) ? (int8_t)(lo - 16) : (int8_t)lo;
                half scale = s_ws[n_row];
                s_b[k_col][n_row] = __hmul(__int2half_rn(q_lo), scale);
                if (k_col + 1 < BK) {
                    s_b[k_col + 1][n_row] = __float2half(0.0f);
                }
            }
        }

        __syncthreads();

        // wmma computation over BK dim, stepping by WMMA_K=16
        for (int kk = 0; kk < bk_eff; kk += WMMA_K) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
            load_matrix_sync(a_frag, &s_a[row_base][kk], BK);

            for (int cg = 0; cg < 8; cg++) {
                int col_base = cg * WMMA_N;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
                load_matrix_sync(b_frag, &s_b[kk][col_base], BN);
                mma_sync(acc[cg], a_frag, b_frag, acc[cg]);
            }
        }

        __syncthreads();
    }

    // Store results
    for (int cg = 0; cg < 8; cg++) {
        int col_base = cg * WMMA_N;
        int g_m = m_start + row_base;
        int g_n = n_start + col_base;
        if (g_m < M && g_n < N) {
            store_matrix_sync(&output[g_m * N + g_n], acc[cg], N, mem_row_major);
        }
    }
}

void launch_fused_gemm_wmma(
    const half* x,
    const uint8_t* packed,
    const half* w_scale,
    half* output,
    int M, int N, int K,
    cudaStream_t stream
) {
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(256);

    fused_gemm_wmma_kernel<<<grid, block, 0, stream>>>(
        x, packed, w_scale, output, M, N, K
    );
}
