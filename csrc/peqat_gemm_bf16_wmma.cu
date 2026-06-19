#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <stdint.h>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 128
#define BN 64
#define BK 64
#define BK_STRIDE (BK + 8)
#define BN_STRIDE (BN + 8)

using namespace nvcuda::wmma;

// Fused GEMV decode kernel (M=1): one warp per output row.
// Reads packed INT4 weights directly, unpacks in registers,
// computes dot with bf16 activation in float accumulator.
// No w_dq global memory write.
__global__ void fused_gemv_bf16_kernel(
    const __nv_bfloat16* __restrict__ x_bf16,
    const uint8_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ w_scale,
    __nv_bfloat16* __restrict__ output,
    int K, int R
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= R) return;

    int tid = threadIdx.x;
    float wsc = __bfloat162float(__ldg(&w_scale[row]));
    float sum = 0.0f;

    int K_packed = (K + 1) / 2;

    for (int k = tid; k < K; k += 32) {
        float act = __bfloat162float(__ldg(&x_bf16[k]));

        int byte_idx = row * K_packed + k / 2;
        uint8_t byte = __ldg(&packed[byte_idx]);
        uint8_t nibble = (k & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
        int8_t qw = (nibble > 7) ? (int8_t)(nibble - 16) : (int8_t)nibble;

        sum = fmaf(act, (float)qw * wsc, sum);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    if (tid == 0) {
        output[row] = __float2bfloat16_rn(sum);
    }
}

void launch_fused_gemv_bf16(
    const __nv_bfloat16* x_bf16,
    const uint8_t* packed,
    const __nv_bfloat16* w_scale,
    __nv_bfloat16* output,
    int K, int R,
    cudaStream_t stream
) {
    int warp_per_block = 32;
    int blocks = (R + warp_per_block - 1) / warp_per_block;
    dim3 block(32, warp_per_block);

    fused_gemv_bf16_kernel<<<blocks, block, 0, stream>>>(
        x_bf16, packed, w_scale, output, K, R
    );
}

// Fused GEMM kernel using bf16 WMMA tensor cores (M > 1).
// INT4 weights loaded from global, unpacked to bf16 in shared memory,
// then consumed by wmma::mma_sync with bf16 fragments.
// No w_dq global memory write.
__global__ void fused_gemm_bf16_wmma_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ w_scale,
    __nv_bfloat16* __restrict__ output,
    int M, int N, int K
) {
    __shared__ __nv_bfloat16 s_a[BM][BK_STRIDE];
    __shared__ __nv_bfloat16 s_b[BK][BN_STRIDE];
    __shared__ __nv_bfloat16 s_ws[BN];

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
        s_ws[tid] = (n_start + tid < N) ? __ldg(&w_scale[n_start + tid]) : __float2bfloat16_rn(0.0f);
    }
    __syncthreads();

    // 4 column-groups of 16 (BN=64)
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        fill_fragment(acc[i], 0.0f);
    }

    for (int k_start = 0; k_start < K; k_start += BK) {
        int bk_eff = min(K - k_start, BK);

        // Load A tile: BM x BK
        int a_elems = (BM * BK) / 256;
        for (int i = 0; i < a_elems; i++) {
            int idx = tid * a_elems + i;
            int row = idx / BK;
            int col = idx % BK;
            if (row < bm_eff && col < bk_eff) {
                s_a[row][col] = __ldg(&x[(m_start + row) * K + (k_start + col)]);
            } else {
                s_a[row][col] = __float2bfloat16_rn(0.0f);
            }
        }

        // Load B tile: packed INT4 -> dequant to bf16 in shared memory
        int b_packed_elems = (BN * BK / 2) / 256;
        for (int j = 0; j < b_packed_elems; j++) {
            int byte_idx_in_tile = tid * b_packed_elems + j;
            int nibble_idx = byte_idx_in_tile * 2;
            int n_row = nibble_idx / BK;
            int k_col = nibble_idx % BK;

            if (n_row < bn_eff && k_col < bk_eff - 1) {
                int g_row = n_start + n_row;
                int byte_global = g_row * K_packed + (k_start + k_col) / 2;
                uint8_t byte = packed[byte_global];
                uint8_t lo = byte & 0xF;
                uint8_t hi = (byte >> 4) & 0xF;
                int8_t q_lo = (lo > 7) ? (int8_t)(lo - 16) : (int8_t)lo;
                int8_t q_hi = (hi > 7) ? (int8_t)(hi - 16) : (int8_t)hi;
                __nv_bfloat16 scale = s_ws[n_row];
                s_b[k_col][n_row]     = __hmul(__float2bfloat16_rn((float)q_lo), scale);
                s_b[k_col + 1][n_row] = __hmul(__float2bfloat16_rn((float)q_hi), scale);
            } else if (n_row < bn_eff && k_col < bk_eff) {
                int g_row = n_start + n_row;
                int byte_global = g_row * K_packed + (k_start + k_col) / 2;
                uint8_t byte = packed[byte_global];
                uint8_t lo = byte & 0xF;
                int8_t q_lo = (lo > 7) ? (int8_t)(lo - 16) : (int8_t)lo;
                __nv_bfloat16 scale = s_ws[n_row];
                s_b[k_col][n_row] = __hmul(__float2bfloat16_rn((float)q_lo), scale);
            }
        }

        __syncthreads();

        // WMMA computation
        for (int kk = 0; kk < bk_eff; kk += WMMA_K) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
            load_matrix_sync(a_frag, &s_a[row_base][kk], BK_STRIDE);

            for (int cg = 0; cg < 4; cg++) {
                int col_base = cg * WMMA_N;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> b_frag;
                load_matrix_sync(b_frag, &s_b[kk][col_base], BN_STRIDE);
                mma_sync(acc[cg], a_frag, b_frag, acc[cg]);
            }
        }

        __syncthreads();
    }

    for (int cg = 0; cg < 4; cg++) {
        int col_base = cg * WMMA_N;
        int g_m = m_start + row_base;
        int g_n = n_start + col_base;
        if (g_m < M && g_n < N) {
            store_matrix_sync(&output[g_m * N + g_n], acc[cg], N, mem_row_major);
        }
    }
}

void launch_fused_gemm_bf16_wmma(
    const __nv_bfloat16* x,
    const uint8_t* packed,
    const __nv_bfloat16* w_scale,
    __nv_bfloat16* output,
    int M, int N, int K,
    cudaStream_t stream
) {
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(256);

    fused_gemm_bf16_wmma_kernel<<<grid, block, 0, stream>>>(
        x, packed, w_scale, output, M, N, K
    );
}
