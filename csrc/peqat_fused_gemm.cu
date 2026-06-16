#include <cuda_runtime.h>
#include <stdint.h>

#define BM 128
#define BN 128
#define BK 32
#define THREADS_PER_BLOCK 256
#define TM 8
#define TN 8

__global__ void fused_int4_gemm_kernel(
    const float* __restrict__ x_dq,
    const uint8_t* __restrict__ packed,
    const float* __restrict__ w_scale,
    float* __restrict__ output,
    int M, int N, int K
) {
    __shared__ float s_A[BM * BK];
    __shared__ float s_B[BN * BK];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int m_start = bx * BM;
    int n_start = by * BN;
    int m_end = min(m_start + BM, M);
    int n_end = min(n_start + BN, N);
    int bm_eff = m_end - m_start;
    int bn_eff = n_end - n_start;

    int tid = threadIdx.x;
    int m_group = tid / 16;
    int n_group = tid % 16;

    float accum[TM][TN] = {0.0f};

    int K_packed = (K + 1) / 2;

    int a_elems = (BM * BK) / THREADS_PER_BLOCK;
    int b_vals = (BN * BK) / THREADS_PER_BLOCK;

    for (int k_start = 0; k_start < K; k_start += BK) {
        int bk_eff = min(K - k_start, BK);

        for (int i = 0; i < a_elems; i++) {
            int idx = tid * a_elems + i;
            int row = idx / BK;
            int col = idx % BK;
            if (row < bm_eff && col < bk_eff) {
                int g_row = m_start + row;
                int g_col = k_start + col;
                s_A[row * BK + col] = x_dq[g_row * K + g_col];
            } else {
                s_A[row * BK + col] = 0.0f;
            }
        }

        for (int i = 0; i < b_vals; i++) {
            int idx = tid * b_vals + i;
            int row = idx / BK;
            int col = idx % BK;
            if (row < bn_eff && col < bk_eff) {
                int g_row = n_start + row;
                int g_col = k_start + col;
                int byte_idx = g_row * K_packed + g_col / 2;
                uint8_t byte = packed[byte_idx];
                uint8_t nibble = (g_col % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF);
                int8_t q = (nibble > 7) ? (int8_t)(nibble - 16) : (int8_t)nibble;
                s_B[row * BK + col] = (float)q * w_scale[g_row];
            } else {
                s_B[row * BK + col] = 0.0f;
            }
        }

        __syncthreads();

        for (int kk = 0; kk < bk_eff; kk++) {
            for (int tm = 0; tm < TM; tm++) {
                float a_val = s_A[(m_group * TM + tm) * BK + kk];
                for (int tn = 0; tn < TN; tn++) {
                    accum[tm][tn] += a_val * s_B[(n_group * TN + tn) * BK + kk];
                }
            }
        }

        __syncthreads();
    }

    for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn++) {
            int g_m = m_start + m_group * TM + tm;
            int g_n = n_start + n_group * TN + tn;
            if (g_m < M && g_n < N) {
                output[g_m * N + g_n] = accum[tm][tn];
            }
        }
    }
}

void launch_fused_int4_gemm(
    const float* x_dq,
    const uint8_t* packed,
    const float* w_scale,
    float* output,
    int M, int N, int K,
    cudaStream_t stream
) {
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(THREADS_PER_BLOCK);

    fused_int4_gemm_kernel<<<grid, block, 0, stream>>>(
        x_dq, packed, w_scale, output, M, N, K
    );
}
