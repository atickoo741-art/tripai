#include <cuda_runtime.h>
#include <stdint.h>

// --- Prefill GEMM (large M) ---
#define BM_L 128
#define BN_L 128
#define BK   32
#define BK_STRIDE (BK + 1)
#define THREADS_L 256
#define TM_L 8
#define TN_L 8

__global__ void fused_preprocess_gemm_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const float* __restrict__ w_scale,
    const float* __restrict__ smooth,
    const float* __restrict__ clip,
    float* __restrict__ output,
    int M, int N, int K
) {
    __shared__ float s_smooth[BK];
    __shared__ float s_clip[BK];
    __shared__ float s_A[BM_L * BK_STRIDE];
    __shared__ float s_B[BN_L * BK_STRIDE];
    __shared__ float s_scale[BN_L];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int m_start = bx * BM_L;
    int n_start = by * BN_L;
    int m_end = min(m_start + BM_L, M);
    int n_end = min(n_start + BN_L, N);
    int bm_eff = m_end - m_start;
    int bn_eff = n_end - n_start;

    int tid = threadIdx.x;
    int m_group = tid / 16;
    int n_group = tid % 16;

    float accum[TM_L][TN_L] = {0.0f};

    int K_packed = (K + 1) / 2;

    if (tid < BN_L) {
        int g_idx = n_start + tid;
        s_scale[tid] = (g_idx < N) ? w_scale[g_idx] : 0.0f;
    }
    __syncthreads();

    int a_elems = (BM_L * BK) / THREADS_L;
    int b_packed_bytes = (BN_L * BK / 2) / THREADS_L;

    for (int k_start = 0; k_start < K; k_start += BK) {
        int bk_eff = min(K - k_start, BK);

        // --- Load smooth and clip for this K-tile ---
        if (tid < BK) {
            int g_k = k_start + tid;
            s_smooth[tid] = (g_k < K) ? smooth[g_k] : 0.0f;
            s_clip[tid]   = (g_k < K) ? clip[g_k] : 0.0f;
        }
        __syncthreads();

        // --- Load A tile with inline activation preprocessing ---
        for (int i = 0; i < a_elems; i++) {
            int idx = tid * a_elems + i;
            int row = idx / BK;
            int col = idx % BK;
            int store_idx = row * BK_STRIDE + col;
            if (row < bm_eff && col < bk_eff) {
                int g_row = m_start + row;
                int g_col = k_start + col;
                float x_val = x[g_row * K + g_col];
                float s = fmaxf(s_smooth[col], 1e-8f);
                float c = fmaxf(s_clip[col], 1e-8f);
                float x_s = x_val / s;
                float x_clipped = fminf(fmaxf(x_s, -c), c);
                float s_x = fmaxf(c / 7.0f, 1e-8f);
                float scaled = x_clipped / s_x;
                int q = __float2int_rn(scaled);
                q = min(max(q, -7), 7);
                s_A[store_idx] = (float)q * s_x;
            } else {
                s_A[store_idx] = 0.0f;
            }
        }

        // --- Load B tile: each thread reads 8 packed bytes, unpacks both nibbles ---
        for (int j = 0; j < b_packed_bytes; j++) {
            int byte_idx_in_tile = tid * b_packed_bytes + j;
            int val_idx = byte_idx_in_tile * 2;
            int row = val_idx / BK;
            int col = val_idx % BK;
            if (row < bn_eff && col < bk_eff - 1) {
                int g_row = n_start + row;
                int byte_global = g_row * K_packed + (k_start + col) / 2;
                uint8_t byte = packed[byte_global];
                uint8_t lo = byte & 0xF;
                uint8_t hi = (byte >> 4) & 0xF;
                int8_t q_lo = (lo > 7) ? (int8_t)(lo - 16) : (int8_t)lo;
                int8_t q_hi = (hi > 7) ? (int8_t)(hi - 16) : (int8_t)hi;
                float scale = s_scale[row];
                s_B[row * BK_STRIDE + col]     = (float)q_lo * scale;
                s_B[row * BK_STRIDE + col + 1] = (float)q_hi * scale;
            } else if (row < bn_eff && col < bk_eff) {
                int g_row = n_start + row;
                int byte_global = g_row * K_packed + (k_start + col) / 2;
                uint8_t byte = packed[byte_global];
                uint8_t lo = byte & 0xF;
                int8_t q_lo = (lo > 7) ? (int8_t)(lo - 16) : (int8_t)lo;
                float scale = s_scale[row];
                s_B[row * BK_STRIDE + col] = (float)q_lo * scale;
                if (col + 1 < BK) {
                    s_B[row * BK_STRIDE + col + 1] = 0.0f;
                }
            } else {
                if (col < BK) {
                    s_B[row * BK_STRIDE + col] = 0.0f;
                }
                if (col + 1 < BK) {
                    s_B[row * BK_STRIDE + col + 1] = 0.0f;
                }
            }
        }

        __syncthreads();

        for (int kk = 0; kk < bk_eff; kk++) {
            for (int tm = 0; tm < TM_L; tm++) {
                float a_val = s_A[(m_group * TM_L + tm) * BK_STRIDE + kk];
                for (int tn = 0; tn < TN_L; tn++) {
                    accum[tm][tn] += a_val * s_B[(n_group * TN_L + tn) * BK_STRIDE + kk];
                }
            }
        }

        __syncthreads();
    }

    for (int tm = 0; tm < TM_L; tm++) {
        for (int tn = 0; tn < TN_L; tn++) {
            int g_m = m_start + m_group * TM_L + tm;
            int g_n = n_start + n_group * TN_L + tn;
            if (g_m < M && g_n < N) {
                output[g_m * N + g_n] = accum[tm][tn];
            }
        }
    }
}

// --- Decode GEMM (small M) ---
#define BM_S 8
#define BN_S 128
#define BK_S 32
#define BK_STRIDE_S (BK_S + 1)
#define THREADS_S 256
#define TM_S 1
#define TN_S 4

__global__ void fused_preprocess_gemm_smallm_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const float* __restrict__ w_scale,
    const float* __restrict__ smooth,
    const float* __restrict__ clip,
    float* __restrict__ output,
    int M, int N, int K
) {
    __shared__ float s_smooth[BK_S];
    __shared__ float s_clip[BK_S];
    __shared__ float s_A[BM_S * BK_STRIDE_S];
    __shared__ float s_B[BN_S * BK_STRIDE_S];
    __shared__ float s_scale[BN_S];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int m_start = bx * BM_S;
    int n_start = by * BN_S;
    int m_end = min(m_start + BM_S, M);
    int n_end = min(n_start + BN_S, N);
    int bm_eff = m_end - m_start;
    int bn_eff = n_end - n_start;

    int tid = threadIdx.x;

    int m_local = tid / 32;
    int n_base  = (tid % 32) * TN_S;

    float accum[TN_S] = {0.0f};

    int K_packed = (K + 1) / 2;

    if (tid < BN_S) {
        int g_idx = n_start + tid;
        s_scale[tid] = (g_idx < N) ? w_scale[g_idx] : 0.0f;
    }
    __syncthreads();

    int b_bytes = (BN_S * BK_S / 2) / THREADS_S;

    for (int k_start = 0; k_start < K; k_start += BK_S) {
        int bk_eff = min(K - k_start, BK_S);

        // --- Load smooth and clip for this K-tile ---
        if (tid < BK_S) {
            int g_k = k_start + tid;
            s_smooth[tid] = (g_k < K) ? smooth[g_k] : 0.0f;
            s_clip[tid]   = (g_k < K) ? clip[g_k] : 0.0f;
        }
        __syncthreads();

        // --- Load A tile with inline activation preprocessing ---
        if ((tid % 32) == 0) {
            int g_row = m_start + m_local;
            for (int kk = 0; kk < BK_S; kk++) {
                if (m_local < bm_eff && kk < bk_eff) {
                    float x_val = x[g_row * K + (k_start + kk)];
                    float s = fmaxf(s_smooth[kk], 1e-8f);
                    float c = fmaxf(s_clip[kk], 1e-8f);
                    float x_s = x_val / s;
                    float x_clipped = fminf(fmaxf(x_s, -c), c);
                    float s_x = fmaxf(c / 7.0f, 1e-8f);
                    float scaled = x_clipped / s_x;
                    int q = __float2int_rn(scaled);
                    q = min(max(q, -7), 7);
                    s_A[m_local * BK_STRIDE_S + kk] = (float)q * s_x;
                } else {
                    s_A[m_local * BK_STRIDE_S + kk] = 0.0f;
                }
            }
        }

        // --- Load B tile (byte-pair load) ---
        for (int j = 0; j < b_bytes; j++) {
            int byte_idx_in_tile = tid * b_bytes + j;
            int val_idx = byte_idx_in_tile * 2;
            int row = val_idx / BK_S;
            int col = val_idx % BK_S;
            if (row < bn_eff && col < bk_eff - 1) {
                int g_row = n_start + row;
                int byte_global = g_row * K_packed + (k_start + col) / 2;
                uint8_t byte = packed[byte_global];
                uint8_t lo = byte & 0xF;
                uint8_t hi = (byte >> 4) & 0xF;
                int8_t q_lo = (lo > 7) ? (int8_t)(lo - 16) : (int8_t)lo;
                int8_t q_hi = (hi > 7) ? (int8_t)(hi - 16) : (int8_t)hi;
                float scale = s_scale[row];
                s_B[row * BK_STRIDE_S + col]     = (float)q_lo * scale;
                s_B[row * BK_STRIDE_S + col + 1] = (float)q_hi * scale;
            } else if (row < bn_eff && col < bk_eff) {
                int g_row = n_start + row;
                int byte_global = g_row * K_packed + (k_start + col) / 2;
                uint8_t byte = packed[byte_global];
                uint8_t lo = byte & 0xF;
                int8_t q_lo = (lo > 7) ? (int8_t)(lo - 16) : (int8_t)lo;
                float scale = s_scale[row];
                s_B[row * BK_STRIDE_S + col] = (float)q_lo * scale;
                if (col + 1 < BK_S) {
                    s_B[row * BK_STRIDE_S + col + 1] = 0.0f;
                }
            } else {
                if (col < BK_S) s_B[row * BK_STRIDE_S + col] = 0.0f;
                if (col + 1 < BK_S) s_B[row * BK_STRIDE_S + col + 1] = 0.0f;
            }
        }

        __syncthreads();

        for (int kk = 0; kk < bk_eff; kk++) {
            float a_val = s_A[m_local * BK_STRIDE_S + kk];
            for (int tn = 0; tn < TN_S; tn++) {
                accum[tn] += a_val * s_B[(n_base + tn) * BK_STRIDE_S + kk];
            }
        }

        __syncthreads();
    }

    int g_m = m_start + m_local;
    if (g_m < M) {
        for (int tn = 0; tn < TN_S; tn++) {
            int g_n = n_start + n_base + tn;
            if (g_n < N) {
                output[g_m * N + g_n] = accum[tn];
            }
        }
    }
}

// --- Launch wrappers ---

void launch_fused_preprocess_gemm(
    const float* x,
    const uint8_t* packed,
    const float* w_scale,
    const float* smooth,
    const float* clip,
    float* output,
    int M, int N, int K,
    cudaStream_t stream
) {
    dim3 grid((M + BM_L - 1) / BM_L, (N + BN_L - 1) / BN_L);
    dim3 block(THREADS_L);

    fused_preprocess_gemm_kernel<<<grid, block, 0, stream>>>(
        x, packed, w_scale, smooth, clip, output, M, N, K
    );
}

void launch_fused_preprocess_gemm_smallm(
    const float* x,
    const uint8_t* packed,
    const float* w_scale,
    const float* smooth,
    const float* clip,
    float* output,
    int M, int N, int K,
    cudaStream_t stream
) {
    dim3 grid((M + BM_S - 1) / BM_S, (N + BN_S - 1) / BN_S);
    dim3 block(THREADS_S);

    fused_preprocess_gemm_smallm_kernel<<<grid, block, 0, stream>>>(
        x, packed, w_scale, smooth, clip, output, M, N, K
    );
}
