#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 32

#define BM 128
#define BN 64
#define BK 32

#define BK_STRIDE (BK + 0)
#define BN_STRIDE (BN + 0)

using namespace nvcuda::wmma;

// Fused INT4 tensor core GEMM using bmma_sync (SM 8.0+).
// A: preprocessed INT4 activations (x_q), row-major int8 in global memory
// B: packed INT4 weights (weight_int4.bin), low_high nibble packing in K dim
// Output: FP16, no global-memory FP16 weight materialization.
__global__ void fused_gemm_int4_tc_kernel(
    const int8_t* __restrict__ x_q,
    const uint8_t* __restrict__ packed,
    const half* __restrict__ w_scale,
    const float* __restrict__ a_scale,
    half* __restrict__ output,
    int M, int N, int K
) {
    // Shared memory layout:
    // s_a: BM×BK INT4 values, packed as uint32_t, col_major (stride = BM elements)
    // s_b: BK×BN INT4 values, packed as uint32_t, col_major (stride = BK elements)
    // s_ws: weight scales for the N tile
    // s_as: activation scales for the K tile
    __shared__ uint32_t s_a[BM * BK / 8];
    __shared__ uint32_t s_b[BK * BN / 8];
    __shared__ half s_ws[BN];
    __shared__ float s_as[BK];

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

    int K_packed = (K + 1) / 2;

    // Load weight scales for this N tile
    if (tid < BN) {
        s_ws[tid] = (n_start + tid < N) ? __ldg(&w_scale[n_start + tid]) : __float2half(0.0f);
    }
    __syncthreads();

    // Accumulators: each warp covers 16 M rows × all BN cols
    // 16×16 int32 fragments for each of 4 column groups (BN=64, 4 groups of 16)
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        fill_fragment(acc[i], 0);
    }

    for (int k_start = 0; k_start < K; k_start += BK) {
        int bk_eff = min(K - k_start, BK);

        // Load activation scales for this K tile
        if (tid < BK) {
            int g_k = k_start + tid;
            s_as[tid] = (g_k < K) ? __ldg(&a_scale[g_k]) : 0.0f;
        }

        // --- Load A tile: x_q int8 → packed INT4 col_major in s_a ---
        // x_q[N, K] is row-major int8. We need A[BM, BK] in col-major INT4 packed.
        // Each column k has BM INT4 values packed as BM/2 bytes = BM/8 uint32_t.
        // Load all elements for this K-tile into s_a.
        // Thread layout: each thread pack a group of rows for a given column
        int a_uint32_per_col = BM / 8;
        int a_total_uint32 = BK * a_uint32_per_col;
        int a_elems_per_thread = a_total_uint32 / 256;
        if (a_total_uint32 % 256 != 0) a_elems_per_thread++;

        for (int i = tid; i < a_total_uint32; i += 256) {
            int col = i / a_uint32_per_col;
            int row_group_in_tile = (i % a_uint32_per_col) * 8; // 8 rows per uint32_t
            if (col < bk_eff && row_group_in_tile < bm_eff) {
                // Pack 8 INT4 values from x_q into one uint32_t
                uint32_t pack = 0;
                #pragma unroll
                for (int rr = 0; rr < 8; rr++) {
                    int g_row = m_start + row_group_in_tile + rr;
                    int g_col = k_start + col;
                    int8_t val = (g_row < M && g_col < K) ? x_q[g_row * K + g_col] : (int8_t)0;
                    uint8_t nib = (uint8_t)(val & 0x0F);
                    pack |= ((uint32_t)nib << (rr * 4));
                }
                s_a[col * a_uint32_per_col + row_group_in_tile / 8] = pack;
            }
        }

        // --- Load B tile: packed_w → INT4 col_major in s_b ---
        // packed_w stores W[N, K] in row-major, low_high packing in K dim.
        // For col_major bmma_b: B[K, N] where column c stores 32 K values from row W[n_start+c].
        // Each column has BK/8 = 4 uint32_t.
        int b_uint32_per_col = BK / 8;
        int b_total_uint32 = BN * b_uint32_per_col;
        for (int i = tid; i < b_total_uint32; i += 256) {
            int n_col = i / b_uint32_per_col;
            int k_uint32_in_col = i % b_uint32_per_col;
            int k_row = k_uint32_in_col * 8;

            if (n_col < bn_eff) {
                int g_row = n_start + n_col;
                // Load 8 INT4 values from row g_row of the weight matrix
                // packed_w is byte-packed with low_high nibble order
                // 8 INT4 values = 4 bytes
                uint32_t pack = 0;
                #pragma unroll
                for (int rr = 0; rr < 8; rr++) {
                    int g_k = k_start + k_row + rr;
                    if (g_k < K) {
                        int byte_idx = g_row * K_packed + g_k / 2;
                        uint8_t byte = packed[byte_idx];
                        uint8_t nibble = (g_k & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                        pack |= ((uint32_t)nibble << (rr * 4));
                    }
                }
                s_b[n_col * b_uint32_per_col + k_uint32_in_col] = pack;
            }
        }

        __syncthreads();

        // bmma computation: each warp processes a 16×32 A subtile × BN columns of B
        {
            // Load A fragment: col_major 16×32, BK_STRIDE = BK (no padding)
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, experimental::precision::b4, col_major> a_frag;
            load_matrix_sync(a_frag, &s_a[0 * (BM/8) + (row_group * WMMA_M) / 8], BM);

            for (int cg = 0; cg < 4; cg++) {
                int col_base = cg * WMMA_N;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, experimental::precision::b4, col_major> b_frag;
                load_matrix_sync(b_frag, &s_b[col_base * (BK/8) + 0], BK);
                mma_sync(acc[cg], a_frag, b_frag, acc[cg]);
            }
        }

        __syncthreads();
    }

    // Post-multiply by weight scale and activation scale, convert to FP16
    for (int cg = 0; cg < 4; cg++) {
        int col_base = cg * WMMA_N;
        int g_m = m_start + row_group * WMMA_M;
        int g_n = n_start + col_base;

        // acc[cg] is int32[16][16] distributed across warp threads
        // Each thread holds elements of the fragment
        half* out_row = &output[g_m * N + g_n];

        // Each thread processes its own elements of the fragment
        #pragma unroll
        for (int i = 0; i < acc[cg].num_elements; i++) {
            int32_t raw = acc[cg].x[i];

            // Determine the global position of this element
            // For a fragment of accumulator, element indices follow the
            // logical (row, col) within the 16×16 output tile
            // We use the mapping: row = i / 16, col = i % 16 (row-major)
            int local_row = i / WMMA_N;
            int local_col = i % WMMA_N;

            int out_r = g_m + local_row;
            int out_c = g_n + local_col;

            if (out_r < M && out_c < N) {
                // Dequantize: multiply int32 accumulator by weight_scale and activation_scale
                // The accumulated value is sum_k (A_q[k] * W_q[k]) where both are INT4
                // A_q = round(x / a_scale), W_q = round(W / w_scale)
                // Actual computation: sum_k A[k] * W[k]
                //   = sum_k (A_q[k] * a_scale[k]) * (W_q[k] * w_scale[out_r])
                //   = a_scale[k] * w_scale[out_r] * sum_k A_q[k] * W_q[k]
                // But a_scale varies per k! We can't factor it out.
                //
                // Since activation scale is per-channel (per K), not per-token or per-group,
                // and it varies along K, we need to handle it differently.
                //
                // For INT4 TC: the int32 accumulator gives us sum_k A_q[k] * W_q[k]
                // We need to multiply by w_scale[out_c] (per output channel)
                // a_scale is per-K-channel, which we can't apply post-hoc
                //
                // APPROACH: Pre-multiply A_q by a_scale factor before MMA, or
                // incorporate a_scale into the A-side fragment load.
                //
                // For now: use the activation FP16 dequantized values instead,
                // and only apply weight dequant post-MMA.
                // Actual: accum = sum_k A_dq[k] * W_q[k]
                // where A_dq[k] = A_q[k] * a_scale[k]
                // Then output = accum * w_scale[out_c]
                //
                // But the bmma does INT4×INT4, so A_q[k] is INT4, not FP16.
                // The multiplication by a_scale needs to happen elsewhere.
                //
                // WORKAROUND: We'll pre-multiply a_scale into the weights.
                // During the B tile load, multiply each weight nibble by a_scale.
                // Then the MMA computes: sum_k A_q[k] * (W_q[k] * a_scale[k])
                //   = sum_k A_q[k] * W_q[k] * a_scale[k]
                // Output = accum * w_scale[out_c]
                //
                // Actually this doesn't work because a_scale varies by k and the
                // bmma is element-wise INT4×INT4.
                //
                // SIMPLEST CORRECT APPROACH: Modify the B tile load to incorporate a_scale.
                // Since a_scale[k] is the same for all rows in the A tile, we can
                // multiply each weight nibble by sqrt(a_scale) or similar.
                //
                // NO - the issue is fundamental: bmma does INT4×INT4 in integer, not FP16.
                // We can't mix scales into the INT4 domain easily.
                //
                // For A100 SM 80+: the only practical way is to use FP16 MMA (wmma),
                // or pre-compute the scaling differently.
                //
                // ACTUAL CORRECT APPROACH:
                // The bmma does integer compute. We load A_q (INT4) and W_q (INT4).
                // Result is int32: sum_k A_q[k] * W_q[k]
                // To get the real result: sum_k A[k] * W[k]
                //   = sum_k (A_q[k] * a_scale[k]) * (W_q[k] * w_scale[out_c])
                //   = w_scale[out_c] * sum_k A_q[k] * W_q[k] * a_scale[k]
                //
                // Since a_scale varies per k, we cannot factor it out.
                // We MUST incorporate a_scale into the MMA somehow.
                //
                // OPTION 1: Pre-scale A values by a_scale during A tile load
                // Problem: A values become FP16, not INT4. Bmma is INT4 only.
                //
                // OPTION 2: Fold a_scale into weights before quantization
                // This is what SmoothQuant does - already done in the export pipeline!
                // The smooth factor is folded into weights and activations.
                //
                // Actually, a_scale here is the activation quant scale (clip/7).
                // The SmoothQuant factor is already applied to inputs and weights.
                // So A_k = x_s[k] = input[k] / smooth[k]
                // Then A_q = round(A_k / a_scale[k]) where a_scale = clip[k] / 7
                // W_q = round(W_orig[k] * smooth[k] / w_scale)
                //
                // The true output = sum_k A_k * W_orig[k]
                //   = sum_k (A_q * a_scale) * (W_q * w_scale / smooth)
                //   = a_scale * w_scale / smooth * sum_k A_q * W_q  (only if smooth≡1)
                // Since smooth varies per k, we CANNOT factor out.
                //
                // BUT: the export pipeline already folds smooth into weights:
                // W_smoothed = W_orig * smooth
                // And divides smooth from activations: A_smoothed = A / smooth
                // So the weight has smooth folded in, and activation has smooth divided out.
                //
                // After smoothing: W_smoothed_q = round(W_smoothed / w_scale)
                //   A_smoothed_q = round(A_smoothed / a_scale) = round(A / smooth / a_scale)
                //
                // True output = sum_k A[k] * W_orig[k]
                //   = sum_k (A_smoothed[k] * smooth[k]) * (W_smoothed / smooth[k])
                //   = sum_k A_smoothed[k] * W_smoothed[k]
                //   = sum_k (A_q * a_scale) * (W_q * w_scale)
                //   = a_scale[k] * w_scale * sum_k A_q[k] * W_q[k]
                //
                // The a_scale still varies per k! So factorization fails per-k.
                //
                // CORRECT APPROACH: Use FP16 MMA (our existing WMMA kernel).
                // The bmma INT4 MMA on A100 doesn't help here because we need
                // per-K-channel scaling that can't be factored out of the dot product.
                //
                // For a per-channel quantization scheme like PE-QAT:
                // - A_scale varies per K channel
                // - W_scale varies per N channel
                // - Only W_scale can be factored out of the K-sum
                // - A_scale must be applied per K element
                //
                // This is exactly why our existing FP16 WMMA path is the correct
                // approach: we dequantize both A and W to FP16 in shared memory,
                // apply per-K scaling, then do FP16 MMA.
                //
                // The INT4 TC on A100 only works for per-tensor or per-group
                // quantization where a_scale is constant across K.
                //
                // CONCLUSION: PE-QAT's per-channel activation quantization
                // cannot use INT4 tensor cores directly. The per-K scaling
                // requires FP16 arithmetic per element.

                // For correctness testing, just store the raw int32 result
                // This will NOT produce correct PPL. Use the FP16 WMMA path instead.
                out_row[local_row * N + local_col] = __float2half((float)raw);
            }
        }
    }
}

void launch_fused_gemm_int4_tc(
    const int8_t* x_q,
    const uint8_t* packed,
    const half* w_scale,
    const float* a_scale,
    half* output,
    int M, int N, int K,
    cudaStream_t stream
) {
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(256);

    fused_gemm_int4_tc_kernel<<<grid, block, 0, stream>>>(
        x_q, packed, w_scale, a_scale, output, M, N, K
    );
}
