#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

/**
 * fused_preprocess_tc_per_token_kernel
 *
 * Replaces the two-step pipeline:
 *   1. preprocess_per_token_kernel  (global write x_packed, s_x)
 *   2. fused_int4_tc_per_token_kernel (global read x_packed, s_x)
 *
 * With a single kernel that fuses both:
 *   Phase 1 (max reduction): per-token s_x = max_k(|clamp(x/smooth, -clip, clip)|) / 7
 *   Phase 2 (TC matmul):     for each K-tile of 64, read bf16 x from global,
 *                              quantize+pack on the fly -> s_a -> mma.sync -> accumulate
 *   Phase 3 (dequant):       scale by s_x * w_scale -> bf16 output
 *
 * Eliminates ~224 kernel launches and ~224 MB intermediate HBM traffic per forward pass.
 *
 * Grid:  (N+63)/64  x  (M+15)/16
 * Block: 256 threads
 *
 * Per-token max reduction distribution (Phase 1):
 *   8 warps each handle 2 rows: warp r handles rows m_start+2r (lanes 0-15) and m_start+2r+1 (lanes 16-31).
 *   Shuffle reduction across 16 lanes within each half-warp.
 *
 * On-the-fly quantization (Phase 2a):
 *   Threads 0-127 each load 8 bf16 values from x, apply smooth/clip/divide-by-s_x,
 *   quantize to INT4, pack 8-per-int32, and write to s_a[16][8+1].
 *   smooth[64] and clip[64] for the current K-tile are loaded into shared memory first.
 *
 * K-tiled mma.sync (Phase 2b):
 *   Identical to fused_int4_tc_per_token_kernel: mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
 *
 * Dequant + write (Phase 3):
 *   Accumulator s32 -> float, scale by s_x[row] * w_scale[col], convert to bf16.
 */
__launch_bounds__(256, 3)
__global__ void fused_preprocess_tc_per_token_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ smooth,
    const __nv_bfloat16* __restrict__ clip,
    const int32_t* __restrict__ w_packed,
    const float* __restrict__ w_scale,
    __nv_bfloat16* __restrict__ y,
    int M,
    int N,
    int K
) {
    int t = threadIdx.x;
    int warp_id = t / 32;
    int lane = t % 32;

    int m_start = blockIdx.y * 16;
    int n_start = blockIdx.x * 64;

    if (m_start >= M) return;

    constexpr float QMAX = 7.0f;

    // --- Phase 1: Per-token max reduction (16 rows in parallel) ---
    // Each warp handles 2 rows: lanes 0-15 -> row_in_warp=0, lanes 16-31 -> row_in_warp=1
    // Warp r -> rows m_start + 2r, m_start + 2r + 1

    __shared__ float s_row_max[16];

    int row_in_warp = lane / 16;
    int lane_in_group = lane % 16;
    int row_idx = m_start + warp_id * 2 + row_in_warp;

    float s_x_val;
    if (row_idx < M) {
        float local_max = 0.0f;

        for (int k = lane_in_group; k < K; k += 16) {
            float xv = __bfloat162float(x[(int64_t)row_idx * K + k]);
            float sv = __bfloat162float(smooth[k]);
            float cv = __bfloat162float(clip[k]);
            float xs = xv / sv;
            float xc = fminf(fmaxf(xs, -cv), cv);
            float axc = fabsf(xc);
            if (axc > local_max) local_max = axc;
        }

        unsigned mask = (row_in_warp == 0) ? 0x0000FFFF : 0xFFFF0000;
        for (int offset = 8; offset > 0; offset >>= 1) {
            float other = __shfl_xor_sync(mask, local_max, offset);
            if (other > local_max) local_max = other;
        }

        float scale = local_max / QMAX;
        if (scale < 1e-8f) scale = 1e-8f;
        s_x_val = scale;
    } else {
        s_x_val = 1.0f;
    }

    if (lane_in_group == 0) {
        s_row_max[warp_id * 2 + row_in_warp] = s_x_val;
    }
    __syncthreads();

    // --- Phase 2: K-tiled mma.sync with on-the-fly quantization ---

    int ldm_w = K / 8;
    int warp_offset = warp_id * 8;
    int32_t d[4] = {0, 0, 0, 0};

    __shared__ int32_t s_a[16][8 + 1];
    __shared__ int32_t s_b[64][8 + 1];
    __shared__ __nv_bfloat16 smooth_s[64];
    __shared__ __nv_bfloat16 clip_s[64];

    for (int k_offset = 0; k_offset < K; k_offset += 64) {
        // Step A: Load smooth[64] and clip[64] for this K tile
        if (t < 64) {
            smooth_s[t] = __ldg(&smooth[k_offset + t]);
            clip_s[t] = __ldg(&clip[k_offset + t]);
        }
        __syncthreads();

        // Step B: Quantize + pack x into s_a (threads 0-127)
        if (t < 128) {
            int q_row = t / 8;
            int k8 = t % 8;
            int actual_row = m_start + q_row;

            if (actual_row < M) {
                float s_x_row = s_row_max[q_row];
                int32_t word = 0;
                int k_base = k_offset + k8 * 8;

                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int k = k_base + i;
                    float xv = __bfloat162float(x[(int64_t)actual_row * K + k]);
                    float sv = __bfloat162float(smooth_s[k8 * 8 + i]);
                    float cv = __bfloat162float(clip_s[k8 * 8 + i]);
                    float xs = xv / sv;
                    float xc = fminf(fmaxf(xs, -cv), cv);
                    float scaled = xc / s_x_row;
                    int q = (int)rintf(scaled);
                    if (q < -8) q = -8;
                    if (q > 7) q = 7;
                    word |= (q & 0xF) << (i * 4);
                }
                s_a[q_row][k8] = word;
            } else {
                s_a[q_row][k8] = 0;
            }
        }
        __syncthreads();

        // Step C: Load w into s_b (all 256 threads)
        int b_col = t / 4;
        int b_k8 = (t % 4) * 2;
        int64_t w_base = (int64_t)(n_start + b_col) * ldm_w + k_offset / 8;
        if (n_start + b_col < N) {
          s_b[b_col][b_k8]     = __ldg(&w_packed[w_base + b_k8]);
          s_b[b_col][b_k8 + 1] = __ldg(&w_packed[w_base + b_k8 + 1]);
        } else {
          s_b[b_col][b_k8]     = 0;
          s_b[b_col][b_k8 + 1] = 0;
        }

        __syncthreads();

        // Step D: mma.sync
        int a_gid = lane / 4, a_tid = lane % 4;
        uint32_t a[4];
        a[0] = (uint32_t)s_a[a_gid][a_tid];
        a[1] = (uint32_t)s_a[a_gid + 8][a_tid];
        a[2] = (uint32_t)s_a[a_gid][a_tid + 4];
        a[3] = (uint32_t)s_a[a_gid + 8][a_tid + 4];

        int b_gid = lane / 4, b_tid = lane % 4;
        uint32_t b[2];
        b[0] = (uint32_t)s_b[warp_offset + b_gid][b_tid];
        b[1] = (uint32_t)s_b[warp_offset + b_gid][b_tid + 4];

        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1])
        );

        __syncthreads();
    }

    // --- Phase 3: Dequant + write output ---

    __shared__ float tile[16][64 + 1];

    int d_row_a = lane / 4;
    int d_col_pair = (lane % 4) * 2;
    tile[d_row_a][warp_id * 8 + d_col_pair]          = (float)d[0];
    tile[d_row_a][warp_id * 8 + d_col_pair + 1]      = (float)d[1];
    tile[d_row_a + 8][warp_id * 8 + d_col_pair]      = (float)d[2];
    tile[d_row_a + 8][warp_id * 8 + d_col_pair + 1]  = (float)d[3];

    __syncthreads();

    const int total_tile = 16 * 64;
    for (int i = t; i < total_tile; i += blockDim.x) {
        int lr = i >> 6;
        int lc = i & 63;
        int row = m_start + lr;
        int col = n_start + lc;
        if (row < M && col < N) {
            float val = tile[lr][lc] * s_row_max[lr] * __ldg(&w_scale[col]);
            y[(int64_t)row * N + col] = __float2bfloat16(val);
        }
    }
}


void launch_fused_preprocess_tc_per_token(
    const __nv_bfloat16* x,
    const __nv_bfloat16* smooth,
    const __nv_bfloat16* clip,
    const int32_t* w_packed,
    const float* w_scale,
    __nv_bfloat16* y,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(
            fused_preprocess_tc_per_token_kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxL1
        );
        attr_set = true;
    }

    dim3 block(256);
    dim3 grid((N + 63) / 64, (M + 15) / 16);
    fused_preprocess_tc_per_token_kernel<<<grid, block, 0, stream>>>(
        x, smooth, clip, w_packed, w_scale, y, M, N, K
    );
}
