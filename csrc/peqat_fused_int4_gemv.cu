#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

/**
 * fused_int4_gemv_kernel
 *
 * INT4 GEMV for single-token decode (M=1) on SM 12.1.
 * Avoids padding overhead of the m16n8k64 MMA tile (16x compute waste).
 *
 * Layout: 8 warps x 32 threads.
 *   Each warp processes consecutive columns.
 *   Within a warp, lane t accumulates K-dim elements at stride 32:
 *     w_packed[n*K8 + t], w_packed[n*K8 + t+32], ...
 *   This gives fully coalesced global reads (consecutive addresses per warp).
 *   After partial dot product, warp shuffle reduction sums across lanes.
 *
 * x_packed: [1, K/8] int32, dense 8 s4 per word
 * w_packed: [N, K/8] int32, dense 8 s4 per word
 * w_scale:  [N] float32
 * s_x:      [1] float32
 * y:        [1, N] bfloat16
 */
__global__ void fused_int4_gemv_kernel(
    const int32_t* __restrict__ x_packed,
    const int32_t* __restrict__ w_packed,
    const float* __restrict__ w_scale,
    const float* __restrict__ s_x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int K8 = K / 8;

    // Phase 1: load x into shared memory
    __shared__ int32_t s_x_packed[2048];
    for (int i = tid; i < K8; i += blockDim.x) {
        s_x_packed[i] = x_packed[i];
    }
    __syncthreads();

    float s_x_val = s_x[0];

    // Phase 2: each warp handles a group of consecutive columns
    int num_warps = blockDim.x / 32;
    int cols_per_warp = (N + num_warps - 1) / num_warps;
    int start_n = warp_id * cols_per_warp;
    int end_n = min(start_n + cols_per_warp, N);

    for (int n = start_n; n < end_n; n++) {
        int32_t partial = 0;

        for (int i = lane; i < K8; i += 32) {
            int32_t xw = s_x_packed[i];
            int32_t ww = w_packed[(int64_t)n * K8 + i];

            for (int j = 0; j < 8; j++) {
                int xv = ((int)(xw >> (j * 4))) & 0xF;
                int wv = ((int)(ww >> (j * 4))) & 0xF;
                xv = (xv << 28) >> 28;
                wv = (wv << 28) >> 28;
                partial += xv * wv;
            }
        }

        // Warp shuffle reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_xor_sync(0xFFFFFFFF, partial, offset);
        }
        if (lane == 0) {
            float val = (float)partial * s_x_val * w_scale[n];
            y[n] = __float2bfloat16(val);
        }
    }
}

void launch_fused_int4_gemv(
    const int32_t* x_packed,
    const int32_t* w_packed,
    const float* w_scale,
    const float* s_x,
    __nv_bfloat16* y,
    int N,
    int K,
    cudaStream_t stream
) {
    int threads = 256;
    fused_int4_gemv_kernel<<<1, threads, 0, stream>>>(
        x_packed, w_packed, w_scale, s_x, y, N, K
    );
}
