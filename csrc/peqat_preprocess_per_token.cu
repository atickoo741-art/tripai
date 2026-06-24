#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

/**
 * preprocess_per_token_kernel
 *
 * One block per token row. 256 threads.
 *
 * Phase 1 (reduction): compute per-token s_x[t] = max_k(|clamp(x/smooth)|) / 7
 * Phase 2 (pack):      quantize to INT4, pack 8 per int32 (direct MMA-ready format)
 *
 * Input:  x[M,K] bf16, smooth[K] bf16, clip[K] bf16
 * Output: x_packed[M, K/8] int32, s_x[M] f32
 *
 * Packing: word = q0 | (q1<<4) | (q2<<8) | (q3<<12) | (q4<<16) | (q5<<20) | (q6<<24) | (q7<<28)
 *          where q0 is at k_base+0, q7 at k_base+7 (little-endian K order)
 */
__global__ void preprocess_per_token_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ smooth,
    const __nv_bfloat16* __restrict__ clip,
    int32_t* __restrict__ x_packed,
    float* __restrict__ s_x,
    int M, int K
) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= M) return;

    constexpr float QMAX = 7.0f;
    constexpr float QMIN = -8.0f;

    // --- Phase 1: Per-token max reduction ---
    float local_max = 0.0f;

    for (int k = tid * 2; k < K; k += blockDim.x * 2) {
        __nv_bfloat162 xw = __ldg((const __nv_bfloat162*)&x[row * K + k]);
        __nv_bfloat162 sw = __ldg((const __nv_bfloat162*)&smooth[k]);
        __nv_bfloat162 cw = __ldg((const __nv_bfloat162*)&clip[k]);

        float xv0 = __bfloat162float(xw.x), xv1 = __bfloat162float(xw.y);
        float sv0 = __bfloat162float(sw.x), sv1 = __bfloat162float(sw.y);
        float cv0 = __bfloat162float(cw.x), cv1 = __bfloat162float(cw.y);

        float xs0 = xv0 / sv0, xs1 = xv1 / sv1;
        float xc0 = fminf(fmaxf(xs0, -cv0), cv0);
        float xc1 = fminf(fmaxf(xs1, -cv1), cv1);
        if (fabsf(xc0) > local_max) local_max = fabsf(xc0);
        if (fabsf(xc1) > local_max) local_max = fabsf(xc1);
    }

    // Warp shuffle reduction (first 32 threads)
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local_max, offset);
        if (other > local_max) local_max = other;
    }

    // Write warp max to shared memory
    __shared__ float s_warp_max[8];
    int warp_id = tid / 32;
    if (tid % 32 == 0) {
        s_warp_max[warp_id] = local_max;
    }
    __syncthreads();

    // Final reduction in warp 0
    float block_max = (warp_id == 0 && tid < (blockDim.x / 32))
        ? s_warp_max[tid]
        : 0.0f;

    if (warp_id == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_xor_sync(0xFFFFFFFF, block_max, offset);
            if (other > block_max) block_max = other;
        }
        if (tid == 0) {
            float scale = block_max / QMAX;
            if (scale < 1e-8f) scale = 1e-8f;
            s_x[row] = scale;
        }
    }
    __syncthreads();

    // --- Phase 2: Quantize and pack 8 per int32 (direct MMA-ready format) ---
    int n_packed = K / 8;
    int groups_per_thread = (n_packed + blockDim.x - 1) / blockDim.x;
    float s_x_val = s_x[row];

    for (int g = 0; g < groups_per_thread; g++) {
        int group = tid * groups_per_thread + g;
        if (group >= n_packed) continue;

        int k0 = group * 8;
        int32_t word = 0;

        for (int i = 0; i < 8; i++) {
            int k = k0 + i;
            if (k >= K) break;

            float xv = __bfloat162float(x[row * K + k]);
            float sv = __bfloat162float(smooth[k]);
            float cv = __bfloat162float(clip[k]);

            float xs = xv / sv;
            float xc = fminf(fmaxf(xs, -cv), cv);
            float scaled = xc / s_x_val;
            int q = (int)rintf(scaled);
            if (q < (int)QMIN) q = (int)QMIN;
            if (q > (int)QMAX) q = (int)QMAX;

            word |= (q & 0xF) << (i * 4);
        }

        x_packed[row * n_packed + group] = word;
    }
}


// Launch wrapper
void launch_preprocess_per_token(
    const __nv_bfloat16* x,
    const __nv_bfloat16* smooth,
    const __nv_bfloat16* clip,
    int32_t* x_packed,
    float* s_x,
    int M, int K,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = M;
    preprocess_per_token_kernel<<<blocks, threads, 0, stream>>>(
        x, smooth, clip, x_packed, s_x, M, K
    );
}
