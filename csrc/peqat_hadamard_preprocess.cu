#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

/**
 * preprocess_hadamard_kernel
 *
 * One block per token row. 256 threads.
 *
 * Phase 0 (Hadamard butterfly): x = H @ x (in-place in shared memory, normalized by sqrt(K))
 * Phase 1 (reduction): compute per-token s_x[t] = max_k(|x_rotated|) / 7
 * Phase 2 (pack): quantize to INT4, pack 8 per int32
 *
 * No smooth/clip parameters — H is fixed, rotation diffuses outliers.
 *
 * Input:  x[M,K] bf16
 * Output: x_packed[M, K/8] int32, s_x[M] f32
 *
 * Packing: word = q0 | (q1<<4) | (q2<<8) | (q3<<12) | (q4<<16) | (q5<<20) | (q6<<24) | (q7<<28)
 *          where q0 is at k_base+0, q7 at k_base+7 (little-endian K order)
 */
__global__ void preprocess_hadamard_kernel(
    const __nv_bfloat16* __restrict__ x,
    int32_t* __restrict__ x_packed,
    float* __restrict__ s_x,
    int M, int K
) {
    extern __shared__ __nv_bfloat16 smem[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= M) return;

    __nv_bfloat16* s_buf = smem;

    // --- Phase 0: Load + Hadamard butterfly ---
    // Load K elements into shared memory
    for (int k = tid; k < K; k += blockDim.x) {
        s_buf[k] = x[row * K + k];
    }
    __syncthreads();

    // Butterfly: log2(K) levels
    for (int stride = 1; stride < K; stride <<= 1) {
        int pairs = K / (stride * 2);
        int elems_per_pair = stride;
        int my_pair = tid / elems_per_pair;

        if (my_pair < pairs) {
            int base = my_pair * stride * 2;
            int offset = tid % elems_per_pair;
            int idx0 = base + offset;
            int idx1 = base + stride + offset;

            float a0 = __bfloat162float(s_buf[idx0]);
            float a1 = __bfloat162float(s_buf[idx1]);
            s_buf[idx0] = __float2bfloat16(a0 + a1);
            s_buf[idx1] = __float2bfloat16(a0 - a1);
        }
        __syncthreads();
    }

    // Normalize by sqrt(K)
    float inv_sqrtK = rsqrtf((float)K);
    for (int k = tid; k < K; k += blockDim.x) {
        float val = __bfloat162float(s_buf[k]);
        s_buf[k] = __float2bfloat16(val * inv_sqrtK);
    }
    __syncthreads();

    // --- Phase 1: Per-token max reduction ---
    constexpr float QMAX = 7.0f;
    float local_max = 0.0f;

    for (int k = tid; k < K; k += blockDim.x) {
        float xc = fabsf(__bfloat162float(s_buf[k]));
        if (xc > local_max) local_max = xc;
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local_max, offset);
        if (other > local_max) local_max = other;
    }

    __shared__ float s_warp_max[8];
    int warp_id = tid / 32;
    if (tid % 32 == 0) {
        s_warp_max[warp_id] = local_max;
    }
    __syncthreads();

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

    // --- Phase 2: Quantize and pack (using shared memory, no smooth/clip needed) ---
    int n_packed = K / 8;
    int groups_per_thread = (n_packed + blockDim.x - 1) / blockDim.x;
    constexpr float QMIN = -8.0f;
    float s_x_val = s_x[row];

    for (int g = 0; g < groups_per_thread; g++) {
        int group = tid * groups_per_thread + g;
        if (group >= n_packed) continue;

        int k0 = group * 8;
        int32_t word = 0;

        for (int i = 0; i < 8; i++) {
            int k = k0 + i;
            if (k >= K) break;

            float xc = __bfloat162float(s_buf[k]);
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
void launch_preprocess_hadamard(
    const __nv_bfloat16* x,
    int32_t* x_packed,
    float* s_x,
    int M, int K,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = M;
    size_t smem_bytes = K * sizeof(__nv_bfloat16);
    preprocess_hadamard_kernel<<<blocks, threads, smem_bytes, stream>>>(
        x, x_packed, s_x, M, K
    );
}
