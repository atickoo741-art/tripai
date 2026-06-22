#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

/**
 * hadamard_butterfly_kernel
 *
 * Applies normalized Walsh-Hadamard transform along the last dimension.
 * One block per row. Shared memory for K bf16 elements.
 *
 * K must be a power of 2 and K * sizeof(__nv_bfloat16) <= shared_mem_per_block.
 *
 * Forward and backward are the same kernel because H is symmetric (H^T = H).
 */
__global__ void hadamard_butterfly_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int M, int K
) {
    extern __shared__ __nv_bfloat16 smem[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= M) return;

    // Coalesced load
    for (int k = tid; k < K; k += blockDim.x) {
        smem[k] = x[row * K + k];
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

            float a0 = __bfloat162float(smem[idx0]);
            float a1 = __bfloat162float(smem[idx1]);
            smem[idx0] = __float2bfloat16(a0 + a1);
            smem[idx1] = __float2bfloat16(a0 - a1);
        }
        __syncthreads();
    }

    // Normalize by sqrt(K)
    float inv_sqrtK = rsqrtf((float)K);
    for (int k = tid; k < K; k += blockDim.x) {
        float val = __bfloat162float(smem[k]);
        smem[k] = __float2bfloat16(val * inv_sqrtK);
    }
    __syncthreads();

    // Coalesced store
    for (int k = tid; k < K; k += blockDim.x) {
        y[row * K + k] = smem[k];
    }
}


void launch_hadamard_butterfly(
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    int M, int K,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = M;
    size_t smem_bytes = K * sizeof(__nv_bfloat16);
    hadamard_butterfly_kernel<<<blocks, threads, smem_bytes, stream>>>(
        x, y, M, K
    );
}
