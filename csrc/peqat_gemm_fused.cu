#include <cuda_runtime.h>
#include <stdint.h>

// Fused INT4 GEMV for decode (N=1) and small batches.
// One thread per output row; unpacks INT4 nibbles on the fly.
// x_dq is read from global memory and cached in L2.
__global__ void fused_gemm_kernel(
    const float* __restrict__ x_dq,
    const uint8_t* __restrict__ packed,
    const half* __restrict__ w_scale,
    float* __restrict__ y,
    int N,
    int D,
    int K
) {
    // Each block handles one row of output across all batch items.
    // threadIdx.x selects the batch item n.
    // blockIdx.x selects the output row j.
    int j = blockIdx.x;
    int n = threadIdx.x;
    if (n >= N) return;

    float scale = (float)w_scale[j];
    float acc = 0.0f;

    // Process K elements for output row j, batch n
    // x_dq[n * K + i] is read by all blocks — shared L2 cache
    for (int i = 0; i < K; i++) {
        int packed_idx = ((j * K + i) >> 1);
        uint8_t byte = packed[packed_idx];
        uint8_t nibble = (i & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
        int8_t q = (nibble > 7) ? (int8_t)(nibble - 16) : (int8_t)nibble;
        acc += x_dq[n * K + i] * (float)q;
    }

    // Dequantize with per-row scale
    y[n * D + j] = acc * scale;
}


// Host wrapper: chooses fused kernel for small N, fallback to external
// dequant+cuBLAS for large N (handled at the binding level).
void launch_fused_gemm(
    const float* x_dq,
    const uint8_t* packed,
    const half* w_scale,
    float* y,
    int N,
    int D,
    int K,
    cudaStream_t stream
) {
    // One block per output row, one thread per batch item
    int threads = N;
    int blocks = D;

    // Limit threads to 1024 (max per block on most architectures)
    if (threads > 1024) threads = 1024;

    fused_gemm_kernel<<<blocks, threads, 0, stream>>>(
        x_dq, packed, w_scale, y, N, D, K
    );
}
