#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Fused GEMV decode kernel: one warp per output row.
// Reads packed INT4 weights directly, unpacks in registers,
// computes dot product with FP16 activation, no w_dq global buffer.
__global__ void fused_gemv_decode_kernel(
    const half* __restrict__ x_f16,      // [K] preprocessed activations, FP16
    const uint8_t* __restrict__ packed,  // [R * K/2] packed INT4 weights
    const half* __restrict__ w_scale,    // [R] per-row scales, FP16
    half* __restrict__ output,           // [R] output
    int K, int R
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= R) return;

    int tid = threadIdx.x;
    half wsc = __ldg(&w_scale[row]);
    half sum = 0.0h;

    int K_packed = (K + 1) / 2;

    for (int k = tid; k < K; k += 32) {
        half act = __ldg(&x_f16[k]);

        int byte_idx = row * K_packed + k / 2;
        uint8_t byte = __ldg(&packed[byte_idx]);
        uint8_t nibble = (k & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
        int8_t qw = (nibble > 7) ? (int8_t)(nibble - 16) : (int8_t)nibble;

        sum = __hfma(act, __hmul(__int2half_rn(qw), wsc), sum);
    }

    // reduction fold 1
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum = __hadd(sum, __shfl_xor_sync(0xFFFFFFFF, sum, offset));
    }

    if (tid == 0) {
        output[row] = sum;
    }
}

void launch_fused_gemv_decode(
    const half* x_f16,
    const uint8_t* packed,
    const half* w_scale,
    half* output,
    int K, int R,
    cudaStream_t stream
) {
    // One warp (32 threads) per output row. Use blockDim.y for rows-per-block.
    int warp_per_block = 32;  // 32 warps * 32 threads = 1024 threads max
    int blocks = (R + warp_per_block - 1) / warp_per_block;
    dim3 block(32, warp_per_block);  // 32 threads x 32 warps = 1024 threads

    fused_gemv_decode_kernel<<<blocks, block, 0, stream>>>(
        x_f16, packed, w_scale, output, K, R
    );
}
