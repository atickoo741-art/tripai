#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

__global__ void unpack_dequant_f16_kernel(
    const uint8_t* __restrict__ packed,
    const half* __restrict__ w_scale,
    half* __restrict__ w_dq,
    int rows,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if (idx < total) {
        int r = idx / cols;
        int c = idx % cols;

        int packed_idx = (r * cols + c) / 2;
        bool is_low = (c % 2 == 0);

        uint8_t byte = packed[packed_idx];
        uint8_t nibble = is_low ? (byte & 0xF) : ((byte >> 4) & 0xF);
        int8_t q = (nibble > 7) ? (int8_t)(nibble - 16) : (int8_t)nibble;

        w_dq[idx] = __hmul(__int2half_rn(q), w_scale[r]);
    }
}

void launch_unpack_dequant_f16(
    const uint8_t* packed,
    const half* w_scale,
    half* w_dq,
    int rows,
    int cols,
    cudaStream_t stream
) {
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    unpack_dequant_f16_kernel<<<blocks, threads, 0, stream>>>(
        packed, w_scale, w_dq, rows, cols
    );
}

__global__ void unpack_dequant_bf16_kernel(
    const uint8_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ w_scale,
    __nv_bfloat16* __restrict__ w_dq,
    int rows,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if (idx < total) {
        int r = idx / cols;
        int c = idx % cols;

        int packed_idx = (r * cols + c) / 2;
        bool is_low = (c % 2 == 0);

        uint8_t byte = packed[packed_idx];
        uint8_t nibble = is_low ? (byte & 0xF) : ((byte >> 4) & 0xF);
        int8_t q = (nibble > 7) ? (int8_t)(nibble - 16) : (int8_t)nibble;

        w_dq[idx] = __hmul(__float2bfloat16_rn((float)q), w_scale[r]);
    }
}

void launch_unpack_dequant_bf16(
    const uint8_t* packed,
    const __nv_bfloat16* w_scale,
    __nv_bfloat16* w_dq,
    int rows,
    int cols,
    cudaStream_t stream
) {
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    unpack_dequant_bf16_kernel<<<blocks, threads, 0, stream>>>(
        packed, w_scale, w_dq, rows, cols
    );
}
