// peqat_launch.cu
#include "peqat_ops.h"
#include <cuda_runtime.h>

// Declare kernel signatures (from peqat_kernels.cu)
__global__ void preprocess_activations_kernel(
    const float* X_in, float* X_out,
    const float* smooth, const float* clip,
    int N, int k
);

__global__ void quantize_activations_kernel(
    const float* X_clipped, int8_t* X_q,
    const float* clip, int N, int k
);

__global__ void peqat_int4_gemm_kernel(
    const int8_t* X_q, const uint8_t* W_packed,
    int32_t* Y_int32, int N, int d, int k
);

__global__ void dequantize_output_optimized_kernel(
    const int32_t* Y_int32, float* Y_fp32,
    const float* scales, int N, int d
);

// Wrapper functions that launch kernels
extern "C" {

void launch_preprocess_activations(
    const float* d_X, float* d_X_preprocessed,
    const float* d_smooth, const float* d_clip,
    int N, int k, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (N * k + threads - 1) / threads;
    
    preprocess_activations_kernel<<<blocks, threads, 0, stream>>>(
        d_X, d_X_preprocessed, d_smooth, d_clip, N, k
    );
}

void launch_quantize_activations(
    const float* d_X_clipped, int8_t* d_X_q,
    const float* d_clip, int N, int k,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (N * k + threads - 1) / threads;
    
    quantize_activations_kernel<<<blocks, threads, 0, stream>>>(
        d_X_clipped, d_X_q, d_clip, N, k
    );
}

void launch_int4_gemm(
    const int8_t* d_X_q, const uint8_t* d_W_packed,
    int32_t* d_Y_int32, int N, int d, int k,
    cudaStream_t stream
) {
    dim3 block_gemm(16, 16);
    dim3 grid_gemm((d + 15) / 16, (N + 15) / 16);
    
    peqat_int4_gemm_kernel<<<grid_gemm, block_gemm, 0, stream>>>(
        d_X_q, d_W_packed, d_Y_int32, N, d, k
    );
}

void launch_dequantize_output(
    const int32_t* d_Y_int32, float* d_Y,
    const float* d_combined_scales,
    int N, int d, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (N * d + threads - 1) / threads;
    
    dequantize_output_optimized_kernel<<<blocks, threads, 0, stream>>>(
        d_Y_int32, d_Y, d_combined_scales, N, d
    );
}

} // extern "C"
