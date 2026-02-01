// peqat_ops.h
#ifndef PEQAT_OPS_H
#define PEQAT_OPS_H

#include <cuda_runtime.h>
#include <stdint.h>

struct PEQATLayer {
    // Preprocessed weights
    uint8_t* d_weight_packed;
    float* d_weight_scales;
    
    // Quantization parameters
    float* d_smooth_factors;
    float* d_clip_thresholds;
    float* d_combined_scales;
    
    // PRE-ALLOCATED workspace (NEW!)
    float* d_workspace_preprocessed;
    int8_t* d_workspace_xq;
    int32_t* d_workspace_yint32;
    int max_batch_size;  // Maximum N supported
    
    int d;
    int k;
};

extern "C" {
    void launch_preprocess_activations(
        const float* X_in, float* X_out,
        const float* smooth, const float* clip,
        int N, int k, int blocks, int threads, cudaStream_t stream
    );
    
    void launch_quantize_activations(
        const float* X_clipped, int8_t* X_q,
        const float* clip, int N, int k,
        int blocks, int threads, cudaStream_t stream
    );
    
    void launch_int4_gemm(
        const int8_t* X_q, const uint8_t* W_packed,
        int32_t* Y_int32, int N, int d, int k,
        dim3 grid, dim3 block, cudaStream_t stream
    );
    
    void launch_dequantize_output(
        const int32_t* Y_int32, float* Y_fp32,
        const float* scales, int N, int d,
        int blocks, int threads, cudaStream_t stream
    );
}

PEQATLayer* load_peqat_layer(const char* filename, int max_batch_size = 256);
void peqat_forward(const float* d_X, float* d_Y, PEQATLayer* layer, int N, cudaStream_t stream = 0);
void free_peqat_layer(PEQATLayer* layer);

#endif
