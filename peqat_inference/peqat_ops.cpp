// peqat_ops.cpp
#include "peqat_ops.h"
#include <fstream>
#include <iostream>
#include <cstring>

PEQATLayer* load_peqat_layer(const char* filename, int max_batch_size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return nullptr;
    }
    
    auto* layer = new PEQATLayer();
    
    // Read header
    uint32_t d, k, rank;
    file.read((char*)&d, sizeof(uint32_t));
    file.read((char*)&k, sizeof(uint32_t));
    file.read((char*)&rank, sizeof(uint32_t));
    
    layer->d = d;
    layer->k = k;
    layer->max_batch_size = max_batch_size;
    
    std::cout << "Loading layer: d=" << d << ", k=" << k << std::endl;
    
    // Allocate and load packed weights
    size_t weight_packed_size = d * (k / 2);
    auto* h_weight_packed = new uint8_t[weight_packed_size];
    file.read((char*)h_weight_packed, weight_packed_size);
    
    cudaMalloc(&layer->d_weight_packed, weight_packed_size);
    cudaMemcpy(layer->d_weight_packed, h_weight_packed, 
               weight_packed_size, cudaMemcpyHostToDevice);
    delete[] h_weight_packed;
    
    // Allocate and load weight scales
    auto* h_weight_scales = new float[d];
    file.read((char*)h_weight_scales, d * sizeof(float));
    
    cudaMalloc(&layer->d_weight_scales, d * sizeof(float));
    cudaMemcpy(layer->d_weight_scales, h_weight_scales, 
               d * sizeof(float), cudaMemcpyHostToDevice);
    
    // Load smoothing factors
    auto* h_smooth = new float[k];
    file.read((char*)h_smooth, k * sizeof(float));
    
    cudaMalloc(&layer->d_smooth_factors, k * sizeof(float));
    cudaMemcpy(layer->d_smooth_factors, h_smooth, 
               k * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_smooth;
    
    // Load clip thresholds
    auto* h_clip = new float[k];
    file.read((char*)h_clip, k * sizeof(float));
    
    cudaMalloc(&layer->d_clip_thresholds, k * sizeof(float));
    cudaMemcpy(layer->d_clip_thresholds, h_clip, 
               k * sizeof(float), cudaMemcpyHostToDevice);
    
    // Precompute combined scales
    auto* h_combined_scales = new float[d];
    float avg_act_scale = 0.0f;
    for (int c = 0; c < k; c++) {
        avg_act_scale += h_clip[c] / 7.0f;
    }
    avg_act_scale /= k;
    
    for (int j = 0; j < d; j++) {
        h_combined_scales[j] = h_weight_scales[j] * avg_act_scale;
    }
    
    cudaMalloc(&layer->d_combined_scales, d * sizeof(float));
    cudaMemcpy(layer->d_combined_scales, h_combined_scales, 
               d * sizeof(float), cudaMemcpyHostToDevice);
    
    delete[] h_weight_scales;
    delete[] h_clip;
    delete[] h_combined_scales;
    
    // PRE-ALLOCATE workspace buffers (NEW!)
    std::cout << "Allocating workspace for max batch size: " << max_batch_size << std::endl;
    
    cudaMalloc(&layer->d_workspace_preprocessed, max_batch_size * k * sizeof(float));
    cudaMalloc(&layer->d_workspace_xq, max_batch_size * k * sizeof(int8_t));
    cudaMalloc(&layer->d_workspace_yint32, max_batch_size * d * sizeof(int32_t));
    
    size_t workspace_mb = (max_batch_size * k * sizeof(float) + 
                          max_batch_size * k * sizeof(int8_t) + 
                          max_batch_size * d * sizeof(int32_t)) / (1024 * 1024);
    std::cout << "Workspace allocated: " << workspace_mb << " MB" << std::endl;
    
    std::cout << "✓ Layer loaded successfully" << std::endl;
    return layer;
}

void peqat_forward(
    const float* d_X,
    float* d_Y,
    PEQATLayer* layer,
    int N,
    cudaStream_t stream
) {
    if (N > layer->max_batch_size) {
        std::cerr << "ERROR: Batch size " << N << " exceeds max " 
                  << layer->max_batch_size << std::endl;
        return;
    }
    
    int k = layer->k;
    int d = layer->d;
    
    // Use pre-allocated workspace (NO malloc/free overhead!)
    float* d_X_preprocessed = layer->d_workspace_preprocessed;
    int8_t* d_X_q = layer->d_workspace_xq;
    int32_t* d_Y_int32 = layer->d_workspace_yint32;
    
    // Step 2-3: Preprocess activations
    int threads = 256;
    int blocks = (N * k + threads - 1) / threads;
    
    launch_preprocess_activations(
        d_X, d_X_preprocessed,
        layer->d_smooth_factors,
        layer->d_clip_thresholds,
        N, k, blocks, threads, stream
    );
    
    // Step 5: Quantize activations
    launch_quantize_activations(
        d_X_preprocessed, d_X_q,
        layer->d_clip_thresholds,
        N, k, blocks, threads, stream
    );
    
    // Step 6: INT4 GEMM
    // dim3 block_gemm(16, 16);
    // dim3 grid_gemm((d + 15) / 16, (N + 15) / 16);
    
    const int TILE_M = 32;
    const int TILE_N = 32;
    dim3 block_gemm(TILE_N, TILE_M);  // 32x32 = 1024 threads
    dim3 grid_gemm((d + TILE_N - 1) / TILE_N, (N + TILE_M - 1) / TILE_M);

    launch_int4_gemm(
        d_X_q, layer->d_weight_packed,
        d_Y_int32, N, d, k,
        grid_gemm, block_gemm, stream
    );
    
    // Dequantize
    blocks = (N * d + threads - 1) / threads;
    launch_dequantize_output(
        d_Y_int32, d_Y,
        layer->d_combined_scales,
        N, d, blocks, threads, stream
    );
}

void free_peqat_layer(PEQATLayer* layer) {
    cudaFree(layer->d_weight_packed);
    cudaFree(layer->d_weight_scales);
    cudaFree(layer->d_smooth_factors);
    cudaFree(layer->d_clip_thresholds);
    cudaFree(layer->d_combined_scales);
    
    // Free workspace
    cudaFree(layer->d_workspace_preprocessed);
    cudaFree(layer->d_workspace_xq);
    cudaFree(layer->d_workspace_yint32);
    
    delete layer;
}
