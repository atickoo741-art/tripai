// test_peqat.cpp
#include "peqat_ops.h"
#include <iostream>
#include <chrono>
#include <cmath>

void benchmark_peqat() {
    const char* weight_file = 
        "./INT4_output_qat_lora_old_clip_backward_co_4867_001/"
        "base_model_model_model_layers_0_self_attn_q_proj.peqat";
    
    std::cout << "=== PE-QAT INT4 Inference Benchmark ===" << std::endl;
    std::cout << "Loading: " << weight_file << std::endl;
    
    // Load layer
    auto* layer = load_peqat_layer(weight_file);
    if (!layer) return;
    
    const int N = 128;    // batch × seq_len
    int k = layer->k;
    int d = layer->d;
    
    std::cout << "\nLayer dimensions:" << std::endl;
    std::cout << "  Input channels: " << k << std::endl;
    std::cout << "  Output channels: " << d << std::endl;
    std::cout << "  Batch size: " << N << std::endl;
    
    // Allocate input/output
    float *h_X, *h_Y, *d_X, *d_Y;
    h_X = new float[N * k];
    h_Y = new float[N * d];
    
    // Random input
    for (int i = 0; i < N * k; i++) {
        h_X[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    cudaMalloc(&d_X, N * k * sizeof(float));
    cudaMalloc(&d_Y, N * d * sizeof(float));
    cudaMemcpy(d_X, h_X, N * k * sizeof(float), cudaMemcpyHostToDevice);
    
    // Warmup
    std::cout << "\nWarming up..." << std::endl;
    for (int i = 0; i < 10; i++) {
        peqat_forward(d_X, d_Y, layer, N);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    std::cout << "Benchmarking (100 iterations)..." << std::endl;
    const int iterations = 100;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        peqat_forward(d_X, d_Y, layer, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iterations;
    
    // Compute metrics
    float gflops = (2.0f * N * d * k) / (avg_ms * 1e6);
    float memory_gb = (N*k*1 + d*k*0.5 + N*d*4) / 1e9f;
    float bandwidth_gbs = memory_gb / (avg_ms / 1000.0f);
    
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "  Avg time: " << avg_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << gflops << " GFLOPS" << std::endl;
    std::cout << "  Bandwidth: " << bandwidth_gbs << " GB/s" << std::endl;
    
    // Copy result back
    cudaMemcpy(h_Y, d_Y, N * d * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "\n  Sample outputs: ";
    for (int i = 0; i < 5; i++) {
        std::cout << h_Y[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    delete[] h_X;
    delete[] h_Y;
    cudaFree(d_X);
    cudaFree(d_Y);
    free_peqat_layer(layer);
    
    std::cout << "\n✓ Test complete!" << std::endl;
}

int main() {
    benchmark_peqat();
    return 0;
}
