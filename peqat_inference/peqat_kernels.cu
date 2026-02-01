// peqat_kernels.cu - PROPERLY OPTIMIZED WITH TILING
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// ============================================================================
// Step 2-3: Preprocess Activations (VECTORIZED)
// ============================================================================

__global__ void preprocess_activations_kernel(
    const float* __restrict__ X_in,
    float* __restrict__ X_smooth_clipped,
    const float* __restrict__ smooth_factors,
    const float* __restrict__ clip_thresholds,
    int N, int k
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;  // Process 4 elements per thread
    
    if (idx < N * k) {
        #pragma unroll
        for (int i = 0; i < 4 && idx + i < N * k; i++) {
            int c = (idx + i) % k;
            float val = X_in[idx + i] / smooth_factors[c];
            float alpha = clip_thresholds[c];
            val = fmaxf(fminf(val, alpha), -alpha);
            X_smooth_clipped[idx + i] = val;
        }
    }
}

// ============================================================================
// Step 5: Quantize Activations (VECTORIZED)
// ============================================================================

__global__ void quantize_activations_kernel(
    const float* __restrict__ X_clipped,
    int8_t* __restrict__ X_q,
    const float* __restrict__ clip_thresholds,
    int N, int k
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx < N * k) {
        #pragma unroll
        for (int i = 0; i < 4 && idx + i < N * k; i++) {
            int c = (idx + i) % k;
            float scale = clip_thresholds[c] / 7.0f;
            float val = X_clipped[idx + i];
            int q_val = __float2int_rn(val / scale);
            q_val = max(-8, min(7, q_val));
            X_q[idx + i] = (int8_t)q_val;
        }
    }
}

// ============================================================================
// OPTIMIZED INT4 GEMM - Tiled with Shared Memory
// Computes Y[N, d] = X_q[N, k] @ W_packed[d, k/2]^T
// ============================================================================

__global__ void peqat_int4_gemm_kernel(
    const int8_t* __restrict__ X_q,        // (N, k) INT8
    const uint8_t* __restrict__ W_packed,  // (d, k/2) packed INT4
    int32_t* __restrict__ Y_int32,         // (N, d) INT32
    int N, int d, int k
) {
    // Tile configuration
    const int TILE_M = 32;   // Rows of X (tokens)
    const int TILE_N = 32;   // Cols of W^T (output channels)
    const int TILE_K = 128;  // Reduction dimension
    
    // Shared memory for tiles
    __shared__ int8_t smem_X[TILE_M][TILE_K + 8];  // +8 for bank conflict avoidance
    __shared__ int8_t smem_W[TILE_N][TILE_K + 8];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread block computes TILE_M x TILE_N output tile
    int row_base = by * TILE_M;
    int col_base = bx * TILE_N;
    
    // Each thread accumulates for one output element
    int row = row_base + ty;
    int col = col_base + tx;
    
    int32_t accumulator = 0;
    
    // Loop over K dimension in tiles
    for (int k_tile = 0; k_tile < k; k_tile += TILE_K) {
        // **Collaborative loading of X tile** (32x32 threads load 32x128 data)
        // Each thread loads 4 elements
        #pragma unroll
        for (int load_iter = 0; load_iter < 4; load_iter++) {
            int load_col = tx + load_iter * 32;
            if (row < N && k_tile + load_col < k) {
                smem_X[ty][load_col] = X_q[row * k + k_tile + load_col];
            } else {
                smem_X[ty][load_col] = 0;
            }
        }
        
        // **Collaborative loading of W tile** (unpack INT4)
        // Load weight for column `col_base + ty`, unpack along K
        #pragma unroll
        for (int load_iter = 0; load_iter < 4; load_iter++) {
            int load_col = tx + load_iter * 32;
            int weight_col = col_base + ty;
            
            if (weight_col < d && k_tile + load_col < k) {
                int packed_idx = weight_col * (k / 2) + (k_tile + load_col) / 2;
                uint8_t packed = W_packed[packed_idx];
                
                // Unpack based on even/odd position
                if ((k_tile + load_col) % 2 == 0) {
                    smem_W[ty][load_col] = (int8_t)(packed << 4) >> 4;  // Lower 4 bits
                } else {
                    smem_W[ty][load_col] = (int8_t)packed >> 4;         // Upper 4 bits
                }
            } else {
                smem_W[ty][load_col] = 0;
            }
        }
        
        __syncthreads();
        
        // **Compute partial dot product** (each thread computes one element)
        if (row < N && col < d) {
            #pragma unroll
            for (int i = 0; i < TILE_K; i++) {
                accumulator += (int32_t)smem_X[ty][i] * (int32_t)smem_W[tx][i];
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < d) {
        Y_int32[row * d + col] = accumulator;
    }
}

// ============================================================================
// Dequantization (VECTORIZED)
// ============================================================================

__global__ void dequantize_output_optimized_kernel(
    const int32_t* __restrict__ Y_int32,
    float* __restrict__ Y_fp32,
    const float* __restrict__ combined_scales,
    int N, int d
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int total = N * d;
    
    if (idx < total) {
        #pragma unroll
        for (int i = 0; i < 4 && idx + i < total; i++) {
            int out_ch = (idx + i) % d;
            Y_fp32[idx + i] = (float)Y_int32[idx + i] * combined_scales[out_ch];
        }
    }
}

// ============================================================================
// C++ Wrapper Functions
// ============================================================================

extern "C" {

void launch_preprocess_activations(
    const float* X_in, float* X_out,
    const float* smooth, const float* clip,
    int N, int k, int blocks, int threads, cudaStream_t stream
) {
    // Adjust for vectorized processing (4 elements per thread)
    int vec_blocks = (N * k + threads * 4 - 1) / (threads * 4);
    preprocess_activations_kernel<<<vec_blocks, threads, 0, stream>>>(
        X_in, X_out, smooth, clip, N, k
    );
}

void launch_quantize_activations(
    const float* X_clipped, int8_t* X_q,
    const float* clip, int N, int k,
    int blocks, int threads, cudaStream_t stream
) {
    int vec_blocks = (N * k + threads * 4 - 1) / (threads * 4);
    quantize_activations_kernel<<<vec_blocks, threads, 0, stream>>>(
        X_clipped, X_q, clip, N, k
    );
}

void launch_int4_gemm(
    const int8_t* X_q, const uint8_t* W_packed,
    int32_t* Y_int32, int N, int d, int k,
    dim3 grid, dim3 block, cudaStream_t stream
) {
    peqat_int4_gemm_kernel<<<grid, block, 0, stream>>>(
        X_q, W_packed, Y_int32, N, d, k
    );
}

void launch_dequantize_output(
    const int32_t* Y_int32, float* Y_fp32,
    const float* scales, int N, int d,
    int blocks, int threads, cudaStream_t stream
) {
    int vec_blocks = (N * d + threads * 4 - 1) / (threads * 4);
    dequantize_output_optimized_kernel<<<vec_blocks, threads, 0, stream>>>(
        Y_int32, Y_fp32, scales, N, d
    );
}

} // extern "C"
