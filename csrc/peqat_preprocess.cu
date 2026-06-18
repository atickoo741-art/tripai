#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

__global__ void preprocess_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ smooth,
    const float* __restrict__ clip,
    int8_t* __restrict__ x_q,
    float* __restrict__ x_dq,
    float* __restrict__ x_scale,
    int N,
    int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;

    if (idx < total) {
        int n = idx / K;
        int k = idx % K;

        float x_val = x[idx];
        float s = smooth[k];
        float c = clip[k];

        float x_s = x_val / s;
        float x_clipped = fminf(fmaxf(x_s, -c), c);
        float s_x = c / 7.0f;

        float scaled = x_clipped / s_x;
        int q = __float2int_rn(scaled);
        q = min(max(q, -8), 7);

        x_q[idx] = (int8_t)q;
        x_dq[idx] = (float)q * s_x * s;

        if (n == 0) {
            x_scale[k] = s_x;
        }
    }
}

__global__ void preprocess_f16_kernel(
    const float* __restrict__ x,
    const float* __restrict__ smooth,
    const float* __restrict__ clip,
    int8_t* __restrict__ x_q,
    half* __restrict__ x_dq,
    float* __restrict__ x_scale,
    int N,
    int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;

    if (idx < total) {
        int n = idx / K;
        int k = idx % K;

        float x_val = x[idx];
        float s = smooth[k];
        float c = clip[k];

        float x_s = x_val / s;
        float x_clipped = fminf(fmaxf(x_s, -c), c);
        float s_x = c / 7.0f;

        float scaled = x_clipped / s_x;
        int q = __float2int_rn(scaled);
        q = min(max(q, -8), 7);

        x_q[idx] = (int8_t)q;
        x_dq[idx] = __float2half((float)q * s_x * s);

        if (n == 0) {
            x_scale[k] = s_x;
        }
    }
}

void launch_preprocess(
    const float* x,
    const float* smooth,
    const float* clip,
    int8_t* x_q,
    float* x_dq,
    float* x_scale,
    int N,
    int K,
    cudaStream_t stream
) {
    int total = N * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    preprocess_forward_kernel<<<blocks, threads, 0, stream>>>(
        x, smooth, clip, x_q, x_dq, x_scale, N, K
    );
}

void launch_preprocess_f16(
    const float* x,
    const float* smooth,
    const float* clip,
    int8_t* x_q,
    half* x_dq,
    float* x_scale,
    int N,
    int K,
    cudaStream_t stream
) {
    int total = N * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    preprocess_f16_kernel<<<blocks, threads, 0, stream>>>(
        x, smooth, clip, x_q, x_dq, x_scale, N, K
    );
}
