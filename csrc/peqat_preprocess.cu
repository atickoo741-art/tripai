#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

__global__ void preprocess_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ smooth,
    const __nv_bfloat16* __restrict__ clip,
    int8_t* __restrict__ x_q,
    __nv_bfloat16* __restrict__ x_dq,
    float* __restrict__ x_scale,
    int N,
    int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;

    if (idx < total) {
        int n = idx / K;
        int k = idx % K;

        __nv_bfloat16 x_val = x[idx];
        __nv_bfloat16 s = smooth[k];
        __nv_bfloat16 c = clip[k];

        __nv_bfloat16 x_s = __hdiv(x_val, s);

        __nv_bfloat16 neg_c = __float2bfloat16_rn(-__bfloat162float(c));
        __nv_bfloat16 x_clipped = __hmax(__hmin(x_s, c), neg_c);

        __nv_bfloat16 s_x = __hdiv(c, __float2bfloat16_rn(7.0f));

        __nv_bfloat16 scaled = __hdiv(x_clipped, s_x);

        float scaled_f = __bfloat162float(scaled);
        int q = __float2int_rn(scaled_f);
        q = min(max(q, -8), 7);

        x_q[idx] = (int8_t)q;
        x_dq[idx] = __hmul(__float2bfloat16_rn((float)q), s_x);

        if (n == 0) {
            x_scale[k] = __bfloat162float(s_x);
        }
    }
}

void launch_preprocess_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* smooth,
    const __nv_bfloat16* clip,
    int8_t* x_q,
    __nv_bfloat16* x_dq,
    float* x_scale,
    int N,
    int K,
    cudaStream_t stream
) {
    int total = N * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    preprocess_bf16_kernel<<<blocks, threads, 0, stream>>>(
        x, smooth, clip, x_q, x_dq, x_scale, N, K
    );
}
