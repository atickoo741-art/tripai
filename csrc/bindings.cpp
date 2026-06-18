#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

void launch_preprocess(
    const float* x, const float* smooth, const float* clip,
    int8_t* x_q, float* x_dq, float* x_scale,
    int N, int K, cudaStream_t stream
);

void launch_preprocess_f16(
    const float*, const float*, const float*,
    int8_t*, half*, float*,
    int, int, cudaStream_t
);

void launch_unpack_dequant(
    const uint8_t* packed, const float* w_scale, float* w_dq,
    int rows, int cols, cudaStream_t stream
);

void launch_unpack_dequant_f16(
    const uint8_t*, const half*, half*,
    int, int, cudaStream_t
);

void launch_unpack_dequant_bf16(
    const uint8_t*, const __nv_bfloat16*, __nv_bfloat16*,
    int, int, cudaStream_t
);

void launch_fused_gemv_decode(
    const half*, const uint8_t*, const half*, half*,
    int, int, cudaStream_t
);

void launch_fused_gemm_wmma(
    const half*, const uint8_t*, const half*, half*,
    int, int, int, cudaStream_t
);

void launch_fused_preprocess_gemm(
    const float* x,
    const uint8_t* packed,
    const float* w_scale,
    const float* smooth,
    const float* clip,
    float* output,
    int M, int N, int K,
    cudaStream_t stream
);

void launch_fused_preprocess_gemm_smallm(
    const float* x,
    const uint8_t* packed,
    const float* w_scale,
    const float* smooth,
    const float* clip,
    float* output,
    int M, int N, int K,
    cudaStream_t stream
);

std::vector<torch::Tensor> preprocess_forward(
    torch::Tensor x,
    torch::Tensor smooth,
    torch::Tensor clip
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(smooth.device().is_cuda(), "smooth must be CUDA tensor");
    TORCH_CHECK(clip.device().is_cuda(), "clip must be CUDA tensor");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [N, K]");
    TORCH_CHECK(smooth.dim() == 1, "smooth must be 1D [K]");
    TORCH_CHECK(clip.dim() == 1, "clip must be 1D [K]");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(smooth.scalar_type() == torch::kFloat32, "smooth must be float32");
    TORCH_CHECK(clip.scalar_type() == torch::kFloat32, "clip must be float32");

    int N = x.size(0);
    int K = x.size(1);

    TORCH_CHECK(smooth.size(0) == K, "smooth dim[0] != K");
    TORCH_CHECK(clip.size(0) == K, "clip dim[0] != K");

    auto x_q = torch::empty({N, K}, torch::dtype(torch::kInt8).device(x.device()));
    auto x_dq = torch::empty({N, K}, torch::dtype(torch::kFloat32).device(x.device()));
    auto x_scale = torch::empty({K}, torch::dtype(torch::kFloat32).device(x.device()));

    launch_preprocess(
        x.data_ptr<float>(),
        smooth.data_ptr<float>(),
        clip.data_ptr<float>(),
        x_q.data_ptr<int8_t>(),
        x_dq.data_ptr<float>(),
        x_scale.data_ptr<float>(),
        N, K,
        at::cuda::getCurrentCUDAStream()
    );

    return {x_q, x_dq, x_scale};
}

std::vector<torch::Tensor> preprocess_forward_f16(
    torch::Tensor x,
    torch::Tensor smooth,
    torch::Tensor clip
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(smooth.device().is_cuda(), "smooth must be CUDA tensor");
    TORCH_CHECK(clip.device().is_cuda(), "clip must be CUDA tensor");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [N, K]");
    TORCH_CHECK(smooth.dim() == 1, "smooth must be 1D [K]");
    TORCH_CHECK(clip.dim() == 1, "clip must be 1D [K]");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(smooth.scalar_type() == torch::kFloat32, "smooth must be float32");
    TORCH_CHECK(clip.scalar_type() == torch::kFloat32, "clip must be float32");

    int N = x.size(0);
    int K = x.size(1);

    TORCH_CHECK(smooth.size(0) == K, "smooth dim[0] != K");
    TORCH_CHECK(clip.size(0) == K, "clip dim[0] != K");

    auto x_q = torch::empty({N, K}, torch::dtype(torch::kInt8).device(x.device()));
    auto x_dq = torch::empty({N, K}, torch::dtype(torch::kFloat16).device(x.device()));
    auto x_scale = torch::empty({K}, torch::dtype(torch::kFloat32).device(x.device()));

    launch_preprocess_f16(
        x.data_ptr<float>(),
        smooth.data_ptr<float>(),
        clip.data_ptr<float>(),
        x_q.data_ptr<int8_t>(),
        reinterpret_cast<half*>(x_dq.data_ptr<at::Half>()),
        x_scale.data_ptr<float>(),
        N, K,
        at::cuda::getCurrentCUDAStream()
    );

    return {x_q, x_dq, x_scale};
}

torch::Tensor linear_forward(
    torch::Tensor x_dq,
    torch::Tensor packed_w,
    torch::Tensor w_scale,
    int64_t rows,
    int64_t cols
) {
    TORCH_CHECK(x_dq.device().is_cuda(), "x_dq must be CUDA tensor");
    TORCH_CHECK(packed_w.device().is_cuda(), "packed_w must be CUDA tensor");
    TORCH_CHECK(w_scale.device().is_cuda(), "w_scale must be CUDA tensor");

    TORCH_CHECK(x_dq.dim() == 2, "x_dq must be 2D [N, K]");
    TORCH_CHECK(x_dq.scalar_type() == torch::kFloat32, "x_dq must be float32");
    TORCH_CHECK(x_dq.size(1) == cols, "x_dq dim[1] != cols");

    TORCH_CHECK(w_scale.dim() == 1, "w_scale must be 1D [rows]");
    TORCH_CHECK(w_scale.scalar_type() == torch::kFloat16, "w_scale must be float16");
    TORCH_CHECK(w_scale.size(0) == rows, "w_scale dim[0] != rows");

    int N = x_dq.size(0);

    auto w_scale_f32 = w_scale.to(torch::kFloat32);
    auto w_dq = torch::empty({rows, cols}, torch::dtype(torch::kFloat32).device(x_dq.device()));

    launch_unpack_dequant(
        packed_w.data_ptr<uint8_t>(),
        w_scale_f32.data_ptr<float>(),
        w_dq.data_ptr<float>(),
        rows, cols,
        at::cuda::getCurrentCUDAStream()
    );

    return torch::matmul(x_dq, w_dq.t());
}

torch::Tensor linear_forward_f16(
    torch::Tensor x_f16,
    torch::Tensor packed_w,
    torch::Tensor w_scale,
    int64_t rows,
    int64_t cols
) {
    TORCH_CHECK(x_f16.device().is_cuda(), "x_f16 must be CUDA tensor");
    TORCH_CHECK(packed_w.device().is_cuda(), "packed_w must be CUDA tensor");
    TORCH_CHECK(w_scale.device().is_cuda(), "w_scale must be CUDA tensor");

    TORCH_CHECK(x_f16.dim() == 2, "x_f16 must be 2D [N, K]");
    TORCH_CHECK(x_f16.scalar_type() == torch::kFloat16, "x_f16 must be float16");
    TORCH_CHECK(x_f16.size(1) == cols, "x_f16 dim[1] != cols");

    TORCH_CHECK(w_scale.dim() == 1, "w_scale must be 1D [rows]");
    TORCH_CHECK(w_scale.scalar_type() == torch::kFloat16, "w_scale must be float16");
    TORCH_CHECK(w_scale.size(0) == rows, "w_scale dim[0] != rows");

    int N = x_f16.size(0);

    auto w_dq = torch::empty({rows, cols}, torch::dtype(torch::kFloat16).device(x_f16.device()));
    launch_unpack_dequant_f16(
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(w_scale.data_ptr<at::Half>()),
        reinterpret_cast<half*>(w_dq.data_ptr<at::Half>()),
        rows, cols,
        at::cuda::getCurrentCUDAStream()
    );

    return torch::matmul(x_f16, w_dq.t());
}

torch::Tensor fused_gemv_decode_forward(
    torch::Tensor x_f16,
    torch::Tensor packed_w,
    torch::Tensor w_scale,
    int64_t rows,
    int64_t cols
) {
    TORCH_CHECK(x_f16.device().is_cuda(), "x_f16 must be CUDA tensor");
    TORCH_CHECK(packed_w.device().is_cuda(), "packed_w must be CUDA tensor");
    TORCH_CHECK(w_scale.device().is_cuda(), "w_scale must be CUDA tensor");

    TORCH_CHECK(x_f16.dim() == 2, "x_f16 must be 2D [N, K]");
    TORCH_CHECK(x_f16.scalar_type() == torch::kFloat16, "x_f16 must be float16");
    TORCH_CHECK(x_f16.size(1) == cols, "x_f16 dim[1] != cols");

    TORCH_CHECK(w_scale.dim() == 1, "w_scale must be 1D [rows]");
    TORCH_CHECK(w_scale.scalar_type() == torch::kFloat16, "w_scale must be float16");
    TORCH_CHECK(w_scale.size(0) == rows, "w_scale dim[0] != rows");

    int N = x_f16.size(0);
    int K = cols;
    int R = rows;

    // Only supports decode (N=1) currently
    TORCH_CHECK(N == 1, "fused_gemv_decode only supports N=1, got N=" + std::to_string(N));

    auto output = torch::empty({1, R}, torch::dtype(torch::kFloat16).device(x_f16.device()));
    launch_fused_gemv_decode(
        reinterpret_cast<half*>(x_f16.data_ptr<at::Half>()),
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(w_scale.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        K, R,
        at::cuda::getCurrentCUDAStream()
    );

    return output;
}

torch::Tensor fused_gemm_wmma_forward(
    torch::Tensor x_f16,
    torch::Tensor packed_w,
    torch::Tensor w_scale,
    int64_t rows,
    int64_t cols
) {
    TORCH_CHECK(x_f16.device().is_cuda(), "x_f16 must be CUDA tensor");
    TORCH_CHECK(packed_w.device().is_cuda(), "packed_w must be CUDA tensor");
    TORCH_CHECK(w_scale.device().is_cuda(), "w_scale must be CUDA tensor");

    TORCH_CHECK(x_f16.dim() == 2, "x_f16 must be 2D [N, K]");
    TORCH_CHECK(x_f16.scalar_type() == torch::kFloat16, "x_f16 must be float16");
    TORCH_CHECK(x_f16.size(1) == cols, "x_f16 dim[1] != cols");

    TORCH_CHECK(w_scale.dim() == 1, "w_scale must be 1D [rows]");
    TORCH_CHECK(w_scale.scalar_type() == torch::kFloat16, "w_scale must be float16");
    TORCH_CHECK(w_scale.size(0) == rows, "w_scale dim[0] != rows");

    int M = x_f16.size(0);
    int K = cols;
    int N = rows;

    TORCH_CHECK(M > 1, "fused_gemm_wmma requires N>1, use fused_gemv_decode for N=1");

    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat16).device(x_f16.device()));
    launch_fused_gemm_wmma(
        reinterpret_cast<half*>(x_f16.data_ptr<at::Half>()),
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(w_scale.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );

    return output;
}

torch::Tensor linear_forward_bf16(
    torch::Tensor x_bf16,
    torch::Tensor packed_w,
    torch::Tensor w_scale,
    int64_t rows,
    int64_t cols
) {
    TORCH_CHECK(x_bf16.device().is_cuda(), "x_bf16 must be CUDA tensor");
    TORCH_CHECK(packed_w.device().is_cuda(), "packed_w must be CUDA tensor");
    TORCH_CHECK(w_scale.device().is_cuda(), "w_scale must be CUDA tensor");

    TORCH_CHECK(x_bf16.dim() == 2, "x_bf16 must be 2D [N, K]");
    TORCH_CHECK(x_bf16.scalar_type() == torch::kBFloat16, "x_bf16 must be bfloat16");
    TORCH_CHECK(x_bf16.size(1) == cols, "x_bf16 dim[1] != cols");

    TORCH_CHECK(w_scale.dim() == 1, "w_scale must be 1D [rows]");
    TORCH_CHECK(w_scale.scalar_type() == torch::kBFloat16, "w_scale must be bfloat16");
    TORCH_CHECK(w_scale.size(0) == rows, "w_scale dim[0] != rows");

    int N = x_bf16.size(0);

    auto w_dq = torch::empty({rows, cols}, torch::dtype(torch::kBFloat16).device(x_bf16.device()));
    launch_unpack_dequant_bf16(
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(w_scale.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(w_dq.data_ptr<at::BFloat16>()),
        rows, cols,
        at::cuda::getCurrentCUDAStream()
    );

    return torch::matmul(x_bf16, w_dq.t());
}

torch::Tensor fused_linear_forward(
    torch::Tensor x,
    torch::Tensor smooth,
    torch::Tensor clip,
    torch::Tensor packed_w,
    torch::Tensor w_scale,
    int64_t rows,
    int64_t cols
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(smooth.device().is_cuda(), "smooth must be CUDA tensor");
    TORCH_CHECK(clip.device().is_cuda(), "clip must be CUDA tensor");
    TORCH_CHECK(packed_w.device().is_cuda(), "packed_w must be CUDA tensor");
    TORCH_CHECK(w_scale.device().is_cuda(), "w_scale must be CUDA tensor");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K]");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.size(1) == cols, "x dim[1] != cols");

    TORCH_CHECK(smooth.dim() == 1, "smooth must be 1D [K]");
    TORCH_CHECK(smooth.scalar_type() == torch::kFloat32, "smooth must be float32");
    TORCH_CHECK(smooth.size(0) == cols, "smooth dim[0] != cols");

    TORCH_CHECK(clip.dim() == 1, "clip must be 1D [K]");
    TORCH_CHECK(clip.scalar_type() == torch::kFloat32, "clip must be float32");
    TORCH_CHECK(clip.size(0) == cols, "clip dim[0] != cols");

    TORCH_CHECK(w_scale.dim() == 1, "w_scale must be 1D [rows]");
    TORCH_CHECK(w_scale.scalar_type() == torch::kFloat16, "w_scale must be float16");
    TORCH_CHECK(w_scale.size(0) == rows, "w_scale dim[0] != rows");

    int M = x.size(0);
    int N = rows;
    int K = cols;

    auto w_scale_f32 = w_scale.to(torch::kFloat32);
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(x.device()));

    if (M <= 8) {
        launch_fused_preprocess_gemm_smallm(
            x.data_ptr<float>(),
            packed_w.data_ptr<uint8_t>(),
            w_scale_f32.data_ptr<float>(),
            smooth.data_ptr<float>(),
            clip.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K,
            at::cuda::getCurrentCUDAStream()
        );
    } else {
        launch_fused_preprocess_gemm(
            x.data_ptr<float>(),
            packed_w.data_ptr<uint8_t>(),
            w_scale_f32.data_ptr<float>(),
            smooth.data_ptr<float>(),
            clip.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K,
            at::cuda::getCurrentCUDAStream()
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("preprocess_forward", &preprocess_forward,
          "PE-QAT activation preprocessing: x/smooth -> clamp(-clip,clip) -> quantize INT4 -> dequantize");
    m.def("preprocess_forward_f16", &preprocess_forward_f16,
          "PE-QAT activation preprocessing with FP16 output");
    m.def("linear_forward", &linear_forward,
          "PE-QAT packed INT4 linear: unpack weights -> dequantize -> matmul");
    m.def("linear_forward_f16", &linear_forward_f16,
          "PE-QAT FP16 tensor core path: unpack -> FP16 dequant -> FP16 matmul");
    m.def("fused_gemv_decode_forward", &fused_gemv_decode_forward,
          "PE-QAT fused GEMV decode: INT4 weights loaded directly, unpack+dot in one kernel");
    m.def("linear_forward_bf16", &linear_forward_bf16,
          "PE-QAT bf16 tensor core path: unpack -> bf16 dequant -> bf16 matmul");
    m.def("fused_gemm_wmma_forward", &fused_gemm_wmma_forward,
          "PE-QAT fused GEMM prefill: INT4 weights -> unpack to FP16 shared -> wmma tensor core GEMM");
    m.def("fused_linear_forward", &fused_linear_forward,
          "PE-QAT fused INT4 linear with inline activation preprocessing: x/smooth -> clamp -> quant -> dequant -> tiled GEMM (one kernel)");
}
