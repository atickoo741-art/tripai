#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

void launch_preprocess(
    const float* x, const float* smooth, const float* clip,
    int8_t* x_q, float* x_dq, float* x_scale,
    int N, int K, cudaStream_t stream
);

void launch_unpack_dequant(
    const uint8_t* packed, const float* w_scale, float* w_dq,
    int rows, int cols, cudaStream_t stream
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("preprocess_forward", &preprocess_forward,
          "PE-QAT activation preprocessing: x/smooth -> clamp(-clip,clip) -> quantize INT4 -> dequantize");
    m.def("linear_forward", &linear_forward,
          "PE-QAT packed INT4 linear: unpack weights -> dequantize -> matmul");
}
