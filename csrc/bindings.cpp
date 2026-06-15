#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

void launch_preprocess(
    const float* x, const float* smooth, const float* clip,
    int8_t* x_q, float* x_dq, float* x_scale,
    int N, int K, cudaStream_t stream
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("preprocess_forward", &preprocess_forward,
          "PE-QAT activation preprocessing: x/smooth -> clamp(-clip,clip) -> quantize INT4 -> dequantize");
}
