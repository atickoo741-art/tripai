#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

void launch_preprocess_bf16(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    int8_t*, __nv_bfloat16*, float*,
    int, int, cudaStream_t
);

void launch_fused_gemv_bf16(
    const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, __nv_bfloat16*,
    int, int, cudaStream_t
);

void launch_fused_gemm_bf16_wmma(
    const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, cudaStream_t
);

void launch_preprocess_per_token(
    const __nv_bfloat16* x,
    const __nv_bfloat16* smooth,
    const __nv_bfloat16* clip,
    int32_t* x_packed,
    float* s_x,
    int M, int K,
    cudaStream_t stream
);

void launch_repack_4_to_8(
    const int32_t* src,
    int32_t* dst,
    int rows, int K,
    cudaStream_t stream
);

void launch_preprocess_hadamard(
    const __nv_bfloat16* x,
    int32_t* x_packed,
    float* s_x,
    int M, int K,
    cudaStream_t stream
);

void launch_hadamard_butterfly(
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    int M, int K,
    cudaStream_t stream
);

void launch_fused_int4_tc_per_token(
    const int32_t* x_packed,
    const int32_t* w_packed,
    const float* w_scale,
    const float* s_x,
    __nv_bfloat16* y,
    int M, int N, int K,
    cudaStream_t stream
);

void launch_fused_int4_gemv(
    const int32_t* x_packed,
    const int32_t* w_packed,
    const float* w_scale,
    const float* s_x,
    __nv_bfloat16* y,
    int N, int K,
    cudaStream_t stream
);

std::vector<torch::Tensor> preprocess_forward_bf16(
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
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(smooth.scalar_type() == torch::kBFloat16, "smooth must be bfloat16");
    TORCH_CHECK(clip.scalar_type() == torch::kBFloat16, "clip must be bfloat16");

    int N = x.size(0);
    int K = x.size(1);

    TORCH_CHECK(smooth.size(0) == K, "smooth dim[0] != K");
    TORCH_CHECK(clip.size(0) == K, "clip dim[0] != K");

    auto x_q = torch::empty({N, K}, torch::dtype(torch::kInt8).device(x.device()));
    auto x_dq = torch::empty({N, K}, torch::dtype(torch::kBFloat16).device(x.device()));
    auto x_scale = torch::empty({K}, torch::dtype(torch::kFloat32).device(x.device()));

    launch_preprocess_bf16(
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(smooth.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(clip.data_ptr<at::BFloat16>()),
        x_q.data_ptr<int8_t>(),
        reinterpret_cast<__nv_bfloat16*>(x_dq.data_ptr<at::BFloat16>()),
        x_scale.data_ptr<float>(),
        N, K,
        at::cuda::getCurrentCUDAStream()
    );

    return {x_q, x_dq, x_scale};
}

// --- Per-token preprocess (for INT4 Tensor Core path) ---

std::vector<torch::Tensor> preprocess_per_token_forward(
    torch::Tensor x,
    torch::Tensor smooth,
    torch::Tensor clip
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(smooth.device().is_cuda(), "smooth must be CUDA tensor");
    TORCH_CHECK(clip.device().is_cuda(), "clip must be CUDA tensor");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K]");
    TORCH_CHECK(smooth.dim() == 1, "smooth must be 1D [K]");
    TORCH_CHECK(clip.dim() == 1, "clip must be 1D [K]");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(smooth.scalar_type() == torch::kBFloat16, "smooth must be bfloat16");
    TORCH_CHECK(clip.scalar_type() == torch::kBFloat16, "clip must be bfloat16");

    int M = x.size(0);
    int K = x.size(1);

    TORCH_CHECK(smooth.size(0) == K, "smooth dim[0] != K");
    TORCH_CHECK(clip.size(0) == K, "clip dim[0] != K");
    TORCH_CHECK(K % 8 == 0, "K must be multiple of 8 for INT4 packing");

    int n_packed = K / 8;
    auto x_packed = torch::empty({M, n_packed}, torch::dtype(torch::kInt32).device(x.device()));
    auto s_x_out = torch::empty({M}, torch::dtype(torch::kFloat32).device(x.device()));

    launch_preprocess_per_token(
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(smooth.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(clip.data_ptr<at::BFloat16>()),
        x_packed.data_ptr<int32_t>(),
        s_x_out.data_ptr<float>(),
        M, K,
        at::cuda::getCurrentCUDAStream()
    );

    return {x_packed, s_x_out};
}

torch::Tensor fused_gemv_bf16_forward(
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
    int K = cols;
    int R = rows;

    TORCH_CHECK(N == 1, "fused_gemv_bf16 only supports N=1, got N=" + std::to_string(N));

    auto output = torch::empty({1, R}, torch::dtype(torch::kBFloat16).device(x_bf16.device()));
    launch_fused_gemv_bf16(
        reinterpret_cast<__nv_bfloat16*>(x_bf16.data_ptr<at::BFloat16>()),
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(w_scale.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        K, R,
        at::cuda::getCurrentCUDAStream()
    );

    return output;
}

torch::Tensor fused_gemm_bf16_wmma_forward(
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

    int M = x_bf16.size(0);
    int K = cols;
    int N = rows;

    TORCH_CHECK(M > 1, "fused_gemm_bf16_wmma requires M>1, use fused_gemv_bf16 for M=1");

    auto output = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(x_bf16.device()));
    launch_fused_gemm_bf16_wmma(
        reinterpret_cast<__nv_bfloat16*>(x_bf16.data_ptr<at::BFloat16>()),
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(w_scale.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );

    return output;
}

// --- INT4 Tensor Core (SM 12.1) ---

torch::Tensor repack_4_to_8_forward(
    torch::Tensor x_packed
) {
    TORCH_CHECK(x_packed.device().is_cuda(), "x_packed must be CUDA tensor");
    TORCH_CHECK(x_packed.dim() == 2, "x_packed must be 2D [M, K/4]");
    TORCH_CHECK(x_packed.scalar_type() == torch::kInt32, "x_packed must be int32");

    int M = x_packed.size(0);
    int K4 = x_packed.size(1);
    int K = K4 * 4;

    auto x_8 = torch::empty({M, K / 8}, torch::dtype(torch::kInt32).device(x_packed.device()));

    launch_repack_4_to_8(
        x_packed.data_ptr<int32_t>(),
        x_8.data_ptr<int32_t>(),
        M, K,
        at::cuda::getCurrentCUDAStream()
    );

    return x_8;
}

torch::Tensor fused_int4_tc_per_token_forward(
    torch::Tensor x_packed_8,
    torch::Tensor w_packed_tc,
    torch::Tensor w_scale,
    torch::Tensor s_x,
    int64_t out_features,
    int64_t in_features
) {
    TORCH_CHECK(x_packed_8.device().is_cuda(), "x_packed_8 must be CUDA tensor");
    TORCH_CHECK(w_packed_tc.device().is_cuda(), "w_packed_tc must be CUDA tensor");
    TORCH_CHECK(w_scale.device().is_cuda(), "w_scale must be CUDA tensor");
    TORCH_CHECK(s_x.device().is_cuda(), "s_x must be CUDA tensor");

    TORCH_CHECK(x_packed_8.dim() == 2, "x_packed_8 must be 2D [M, K/8]");
    TORCH_CHECK(x_packed_8.scalar_type() == torch::kInt32, "x_packed_8 must be int32");

    TORCH_CHECK(w_packed_tc.dim() == 2, "w_packed_tc must be 2D [N, K/8]");
    TORCH_CHECK(w_packed_tc.scalar_type() == torch::kInt32, "w_packed_tc must be int32");

    TORCH_CHECK(w_scale.dim() == 1, "w_scale must be 1D [N]");
    TORCH_CHECK(w_scale.scalar_type() == torch::kFloat32, "w_scale must be float32");

    TORCH_CHECK(s_x.dim() == 1, "s_x must be 1D [M]");
    TORCH_CHECK(s_x.scalar_type() == torch::kFloat32, "s_x must be float32");

    int M = x_packed_8.size(0);
    int N = out_features;
    int K = in_features;

    TORCH_CHECK(w_packed_tc.size(0) == N, "w_packed_tc dim[0] != out_features");
    TORCH_CHECK(w_packed_tc.size(1) == K / 8, "w_packed_tc dim[1] != K/8");
    TORCH_CHECK(w_scale.size(0) == N, "w_scale dim[0] != N");
    TORCH_CHECK(s_x.size(0) == M, "s_x dim[0] != M");
    TORCH_CHECK(K % 64 == 0, "K must be multiple of 64 for INT4 TC");

    auto y = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(x_packed_8.device()));

    launch_fused_int4_tc_per_token(
        x_packed_8.data_ptr<int32_t>(),
        w_packed_tc.data_ptr<int32_t>(),
        w_scale.data_ptr<float>(),
        s_x.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );

    return y;
}

torch::Tensor fused_int4_gemv_per_token_forward(
    torch::Tensor x_packed_8,
    torch::Tensor w_packed_tc,
    torch::Tensor w_scale,
    torch::Tensor s_x,
    int64_t out_features,
    int64_t in_features
) {
    TORCH_CHECK(x_packed_8.device().is_cuda(), "x_packed_8 must be CUDA tensor");
    TORCH_CHECK(w_packed_tc.device().is_cuda(), "w_packed_tc must be CUDA tensor");
    TORCH_CHECK(w_scale.device().is_cuda(), "w_scale must be CUDA tensor");
    TORCH_CHECK(s_x.device().is_cuda(), "s_x must be CUDA tensor");

    TORCH_CHECK(x_packed_8.dim() == 2, "x_packed_8 must be 2D [1, K/8]");
    TORCH_CHECK(x_packed_8.scalar_type() == torch::kInt32, "x_packed_8 must be int32");
    TORCH_CHECK(x_packed_8.size(0) == 1, "fused_int4_gemv only supports M=1");

    TORCH_CHECK(w_packed_tc.dim() == 2, "w_packed_tc must be 2D [N, K/8]");
    TORCH_CHECK(w_packed_tc.scalar_type() == torch::kInt32, "w_packed_tc must be int32");

    TORCH_CHECK(w_scale.dim() == 1, "w_scale must be 1D [N]");
    TORCH_CHECK(w_scale.scalar_type() == torch::kFloat32, "w_scale must be float32");

    TORCH_CHECK(s_x.dim() == 1, "s_x must be 1D [1]");
    TORCH_CHECK(s_x.scalar_type() == torch::kFloat32, "s_x must be float32");

    int N = out_features;
    int K = in_features;

    TORCH_CHECK(w_packed_tc.size(0) == N, "w_packed_tc dim[0] != out_features");
    TORCH_CHECK(w_packed_tc.size(1) == K / 8, "w_packed_tc dim[1] != K/8");
    TORCH_CHECK(w_scale.size(0) == N, "w_scale dim[0] != N");
    TORCH_CHECK(s_x.size(0) == 1, "s_x dim[0] != 1");
    TORCH_CHECK(K % 8 == 0, "K must be multiple of 8");

    auto y = torch::empty({1, N}, torch::dtype(torch::kBFloat16).device(x_packed_8.device()));

    launch_fused_int4_gemv(
        x_packed_8.data_ptr<int32_t>(),
        w_packed_tc.data_ptr<int32_t>(),
        w_scale.data_ptr<float>(),
        s_x.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        N, K,
        at::cuda::getCurrentCUDAStream()
    );

    return y;
}

// --- Hadamard preprocess (no smooth/clip params) ---

std::vector<torch::Tensor> preprocess_hadamard_forward(
    torch::Tensor x
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K]");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");

    int M = x.size(0);
    int K = x.size(1);
    TORCH_CHECK(K % 8 == 0, "K must be multiple of 8 for INT4 packing");

    int n_packed = K / 8;
    auto x_packed = torch::empty({M, n_packed}, torch::dtype(torch::kInt32).device(x.device()));
    auto s_x_out = torch::empty({M}, torch::dtype(torch::kFloat32).device(x.device()));

    launch_preprocess_hadamard(
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        x_packed.data_ptr<int32_t>(),
        s_x_out.data_ptr<float>(),
        M, K,
        at::cuda::getCurrentCUDAStream()
    );

    return {x_packed, s_x_out};
}

torch::Tensor hadamard_butterfly_forward(
    torch::Tensor x
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K]");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");

    int M = x.size(0);
    int K = x.size(1);
    TORCH_CHECK(K > 0 && (K & (K - 1)) == 0, "K must be a power of 2, got ", K);

    auto y = torch::empty_like(x);

    launch_hadamard_butterfly(
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        M, K,
        at::cuda::getCurrentCUDAStream()
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("preprocess_forward_bf16", &preprocess_forward_bf16,
          "PE-QAT activation preprocessing with bf16 arithmetic and bf16 output");
    m.def("fused_gemv_bf16_forward", &fused_gemv_bf16_forward,
          "PE-QAT fused GEMV decode (bf16): INT4 weights loaded directly, unpack+dot in one kernel");
    m.def("fused_gemm_bf16_wmma_forward", &fused_gemm_bf16_wmma_forward,
          "PE-QAT fused GEMM prefill (bf16 WMMA): INT4 weights -> unpack to bf16 shared -> wmma bf16 tensor core GEMM");
    m.def("preprocess_per_token_forward", &preprocess_per_token_forward,
          "Per-token preprocess for INT4 TC: x/smooth -> clamp(-clip,clip) -> per-token max/7 -> pack 4xINT4 per int32");
    m.def("repack_4_to_8_forward", &repack_4_to_8_forward,
          "Repack INT4 from 4-per-int32 (gappy) to 8-per-int32 (dense) for MMA");
    m.def("fused_int4_tc_per_token_forward", &fused_int4_tc_per_token_forward,
          "SM 12.1 INT4 Tensor Core GEMM: per-token quantized INT4 x @ INT4 w -> bf16");
    m.def("fused_int4_gemv_per_token_forward", &fused_int4_gemv_per_token_forward,
          "SM 12.1 INT4 GEMV (M=1): per-token quantized INT4 x @ INT4 w -> bf16, no padding waste");
    m.def("preprocess_hadamard_forward", &preprocess_hadamard_forward,
          "Hadamard PE-QAT preprocess: H(x) -> per-token max/7 -> pack 8xINT4 per int32. No smooth/clip.");
    m.def("hadamard_butterfly_forward", &hadamard_butterfly_forward,
          "Fused Walsh-Hadamard butterfly (training): H(x) normalized by sqrt(K). Single kernel, bf16 in/out. K must be power of 2.");
}
