from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="peqat_cuda",
    ext_modules=[
        CUDAExtension("peqat_cuda", [
            "bindings.cpp",
            "peqat_preprocess.cu",
            "peqat_preprocess_per_token.cu",
            "peqat_fused_int4_tc_sm121.cu",
            "peqat_fused_int4_gemv.cu",
            "peqat_gemm_bf16_wmma.cu",
            "depr/peqat_gemm.cu",
            "depr/peqat_gemm_f16.cu",
            "depr/peqat_fused_gemm.cu",
            "depr/peqat_fused_gemv.cu",
            "depr/peqat_fused_gemm_wmma.cu",
        ]),
    ],
    cmdclass={"build_ext": BuildExtension},
)
