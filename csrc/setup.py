from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="peqat_cuda",
    ext_modules=[
        CUDAExtension("peqat_cuda",
            sources=[
            "bindings.cpp",
            "peqat_preprocess.cu",
            "peqat_preprocess_per_token.cu",
            "peqat_hadamard_preprocess.cu",
            "hadamard_butterfly.cu",
            "peqat_fused_int4_tc_sm121.cu",
            "peqat_fused_int4_gemv.cu",
            "peqat_gemm_bf16_wmma.cu",
        ],
            extra_compile_args={"nvcc": ["-O3", "--ptxas-options=-v"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
