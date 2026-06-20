from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="peqat_cuda",
    ext_modules=[
        CUDAExtension("peqat_cuda", [
            "bindings.cpp",
            "peqat_preprocess.cu",
            "peqat_preprocess_per_token.cu",
            "peqat_gemm_bf16_wmma.cu",
        ]),
    ],
    cmdclass={"build_ext": BuildExtension},
)
