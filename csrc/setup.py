from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="peqat_cuda",
    ext_modules=[
        CUDAExtension("peqat_cuda", [
            "bindings.cpp",
            "peqat_preprocess.cu",
        ]),
    ],
    cmdclass={"build_ext": BuildExtension},
)
