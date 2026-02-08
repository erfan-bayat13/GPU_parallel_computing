from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="transpose_ext",
    ext_modules=[
        CUDAExtension(
            "transpose_ext",
            sources=["transpose.cpp",
                     "kernels/transpose_naive.cu",
                     "kernels/transpose_tiled.cu"]
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
