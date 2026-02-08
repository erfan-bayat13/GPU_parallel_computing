from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="transpose_ext",
    ext_modules=[
        CUDAExtension(
            name="transpose_ext",
            sources=["transpose.cpp", "transpose_cuda.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
