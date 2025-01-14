from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="flaggems",
    ext_modules=[
        CUDAExtension(
            name="flaggems._C",
            sources=[
                "csrc/python_gems_funcs_everything.cc",
                "csrc/core/triton_kernel.cc",
                "csrc/kernels/add.cc",
                "csrc/gems_generated/shim.add_kernel.cc",
            ],
            extra_compile_args=[
                "-Wno-unused-function",
                "-std=c++17",
            ],
            extra_link_args=[
                "-lcuda",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
