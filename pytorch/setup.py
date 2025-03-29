from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='conv2d_cuda',
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension('conv2d_cuda', [
            'cpp/conv2d_cuda.cpp',
            'cuda/conv2d_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })