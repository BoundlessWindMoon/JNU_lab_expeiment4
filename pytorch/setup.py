from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 基于 implicitGemm 的卷积实现
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

# 基于 直接卷积 的卷积实现
setup(
    name='conv2d_baseline_cuda',
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension('conv2d_baseline_cuda', [
            'cpp/conv2d_baseline_cuda.cpp',
            'cuda/conv2d_baseline_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })