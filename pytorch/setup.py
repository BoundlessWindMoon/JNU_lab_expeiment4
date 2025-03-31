from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 基于 implicitGemm 的卷积实现(FP32)
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

# 基于 implicitGemm 的卷积实现(FP16)
setup(
    name='conv2d_cuda_fp16',
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension('conv2d_cuda_fp16', [
            'cpp/conv2d_cuda_fp16.cpp',
            'cuda/conv2d_cuda_kernel_fp16.cu',
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