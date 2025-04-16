from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 基于 implicitGemm 的卷积实现
setup(
    name='conv2d_optim_fp16',
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension('conv2d_optim_fp16', [
            'cpp/conv2d_optim_fp16.cpp',
            'cuda/conv2d_optim_kernel_fp16.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

# 基于 直接卷积 的卷积实现
setup(
    name='conv2d_baseline_fp32',
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension('conv2d_baseline_fp32', [
            'cpp/conv2d_baseline_fp32.cpp',
            'cuda/conv2d_baseline_kernel_fp32.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })