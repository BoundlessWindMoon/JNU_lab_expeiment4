#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>
#include <cmath>

using namespace nvcuda;

__global__ void wmma_gemm(half *a, half *b, float *c, int M, int N, int K) {
    // 声明矩阵片段
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // 初始化累加器
    wmma::fill_fragment(c_frag, 0.0f);

    // 分块计算
    for (int ki = 0; ki < K; ki += 16) {
        int a_row = blockIdx.y * 16;  // 当前行块起始位置
        int b_col = blockIdx.x * 16;  // 当前列块起始位置
        
        wmma::load_matrix_sync(a_frag, a + a_row * K + ki, K);
        wmma::load_matrix_sync(b_frag, b + ki * N + b_col, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 存储结果
    wmma::store_matrix_sync(c + blockIdx.y * 16 * N + blockIdx.x * 16, c_frag, N, wmma::mem_row_major);
}

int main() {
    int M = 1024, N = 1024, K = 1024;
    
    // 主机端初始化
    half *h_a = new half[M*K];
    half *h_b = new half[K*N];
    float *h_c = new float[M*N];
    float *h_c_ref = new float[M*N];

    for (int i = 0; i < M*K; i++) h_a[i] = __float2half(1.0f);
    for (int i = 0; i < K*N; i++) h_b[i] = __float2half(1.0f);
    for (int i = 0; i < M*N; i++) h_c_ref[i] = K;

    // GPU内存分配
    half *d_a, *d_b;
    float *d_c;
    cudaMalloc(&d_a, M*K*sizeof(half));
    cudaMalloc(&d_b, K*N*sizeof(half));
    cudaMalloc(&d_c, M*N*sizeof(float));

    // 数据传输
    cudaMemcpy(d_a, h_a, M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K*N*sizeof(half), cudaMemcpyHostToDevice);

    // 执行核函数
    dim3 block(32);  // 1 Warp = 32 线程
    dim3 grid(M/16, N/16);
    wmma_gemm<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();

    // 结果回传
    cudaMemcpy(h_c, d_c, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    float max_error = 0.0f;
    for (int i = 0; i < M*N; i++) {
        float error = std::abs(h_c[i] - h_c_ref[i]);
        if (error > max_error) max_error = error;
    }

    std::cout << "最大绝对误差: " << max_error << std::endl;
    if (max_error < 1e-3f) {
        std::cout << "验证通过！" << std::endl;
    } else {
        std::cout << "验证失败！" << std::endl;
    }

    // 释放内存
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_c_ref;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
