#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <cstdlib>
#define PLACEHOLDER 0
using namespace nvcuda;

const int WARP_SIZE = 32;
const int BLOCK_ROW_WARPS = 2;
const int BLOCK_COL_WARPS = 2;
const int WARP_ROW_TILES = 2;
const int WARP_COL_TILES = 2;
const int BLOCK_ROW_TILES = (WARP_ROW_TILES * BLOCK_ROW_WARPS);
const int BLOCK_COL_TILES = (WARP_COL_TILES * BLOCK_COL_WARPS);

__global__ void wmma_gemm(half *a, half *b, float *c, int M, int N, int K) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int ki = 0; ki < K; ki += 16) {
        int a_row = blockIdx.y * 16;  
        int b_col = blockIdx.x * 16;  
        __half* addr_a = a + PLACEHOLDER;
        __half* addr_b = b + PLACEHOLDER;
        wmma::load_matrix_sync(a_frag, addr_a, K);
        wmma::load_matrix_sync(b_frag, addr_b, K); 
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 存储结果
    wmma::store_matrix_sync(c + blockIdx.y * 16 * N + blockIdx.x * 16, c_frag, N, wmma::mem_row_major);
}


__global__ void basic_gemm(half *a, half *b, float *c, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += (float)__hmul(a[row * K + k] , b[col * K + k]);
        }
        c[row * N + col] = sum;
    }
}

void run_test(int M, int N, int K) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.1f, 1.0f);

    half *h_a = new half[M * K];
    half *h_b = new half[K * N];
    float *h_c_wmma = new float[M * N];
    float *h_c_basic = new float[M * N];
    float *h_c_ref = new float[M * N];

    for (int i = 0; i < M * K; ++i)
        h_a[i] = __float2half(dist(gen));
    for (int i = 0; i < K * N; ++i)
        h_b[i] = __float2half(dist(gen));

    half *d_a, *d_b;
    float *d_c_wmma, *d_c_basic;
    cudaMalloc(&d_a, M * K * sizeof(half));
    cudaMalloc(&d_b, K * N * sizeof(half));
    cudaMalloc(&d_c_wmma, M * N * sizeof(float));
    cudaMalloc(&d_c_basic, M * N * sizeof(float));

    cudaMemcpy(d_a, h_a, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_c_wmma, 0, M * N * sizeof(float));
    cudaMemset(d_c_basic, 0, M * N * sizeof(float));

    dim3 grid_wmma((N + 15)/16, (M + 15)/16);
    dim3 block_wmma(32);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    wmma_gemm<<<grid_wmma, block_wmma>>>(d_a, d_b, d_c_wmma, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_wmma;
    cudaEventElapsedTime(&time_wmma, start, stop);

    dim3 block_basic(16, 16);
    dim3 grid_basic((N + block_basic.x - 1) / block_basic.x,
                    (M + block_basic.y - 1) / block_basic.y);
    
    cudaEventRecord(start);
    basic_gemm<<<grid_basic, block_basic>>>(d_a, d_b, d_c_basic, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_basic;
    cudaEventElapsedTime(&time_basic, start, stop);

    cudaMemcpy(h_c_wmma, d_c_wmma, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_basic, d_c_basic, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    const float relative_tolerance = 1e-3f;
    const float absolute_tolerance = 1e-2f;
    bool wmma_valid = true;
    int error_count = 0;

    for (int i = 0; i < M * N; ++i) {
        float wmma_val = h_c_wmma[i];
        float basic_val = h_c_basic[i];
        
        float diff = fabs(wmma_val - basic_val);
        float rel_err = diff / fmaxf(1.0f, fabs(basic_val));
        if (diff > absolute_tolerance && rel_err > relative_tolerance) {
            if (error_count < 5) {
                std::cout << "WMMA错误 [" << i/N << "," << i%N << "] "
                          << "参考值: " << basic_val << " 实际值: " << wmma_val 
                          << " 相对误差: " << rel_err*100 << "%" << std::endl;
            }
            wmma_valid = false;
            error_count++;
        }
    }

    std::cout << "\n矩阵尺寸 " << M << "x" << N << "x" << K << " 测试结果:\n"
              << "Tensor Core验证: " << (wmma_valid ? "通过" : "失败") << "\n"
              << "执行时间: WMMA=" << time_wmma << "ms Basic=" << time_basic << "ms\n"
              << "加速比: " << time_basic/time_wmma << "x\n"
              << "总错误数: " << error_count << "/" << M*N << std::endl;

    delete[] h_a;
    delete[] h_b;
    delete[] h_c_wmma;
    delete[] h_c_basic;
    delete[] h_c_ref;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_wmma);
    cudaFree(d_c_basic);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048};
    
    for (int size : sizes) {
        if (size % 16 != 0) {
            std::cerr << "跳过非16倍数的尺寸: " << size << std::endl;
            continue;
        }
        try {
            std::cout << "\n======== 开始测试 " << size << " ========" << std::endl;
            run_test(size, size, size);
        } catch (const std::bad_alloc& e) {
            std::cerr << "内存分配失败: " << e.what() << std::endl;
            break;
        }
    }
    
    return 0;
}
