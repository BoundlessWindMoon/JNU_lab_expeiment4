#include <torch/extension.h>
#include <cuda_fp16.h>
#include "conv2d_fp16.h"

// FP16版本的implgemm卷积
__global__ void implgemm_fp16(param_t param)
{
    uint32_t tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Warp tile配置和原来相同
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t mma_tid_x = (lane_id / 2) % 8;
    const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    
    // lds addr计算不变
    uint32_t weight_lds_addr = (warp_id / 2) * 32 + mma_tid_y * 4;
    uint32_t input_lds_addr = (warp_id % 2) * 64 + mma_tid_x * 4;

    int x = bx * 128 + input_lds_addr;
    int y = by * 128 + weight_lds_addr;
    int z = blockIdx.z;

    // 修改共享内存为half2类型提高带宽利用率
    __shared__ half smeminput[8 * 128 * 2];
    __shared__ half smemweight[8 * 132 * 2];

    half2 weight_ldg_reg[4];
    half2 input_ldg_reg[4];
    
    // 位置计算保持不变
    int posh_ori[4];
    int posw_ori[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        posh_ori[i] = ((bx * 128 + tx % 32 + i * 32) / param.Ow) * param.u - param.p;
        posw_ori[i] = ((bx * 128 + tx % 32 + i * 32) % param.Ow) * param.v - param.q;
    }

    int inOffset = z * param.c * param.h * param.w;
    int weiOffset = (by * 128 + tx / 8 * 4) * param.c * param.r * param.s;
    int inChannelOffset = param.h * param.w;
    int weightKOffset = param.c * param.r * param.s;

    // sts addr计算与原来相同
    uint32_t weight_sts_addr = (tx % 8) * 132 * 2 + (tx / 8) * 4 * 2;
    uint32_t input_sts_addr = (tx / 32) * 128 * 2 + (tx % 32) * 2;

    half weight_frag[8];
    half input_frag[8];
    float output_frag[8][8];  // 中间结果仍使用float累积以保持精度
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            output_frag[i][j] = 0;
        }
    }

    for (int crs = 0; crs < param.r * param.s * param.c; crs += 8)
    {
        // 加载权重数据（从half类型数据）
        int weiOffsetTmp = crs + tx % 8;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if (weiOffsetTmp < weightKOffset) {
                half2 tmp;
                tmp.x = param.weight_half[weiOffset + weiOffsetTmp + i * weightKOffset];
                tmp.y = param.weight_half[weiOffset + weiOffsetTmp + i * weightKOffset + 1];
                weight_ldg_reg[i] = tmp;
            } else {
                weight_ldg_reg[i].x = __float2half(0.0f);
                weight_ldg_reg[i].y = __float2half(0.0f);
            }
        }
        
        // 加载输入数据（从half类型数据）
        int curC = (crs + tx / 32) / (param.r * param.s);
        int curR = ((crs + tx / 32) % (param.r * param.s)) / param.s;
        int curS = ((crs + tx / 32) % (param.r * param.s)) % param.s;

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int curH = posh_ori[i] + curR;
            int curW = posw_ori[i] + curS;
            int inOffsetTmp = curC * inChannelOffset + curH * param.w + curW;
            
            if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h) {
                half2 tmp;
                tmp.x = param.input_half[inOffset + inOffsetTmp];
                tmp.y = param.input_half[inOffset + inOffsetTmp + 1];
                input_ldg_reg[i] = tmp;
            } else {
                input_ldg_reg[i].x = __float2half(0.0f);
                input_ldg_reg[i].y = __float2half(0.0f);
            }
        }
        
        // 存入共享内存
        for (int i = 0; i < 4; ++i) {
            reinterpret_cast<half2*>(&smemweight[weight_sts_addr])[i] = weight_ldg_reg[i];
        }
        
        for (int i = 0; i < 4; ++i) {
            reinterpret_cast<half2*>(&smeminput[input_sts_addr + i * 32 * 2])[0] = input_ldg_reg[i];
        }
        
        __syncthreads();
        
        // 从共享内存读取并计算
#pragma unroll
        for (int subcrs = 0; subcrs < 8; ++subcrs)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                weight_frag[i] = smemweight[weight_lds_addr * 2 + subcrs * 132 * 2 + i * 2];
                weight_frag[i + 4] = smemweight[weight_lds_addr * 2 + subcrs * 132 * 2 + (i + 16) * 2];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                input_frag[i] = smeminput[input_lds_addr * 2 + subcrs * 128 * 2 + i * 2];
                input_frag[i + 4] = smeminput[input_lds_addr * 2 + subcrs * 128 * 2 + (i + 32) * 2];
            }

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    output_frag[i][j] += __half2float(weight_frag[i]) * __half2float(input_frag[j]);
                }
            }
        }
        __syncthreads();
    }

    // 计算输出偏移并写回输出张量
    int outOffset;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            outOffset = z * param.k * param.Oh * param.Ow + (y + i) * param.Oh * param.Ow + x + j;
            if (x + j < param.Oh * param.Ow && y + i < param.k)
            {
                param.output_half[outOffset] = __float2half(output_frag[i][j]);
            }
            
            outOffset = z * param.k * param.Oh * param.Ow + (y + i) * param.Oh * param.Ow + x + j + 32;
            if (x + j + 32 < param.Oh * param.Ow && y + i < param.k)
            {
                param.output_half[outOffset] = __float2half(output_frag[i][j + 4]);
            }
            
            outOffset = z * param.k * param.Oh * param.Ow + (y + i + 16) * param.Oh * param.Ow + x + j;
            if (x + j < param.Oh * param.Ow && y + i + 16 < param.k)
            {
                param.output_half[outOffset] = __float2half(output_frag[i + 4][j]);
            }
            
            outOffset = z * param.k * param.Oh * param.Ow + (y + i + 16) * param.Oh * param.Ow + x + j + 32;
            if (x + j + 32 < param.Oh * param.Ow && y + i + 16 < param.k)
            {
                param.output_half[outOffset] = __float2half(output_frag[i + 4][j + 4]);
            }
        }
    }
}

// 保留原始float版本的implgemm
__global__ void implgemm_fp32(param_t param) {
    // 保留原有代码...
}

void conv2d_cuda_forward(param_t param)
{
    int blockx = ((param.Oh * param.Ow + 127) / 128);
    int blocky = (param.k + 127) / 128;
    int blockz = param.n;
    int threadx = 256;
    int thready = 1;
    int threadz = 1;
    dim3 block(threadx, thready, threadz);
    dim3 grid(blockx, blocky, blockz);
    
    // 根据精度类型调用不同的核函数
    if (param.use_half) {
        implgemm_fp16<<<grid, block>>>(param);
    } else {
        implgemm_fp32<<<grid, block>>>(param);
    }
}

// 类似地修改反向传播函数...
// 为了简洁，这里省略反向传播代码的修改
void conv2d_cuda_backward(param_t param)
{
    // 相应修改类似于前向传播...
}