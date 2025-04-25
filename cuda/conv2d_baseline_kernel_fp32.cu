#include <torch/extension.h>
#include "conv2d_fp32.h"

__global__ void directConvolution(param_t param) 
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    if (x >= param.Oh * param.Ow || y >= param.k || z >= param.n)
    {
        return;
    }

    int posOh = x / param.Ow;
    int posOw = x % param.Ow;

    int posh_ori = posOh * param.u - param.p;
    int posw_ori = posOw * param.v - param.q;

    float sum = 0.0;

    int inOffset = z * param.c * param.h * param.w + posh_ori * param.w + posw_ori;
    int weiOffset = y * param.c * param.r * param.s;
    int inChannelOffset = param.h * param.w;
    int weightChannelOffset = param.r * param.s;

    for (int i = 0; i < param.r; i++)
    {
        for (int j = 0; j < param.s; j++)
        {
            int posh_real = posh_ori + i;
            int posw_real = posw_ori + j;

            if (posh_real >= 0 && posw_real >= 0 && posw_real < param.w && posh_real < param.h)
            {
                int inOffsetTmp = inOffset;
                int weiOffsetTmp = weiOffset;
                for (int channel = 0; channel < param.c; channel++)
                {
                    sum += (float)(param.input[inOffsetTmp + i * param.w + j] * param.weight[weiOffsetTmp + i * param.s + j]);
                    inOffsetTmp += inChannelOffset;
                    weiOffsetTmp += weightChannelOffset;
                }
            }
        }
    }

    int outOffset = z * param.k * param.Oh * param.Ow + y * param.Oh * param.Ow + x;
    param.output[outOffset] = sum;
}


void conv2d_cuda_forward(param_t param) {
    const dim3 block(16, 16);  
    const dim3 grid(
        (param.Oh*param.Ow+15) / 16,   
        (param.k+15) / 16,    
        param.n                               
    );
    directConvolution<<<grid, block>>>(param);
}

__global__ void implgemmbwddata(param_t param)
{
    uint32_t tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t mma_tid_x = (lane_id / 2) % 8;
    const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);

    uint32_t weight_lds_addr = (warp_id / 2) * 32 + mma_tid_y * 4;
    uint32_t gradoutput_lds_addr = (warp_id % 2) * 64 + mma_tid_x * 4;

    int x = bx * 128 + gradoutput_lds_addr;
    int y = by * 128 + weight_lds_addr;
    int z = blockIdx.z;

    __shared__ float smemgradoutput[8 * 128];
    __shared__ float smemweight[8 * 132];

    float weight_ldg_reg[4];
    float gradoutput_ldg_reg[4];

    int posOh_ori[4];
    int posOw_ori[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        posOh_ori[i] = ((bx * 128 + tx % 32 + i * 32) / param.w) - (param.r - 1 - param.p);
        posOw_ori[i] = ((bx * 128 + tx % 32 + i * 32) % param.w) - (param.s - 1 - param.q);
    }

    int outOffset = z * param.k * param.Oh * param.Ow;
    int weiC = (by * 128 + tx / 8 * 4);
    int outKOffset = param.Oh * param.Ow;
    int weiCOffset = param.r * param.s;
    int weiKOffset = param.c * param.r * param.s;

    uint32_t weight_sts_addr = (tx % 8) * 132 +
                               (tx / 8) * 4;
    uint32_t gradoutput_sts_addr = (tx / 32) * 128 + (tx % 32);

    float weight_frag[8];
    float gradoutput_frag[8];
    float gradinput_frag[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            gradinput_frag[i][j] = 0;
        }
    }

    for (int krs = 0; krs < param.r * param.s * param.k; krs += 8)
    {
        // ldg
        int curKRS = krs + tx % 8;
        int rs = param.r * param.s - 1 - curKRS % (param.r * param.s);
        int curK = curKRS / (param.r * param.s);
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if ((curK * param.r * param.s + rs) < param.r * param.s * param.k)
            {
                weight_ldg_reg[i] = param.weight[curK * weiKOffset + (weiC + i) * weiCOffset + rs];
            }
            else
            {
                weight_ldg_reg[i] = 0.0;
            }
        }
        int curK2 = (krs + tx / 32) / (param.r * param.s);           
        int curR = ((krs + tx / 32) % (param.r * param.s)) / param.s; 
        int curS = ((krs + tx / 32) % (param.r * param.s)) % param.s; 

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int curOh = posOh_ori[i] + curR; 
            int curOw = posOw_ori[i] + curS; 
            int outOffsetTmp = curK2 * outKOffset + curOh * param.Ow + curOw;
            if (curOh >= 0 && curOw >= 0 && curOw < param.Ow && curOh < param.Oh)
            {
                gradoutput_ldg_reg[i] = param.grad_output[outOffset + outOffsetTmp];
            }
            else
            {
                gradoutput_ldg_reg[i] = 0.0;
            }
        }

        for (int i = 0; i < 4; ++i)
        {
            smemweight[weight_sts_addr + i] = weight_ldg_reg[i];
        }
        for (int i = 0; i < 4; ++i)
        {
            smemgradoutput[gradoutput_sts_addr + i * 32] = gradoutput_ldg_reg[i];
        }
        __syncthreads();
#pragma unroll
        for (int subkrs = 0; subkrs < 8; ++subkrs)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                weight_frag[i] = smemweight[weight_lds_addr + subkrs * 132 + i];
                weight_frag[i + 4] = smemweight[weight_lds_addr + subkrs * 132 + i + 16];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                gradoutput_frag[i] = smemgradoutput[gradoutput_lds_addr + subkrs * 128 + i];
                gradoutput_frag[i + 4] = smemgradoutput[gradoutput_lds_addr + subkrs * 128 + i + 32];
            }

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    gradinput_frag[i][j] += weight_frag[i] * gradoutput_frag[j];
                }
            }
        }
        __syncthreads();
    }

    int gradinputOffset;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            gradinputOffset = z * param.c * param.h * param.w + (y + i) * param.h * param.w + x + j;
            if (x + j < param.h * param.w && y + i < param.c)
            {
                param.grad_input[gradinputOffset] = gradinput_frag[i][j];
            }
            gradinputOffset = z * param.c * param.h * param.w + (y + i) * param.h * param.w + x + j + 32;
            if (x + j + 32 < param.h * param.w && y + i < param.c)
            {
                param.grad_input[gradinputOffset] = gradinput_frag[i][j + 4];
            }
            gradinputOffset = z * param.c * param.h * param.w + (y + i + 16) * param.h * param.w + x + j;
            if (x + j < param.h * param.w && y + i + 16 < param.c)
            {
                param.grad_input[gradinputOffset] = gradinput_frag[i + 4][j];
            }
            gradinputOffset = z * param.c * param.h * param.w + (y + i + 16) * param.h * param.w + x + j + 32;
            if (x + j + 32 < param.h * param.w && y + i + 16 < param.c)
            {
                param.grad_input[gradinputOffset] = gradinput_frag[i + 4][j + 4];
            }
        }
    }
}
__global__ void implgemmbwdweight(param_t param)
{
    uint32_t tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Warp tile
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t mma_tid_x = (lane_id / 2) % 8;
    const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    // lds addr
    uint32_t gradoutput_lds_addr = (warp_id / 2) * 32 + mma_tid_y * 4;
    uint32_t input_lds_addr = (warp_id % 2) * 64 + mma_tid_x * 4;

    int x = bx * 128 + input_lds_addr;
    int y = by * 128 + gradoutput_lds_addr;
    int z = blockIdx.z;

    __shared__ float smeminput[8 * 128];
    __shared__ float smemgradoutput[8 * 132];
    // 当前线程处理的数据点在oh、ow上的坐标
    int posh_ori[4];
    int posw_ori[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        posh_ori[i] = ((bx * 128 + tx % 32 + i * 32) / param.s) - param.p;
        posw_ori[i] = ((bx * 128 + tx % 32 + i * 32) % param.s) - param.q;
    }

    int inOffset = z * param.h * param.w;
    int outK = (by * 128 + tx / 8 * 4);
    int inNOffset = param.c * param.h * param.w;
    int outKOffset = param.Oh * param.Ow;
    int outNOffset = param.k * param.Oh * param.Ow;

    // sts addr
    uint32_t gradoutput_sts_addr = (tx % 8) * 132 +
                                   (tx / 8) * 4;
    uint32_t input_sts_addr = (tx / 32) * 128 + (tx % 32);

    float gradoutput_frag[8];
    float input_frag[8];
    float gradweight_frag[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            gradweight_frag[i][j] = 0;
        }
    }

    for (int nohow = 0; nohow < param.Oh * param.Ow * param.n; nohow += 8)
    {
        int curNOHOW = nohow + tx % 8;
        int ohow = curNOHOW % (param.Oh * param.Ow);
        int curN_1 = curNOHOW / (param.Oh * param.Ow);
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if (curNOHOW < param.Oh * param.Ow * param.n)
            {
                smemgradoutput[gradoutput_sts_addr + i] = param.grad_output[curN_1 * outNOffset + (outK + i) * outKOffset + ohow];
            }
            else
            {
                smemgradoutput[gradoutput_sts_addr + i] = 0.0;
            }
        }

        int curN_2 = (nohow + tx / 32) / (param.Oh * param.Ow);             // output n offset
        int curOh = ((nohow + tx / 32) % (param.Oh * param.Ow)) / param.Ow; // output h offset
        int curOw = ((nohow + tx / 32) % (param.Oh * param.Ow)) % param.Ow; // output w offset

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int curH = posh_ori[i] + curOh; // input h
            int curW = posw_ori[i] + curOw; // input w
            int inOffsetTmp = curN_2 * inNOffset + curH * param.w + curW;
            if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h)
            {
                smeminput[input_sts_addr + i * 32] = param.input[inOffset + inOffsetTmp];
            }
            else
            {
                smeminput[input_sts_addr + i * 32] = 0.0;
            }
        }
        __syncthreads();
#pragma unroll
        for (int subnohow = 0; subnohow < 8; ++subnohow)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                gradoutput_frag[i] = smemgradoutput[gradoutput_lds_addr + subnohow * 132 + i];
                gradoutput_frag[i + 4] = smemgradoutput[gradoutput_lds_addr + subnohow * 132 + i + 16];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                input_frag[i] = smeminput[input_lds_addr + subnohow * 128 + i];
                input_frag[i + 4] = smeminput[input_lds_addr + subnohow * 128 + i + 32];
            }

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    gradweight_frag[i][j] += gradoutput_frag[i] * input_frag[j];
                }
            }
        }
        __syncthreads();
    }

    // 计算输出偏移
    int gradweightoffset;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            gradweightoffset = z * param.r * param.s + (y + i) * param.c * param.r * param.s + x + j;
            if (x + j < param.r * param.s && y + i < param.k)
            {
                param.grad_weight[gradweightoffset] = gradweight_frag[i][j];
            }
            gradweightoffset = z * param.r * param.s + (y + i) * param.c * param.r * param.s + x + j + 32;
            if (x + j + 32 < param.r * param.s && y + i < param.k)
            {
                param.grad_weight[gradweightoffset] = gradweight_frag[i][j + 4];
            }
            gradweightoffset = z * param.r * param.s + (y + i + 16) * param.c * param.r * param.s + x + j;
            if (x + j < param.r * param.s && y + i + 16 < param.k)
            {
                param.grad_weight[gradweightoffset] = gradweight_frag[i + 4][j];
            }
            gradweightoffset = z * param.r * param.s + (y + i + 16) * param.c * param.r * param.s + x + j + 32;
            if (x + j + 32 < param.r * param.s && y + i + 16 < param.k)
            {
                param.grad_weight[gradweightoffset] = gradweight_frag[i + 4][j + 4];
            }
        }
    }
}

void conv2d_cuda_backward(param_t param)
{
    int blockx = ((param.h * param.w + 127) / 128); // blockx  number
    int blocky = (param.c + 127) / 128;       // blocky  number
    int blockz = param.n;                     // blockz  number
    // 合并threadx与thready
    int threadx = 256; // threadx number per block
    int thready = 1;   // thready number per block
    int threadz = 1;   // threadz number per block
    dim3 blockbwddata(threadx, thready, threadz);
    dim3 gridbwddata(blockx, blocky, blockz);
    implgemmbwddata<<<gridbwddata, blockbwddata>>>(param);

    blockx = (param.r * param.s + 127) / 128; // blockx  number
    blocky = (param.k + 127) / 128;     // blocky  number
    blockz = param.c;                   // blockz  number
    // 合并threadx与thready
    threadx = 256; // threadx number per block
    thready = 1;   // thready number per block
    threadz = 1;   // threadz number per block
    dim3 blockbwdweight(threadx, thready, threadz);
    dim3 gridbwdweight(blockx, blocky, blockz);
    implgemmbwdweight<<<gridbwdweight, blockbwdweight>>>(param);
}