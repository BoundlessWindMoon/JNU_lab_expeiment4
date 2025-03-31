#include <torch/extension.h>
#include <vector>
#include <cuda_fp16.h>
#include "conv2d_fp16.h"

// CUDA forward/backward声明
void conv2d_cuda_forward(param_t param);
void conv2d_cuda_backward(param_t param);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);

    // 卷积参数配置
    param_t param;
    param.use_half = (input.scalar_type() == torch::kHalf);
    
    // 通用参数设置
    param.n = input.size(0);
    param.c = input.size(1);
    param.h = input.size(2);
    param.w = input.size(3);
    param.k = weight.size(0);
    param.r = weight.size(2);
    param.s = weight.size(3);
    param.u = stride[0];
    param.v = stride[1];
    param.p = padding[0];
    param.q = padding[1];

    int64_t outh = (param.h - param.r + 2 * param.p) / param.u + 1;
    int64_t outw = (param.w - param.s + 2 * param.q) / param.v + 1;

    param.Oh = outh;
    param.Ow = outw;

    // 根据精度类型设置指针
    if (param.use_half) {
        auto output = torch::zeros(
            torch::IntArrayRef({input.size(0), weight.size(0), outh, outw}),
            torch::TensorOptions().device(input.device()).dtype(torch::kHalf)
        );
        
        param.input_half = reinterpret_cast<half*>(input.data_ptr());
        param.weight_half = reinterpret_cast<half*>(weight.data_ptr());
        param.output_half = reinterpret_cast<half*>(output.data_ptr());
        
        // 为了保持兼容性，设置float指针为nullptr
        param.input = nullptr;
        param.weight = nullptr;
        param.output = nullptr;
        
        conv2d_cuda_forward(param);
        return output;
    } else {
        auto output = torch::zeros(
            torch::IntArrayRef({input.size(0), weight.size(0), outh, outw}),
            input.options()
        );
        
        param.input = static_cast<float*>(input.data_ptr());
        param.weight = static_cast<float*>(weight.data_ptr());
        param.output = static_cast<float*>(output.data_ptr());
        
        // 设置half指针为nullptr
        param.input_half = nullptr;
        param.weight_half = nullptr;
        param.output_half = nullptr;
        
        conv2d_cuda_forward(param);
        return output;
    }
}

std::vector<torch::Tensor> conv2d_backward(
    torch::Tensor input,
    torch::Tensor grad_output,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding)
{
    CHECK_INPUT(input);
    CHECK_INPUT(grad_output);
    CHECK_INPUT(weight);

    // 卷积参数配置
    param_t param;
    param.use_half = (input.scalar_type() == torch::kHalf);
    
    // 相同的通用参数设置...
    param.n = input.size(0);
    param.c = input.size(1);
    param.h = input.size(2);
    param.w = input.size(3);
    param.k = weight.size(0);
    param.r = weight.size(2);
    param.s = weight.size(3);
    param.u = stride[0];
    param.v = stride[1];
    param.p = padding[0];
    param.q = padding[1];
    
    int64_t outh = (param.h - param.r + 2 * param.p) / param.u + 1;
    int64_t outw = (param.w - param.s + 2 * param.q) / param.v + 1;
    
    param.Oh = outh;
    param.Ow = outw;

    // 同样根据精度类型设置输出和指针...
    if (param.use_half) {
        auto grad_input = torch::zeros(
            torch::IntArrayRef({input.size(0), input.size(1), input.size(2), input.size(3)}),
            torch::TensorOptions().device(input.device()).dtype(torch::kHalf)
        );
        auto grad_weight = torch::zeros(
            torch::IntArrayRef({weight.size(0), input.size(1), weight.size(2), weight.size(3)}),
            torch::TensorOptions().device(input.device()).dtype(torch::kHalf)
        );
        
        param.input_half = reinterpret_cast<half*>(input.data_ptr());
        param.grad_output_half = reinterpret_cast<half*>(grad_output.data_ptr());
        param.weight_half = reinterpret_cast<half*>(weight.data_ptr());
        param.grad_input_half = reinterpret_cast<half*>(grad_input.data_ptr());
        param.grad_weight_half = reinterpret_cast<half*>(grad_weight.data_ptr());
        
        // 设置float指针为nullptr
        param.input = nullptr;
        param.grad_output = nullptr;
        param.weight = nullptr;
        param.grad_input = nullptr;
        param.grad_weight = nullptr;
        
        conv2d_cuda_backward(param);
        return {grad_input, grad_weight};
    } else {
        auto grad_input = torch::zeros(
            torch::IntArrayRef({input.size(0), input.size(1), input.size(2), input.size(3)}),
            grad_output.options()
        );
        auto grad_weight = torch::zeros(
            torch::IntArrayRef({weight.size(0), input.size(1), weight.size(2), weight.size(3)}),
            grad_output.options()
        );
        
        param.input = static_cast<float*>(input.data_ptr());
        param.grad_output = static_cast<float*>(grad_output.data_ptr());
        param.weight = static_cast<float*>(weight.data_ptr());
        param.grad_input = static_cast<float*>(grad_input.data_ptr());
        param.grad_weight = static_cast<float*>(grad_weight.data_ptr());
        
        // 设置half指针为nullptr
        param.input_half = nullptr;
        param.grad_output_half = nullptr;
        param.weight_half = nullptr;
        param.grad_input_half = nullptr;
        param.grad_weight_half = nullptr;
        
        conv2d_cuda_backward(param);
        return {grad_input, grad_weight};
    }
}

// Python模块接口绑定（不变）
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &conv2d_forward, "Convolution2d forward (CUDA)");
    m.def("backward", &conv2d_backward, "Convolution2d backward (CUDA)");
}