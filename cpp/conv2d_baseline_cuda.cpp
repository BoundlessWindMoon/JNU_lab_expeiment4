#include <torch/extension.h>
#include <vector>

#include "conv2d.h"

// CUDA forward/backward declarations

void conv2d_cuda_forward(param_t param);

void conv2d_cuda_backward(param_t param);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
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

    // Convolution parameter

    param_t param;
    param.input = (float *)input.data_ptr();
    param.weight = (float *)weight.data_ptr();
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

    auto output = torch::zeros(torch::IntArrayRef({input.size(0), weight.size(0), outh, outw}),
                               input.options());

    param.output = (float *)output.data_ptr();
    conv2d_cuda_forward(param);
    return output;
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

    // Convolution parameter

    param_t param;
    param.input = (float *)input.data_ptr();
    param.grad_output = (float *)grad_output.data_ptr();
    param.weight = (float *)weight.data_ptr();
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

    auto grad_input = torch::zeros(torch::IntArrayRef({input.size(0), input.size(1), input.size(2), input.size(3)}),
                                    grad_output.options());
    auto grad_weight = torch::zeros(torch::IntArrayRef({weight.size(0), input.size(1), weight.size(2), weight.size(3)}),
                                    grad_output.options());

    param.grad_input = (float *)grad_input.data_ptr();
    param.grad_weight = (float *)grad_weight.data_ptr();
    conv2d_cuda_backward(param);
    return {grad_input, grad_weight};
}


// Python module interface bind
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &conv2d_forward, "Convolution2d forward (CUDA)");
    m.def("backward", &conv2d_backward, "Convolution2d backward (CUDA)");
}