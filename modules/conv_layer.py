import torch
import torch.nn as nn
import conv2d_cuda
import math
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple

from typing import Optional, List, Tuple, Union

class Conv2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, params):
        stride = _pair(params[0])
        padding = _pair(params[1])
        output = conv2d_cuda.forward(input.contiguous(),weight.contiguous(),(stride),(padding))
        variables = [input, weight, params]
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, params = ctx.saved_variables 
        stride = _pair(params[0])
        padding = _pair(params[1])
        grad_input,grad_weight = conv2d_cuda.backward(input.contiguous(), grad_output.contiguous(), weight.contiguous(),(stride),(padding))
        return grad_input, grad_weight, None
    

class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        bias: bool = False,
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        super(Conv2d, self).__init__()

        self.params=torch.Tensor([stride_,padding_])
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size_, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        return Conv2DFunction.apply(input, self.weight, self.params)