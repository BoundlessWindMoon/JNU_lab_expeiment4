# PyTorch Custom Convolution Layer

This project is about how to define a custom convolution layer in PyTorch, and use CUDA function to implement convolution.

## Content

[/cpp](https://github.com/Qwesh157/pytorch_custom_convolution_layer/blob/main/cpp) C++ extension include CUDA interface and Python module bind.

[/cuda](https://github.com/Qwesh157/pytorch_custom_convolution_layer/tree/main/cuda) Implicit gemm convolution implementation.

[/include](https://github.com/Qwesh157/pytorch_custom_convolution_layer/tree/main/include) Declaration about forward/backward convolution.

[/pytorch](https://github.com/Qwesh157/pytorch_custom_convolution_layer/tree/main/pytorch) Include setup.py script, custom convolution layer definition.

## Build

```bash
$ sh setup.sh
```

If you don't have root permission, add environment option `--prefix="/home/user/.conda/envs/yourenvname/`.

## TEST

```bash
$ cd pytorch
vim test.py
```


## RUN

run inference program

```
$ python inference.py
```

