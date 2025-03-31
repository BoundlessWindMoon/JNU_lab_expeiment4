#include <torch/types.h>
#include <cuda_fp16.h>  // 添加对半精度的支持

// 修改参数结构体，添加FP16支持
typedef struct
{
    // 原始定义保持不变，作为float类型的参数
    float*   input;                                   //输入数据地址
    float*   grad_input;                              //输入梯度数据地址
    float*   weight;                                  //权值数据地址
    float*   grad_weight;                             //权值梯度数据地址
    float*   output;                                  //输出数据地址
    float*   grad_output;                             //输出梯度数据地址

    // 添加half类型的参数
    half*   input_half;                               //半精度输入数据地址
    half*   grad_input_half;                          //半精度输入梯度数据地址
    half*   weight_half;                              //半精度权值数据地址
    half*   grad_weight_half;                         //半精度权值梯度数据地址
    half*   output_half;                              //半精度输出数据地址
    half*   grad_output_half;                         //半精度输出梯度数据地址
    
    unsigned int      n;                              //batch szie              
    unsigned int      c;                              //channel number          
    unsigned int      h;                              //数据高                  
    unsigned int      w;                              //数据宽                  
    unsigned int      k;                              //卷积核数量              
    unsigned int      r;                              //卷积核高                
    unsigned int      s;                              //卷积核宽                
    unsigned int      u;                              //卷积在高方向上的步长    
    unsigned int      v;                              //卷积在宽方向上的步长    
    unsigned int      p;                              //卷积在高方向上的补边    
    unsigned int      q;                              //卷积在宽方向上的补边    
    unsigned int      Oh;                             //卷积结果高             
    unsigned int      Ow;                             //卷积结果宽
    bool              use_half;                       //是否使用半精度
}param_t;