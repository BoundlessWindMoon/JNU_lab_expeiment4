import torch
import torchvision
import torchvision.transforms as transforms
import time
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from modules.resnet_18 import ResNet18 as ResNet18_optim
from modules.resnet_18_baseline_fp32 import ResNet18 as ResNet18_baseline

# --------------------------
# 参数配置
# --------------------------
MODEL_PATH = "./pytorch/model/net_123.pth"
DATA_ROOT = "./pytorch/data"
MODEL_DTYPE = "FP16"  # 优化模型的推理精度
BATCH_SIZES = [8, 16, 32, 64, 128, 256, 512]  # 需要测试的batch size列表
NUM_WORKERS = 4
RUN_TIMES = 1  # 每个batch size的测试轮次（仅优化模型）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# 模型加载（保持原样）
# --------------------------
def load_model(model_path, model_type='baseline', model_dtype='FP32'):
    if model_type == 'optim':
        model = ResNet18_optim()
    elif model_type == 'baseline':
        model = ResNet18_baseline()
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint)
    
    dtype = torch.half if model_dtype == "FP16" else torch.float32
    model = model.to(DEVICE, dtype=dtype)
    model.eval()
    return model

# --------------------------
# 推理性能测试函数
# --------------------------
def process_single_run(model, dataloader, device, model_dtype='FP32'):
    warmup_dtype = torch.half if model_dtype == "FP16" else torch.float32
    total_images = len(dataloader.dataset)
    
    # Warmup阶段（保持简洁）
    with torch.no_grad():
        warmup_tensor = torch.randn(dataloader.batch_size, 3, 32, 32, 
                                   dtype=warmup_dtype).to(device)
        model(warmup_tensor)  # 单次warmup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    # 正式推理（添加tqdm进度条）
    start_time = time.perf_counter()
    
    with torch.no_grad():
        # 创建可自定义的进度条
        progress_bar = tqdm(
            dataloader,
            desc=f"reasoning (bs={dataloader.batch_size})",
            ncols=100,  # 进度条宽度
            bar_format="{l_bar}{bar} [{elapsed}<{remaining}]"
        )
        
        for images, _ in progress_bar:
            # 数据传输与推理
            images = images.to(device=device, dtype=warmup_dtype)
            _ = model(images)
            
            # 实时更新吞吐量信息
            processed = (progress_bar.n + 1) * dataloader.batch_size
            current_speed = processed / (time.perf_counter() - start_time)
            progress_bar.set_postfix({
                "speed": f"{current_speed:.1f} img/s"
            })
    # 最终同步与计时
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    
    return total_images / elapsed


if __name__ == "__main__":
    # 加载数据集（仅加载一次）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    testset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )
    # 加载模型
    model_optim = load_model(MODEL_PATH, 'optim', MODEL_DTYPE)
    model_baseline = load_model(MODEL_PATH, 'baseline', 'FP32')
    ##############################################
    # 新增部分：可视化前20张图片的预测结果
    ##############################################
    # 获取前20个样本
    subset_indices = range(50)
    subset = torch.utils.data.Subset(testset, subset_indices)
    subset_loader = DataLoader(subset, batch_size=50, shuffle=False)
    # 反标准化函数
    def denormalize(tensor):
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1)
        return tensor.cpu() * std + mean
    # 执行推理
    with torch.no_grad():
        images, labels = next(iter(subset_loader))
        images = images.to(DEVICE, dtype=torch.half if MODEL_DTYPE == "FP16" else torch.float32)
        outputs = model_optim(images)
        preds = outputs.argmax(dim=1).cpu()

    # 可视化设置
    plt.figure(figsize=(15, 12))
    for i in range(50):
        plt.subplot(5, 10, i+1)
        
        # 反标准化并转换为可显示格式
        img = denormalize(images[i].cpu().float()).clamp(0, 1)
        plt.imshow(img.permute(1, 2, 0))  # CHW -> HWC
        
        # 获取标签文本
        true_label = testset.classes[labels[i]]
        pred_label = testset.classes[preds[i]]
        
        # 设置标题颜色
        title_color = "red" if true_label != pred_label else "black"
        plt.title(f"Label: {true_label}\nPred: {pred_label}", 
                 color=title_color, fontsize=9)
        plt.axis('off')
    
    plt.suptitle("Optimized Model Prediction Visualization (First 50 Samples)", 
                y=0.99, fontsize=14)
    plt.tight_layout()
    plt.show()
    ##############################################
    # 原有性能测试逻辑（保持不变）
    ##############################################
    # 存储结果
    optim_throughputs = []
    baseline_throughput = None

    # 测试基准模型（仅batch_size=128时运行一次）
    baseline_loader = DataLoader(
        testset,
        batch_size=128,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    print(f"Testing baseline:")
    baseline_throughput = process_single_run(
        model=model_baseline,
        dataloader=baseline_loader,
        device=DEVICE,
        model_dtype='FP32'
    )
    print(f"[Baseline] Batch Size=128 | Throughput: {baseline_throughput:.2f} img/s")

    # 测试优化模型（所有batch size）
    for batch_size in BATCH_SIZES:
        
        # 创建当前batch size的数据加载器
        testloader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        
        # 多轮测试取平均
        throughputs = []
        print(f"\nTesting batch_size = {batch_size}")
        for run in range(RUN_TIMES):
            throughput = process_single_run(
                model=model_optim,
                dataloader=testloader,
                device=DEVICE,
                model_dtype=MODEL_DTYPE
            )
            throughputs.append(throughput)
        
        avg_throughput = sum(throughputs) / RUN_TIMES
        optim_throughputs.append(avg_throughput)
        print(f"[Optim] Batch Size={batch_size:3d} | Average Throughput: {avg_throughput:.2f} img/s")
    # --------------------------
    # 绘制结果对比图
    # --------------------------
    plt.figure(figsize=(12, 6))
    
    # 绘制优化模型曲线
    plt.plot(BATCH_SIZES, optim_throughputs, 
            marker='o', linestyle='-', color='#FF6F00', 
            linewidth=2, markersize=10, label='Optimized Model')
    
    # 绘制基准模型参考线
    plt.axhline(y=baseline_throughput, color='#1F77B4', linestyle='--', 
                linewidth=2, label=f'Baseline (Batch Size=128)')
    
    # 标注关键数据点
    plt.scatter([128], [optim_throughputs[-1]], color='red', zorder=5, 
                label=f'Optimized @128: {optim_throughputs[-1]:.1f} img/s')
    
    # 图表装饰
    plt.title('Optimized Model Throughput vs Baseline (Batch Size=128)', fontsize=14, pad=20)
    plt.xlabel('Batch Size', fontsize=12, labelpad=10)
    plt.ylabel('Throughput (images/sec)', fontsize=12, labelpad=10)
    plt.xticks(BATCH_SIZES, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper left')
    
    # 显示优化模型数据标签
    for x, y in zip(BATCH_SIZES, optim_throughputs):
        plt.text(x, y+50, f'{y:.1f}', ha='center', va='bottom', fontsize=10, color='#FF6F00')
    
    plt.tight_layout()
    plt.show()
