import torch
import torchvision
import torchvision.transforms as transforms
import time
import sys
from tqdm import tqdm
from torch.utils.data import ConcatDataset
from modules.resnet_18_fp16 import ResNet18 as ResNet18_optim_fp16
from modules.resnet_18_baseline_fp16 import ResNet18 as ResNet18_baseline_fp16
from torch.amp import autocast

# --------------------------
# 参数配置
# --------------------------
MODEL_PATH = "./pytorch/model/net_123.pth"
DATA_ROOT = "./pytorch/data"
BATCH_SIZE = 256
NUM_WORKERS = 4
RUN_TIMES = 5  # 运行次数参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROGRESS_UNIT = "img"

# --------------------------
# 模型加载（FP16版本）
# --------------------------
def load_model(model_path, model_type='baseline'):
    # 根据类型选择模型类
    if model_type == 'optim':
        model = ResNet18_optim_fp16()
    elif model_type == 'baseline':
        model = ResNet18_baseline_fp16()
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Choose ['optim', 'baseline']")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint)
    model = model.to(DEVICE).eval()
    
    # 完整的 FP16 转换
    model = model.half()
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            # 保持批量归一化层为 float32
            layer.float()
    
    return model
    
model_optim = load_model(MODEL_PATH, model_type='optim')
model_baseline = load_model(MODEL_PATH, model_type='baseline')

def calculate_final_score(accuracy, throughput_optim, throughput_baseline):
    accuracy_weight = 4
    speed_weight = 15
    
    normalized_throughput = throughput_optim / throughput_baseline

    final_score = accuracy_weight * (accuracy / 10) + speed_weight * normalized_throughput
    return min(final_score, 100)

# Evaluate accuracy with FP16
def evaluate_accuracy():
    correct = 0
    total = 0
    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model_optim(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy (FP16): {accuracy:.3f}%')
    if accuracy < 87:
        print("[警告] FP16精度低于预期，但继续执行以进行性能测试")
    
    return accuracy

def process_single_run(model, dataloader, device, run_num, total_runs):
    total_images = len(dataloader.dataset)
    
    progress_bar = tqdm(
        dataloader,
        desc=f"轮次 {run_num:02d}/{total_runs}",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        unit='img',
        unit_scale=BATCH_SIZE,
        dynamic_ncols=True
    )

    # 预热逻辑 - 使用autocast而不是手动转换
    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16, enabled=True):
        warmup_tensor = torch.randn(BATCH_SIZE, 3, 32, 32, device=device)
        for _ in range(3):
            _ = model(warmup_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # 正式推理逻辑
    run_start = time.perf_counter()
    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16, enabled=True):
        for images, _ in progress_bar:
            images = images.to(device, non_blocking=True)
            _ = model(images)
            
            current_time = time.perf_counter() - run_start
            current_throughput = (progress_bar.n * BATCH_SIZE) / current_time

    # 最终计算
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    run_elapsed = time.perf_counter() - run_start
    final_throughput = total_images / run_elapsed

    progress_bar.set_postfix_str(f"{final_throughput:.1f} img/s (最终)")
    progress_bar.close()
    return run_elapsed


# --------------------------
# 数据加载
# --------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

testset = torchvision.datasets.CIFAR10(
    root=DATA_ROOT, train=False, download=True, transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("=" * 50)
    print("ResNet-18 FP16 (半精度) 推理性能测试")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed = len(testset)
    total_processed = processed * RUN_TIMES
    elapsed_recorder = []
    
    # 原始评估流程
    accuracy = evaluate_accuracy()
    
    # 优化模型测试
    for run_idx in range(1, RUN_TIMES+1):
        elapsed = process_single_run(model_optim, testloader, DEVICE, run_idx, RUN_TIMES)
        elapsed_recorder.append(elapsed)
    
    total_elapsed = sum(elapsed_recorder)
    avg_throughput = total_processed / total_elapsed
    print(f"FP16手写算子平均吞吐量: {avg_throughput:.2f} img/s\n")
    
    # 基准测试
    baseline_elapsed = process_single_run(model_baseline, testloader, DEVICE, 1, 1)
    baseline_throughput = processed / baseline_elapsed
    print(f"FP16基准吞吐量: {baseline_throughput:.2f} img/s\n")
    
    # 最终得分计算
    final_score = calculate_final_score(accuracy, avg_throughput, baseline_throughput)
    print(f'FP16最终得分: {final_score:.3f}')
    
    # 准备绘图数据
    optim_throughputs = []
    for run_idx in range(RUN_TIMES):
        current_processed = processed * (run_idx + 1)
        current_throughput = current_processed / sum(elapsed_recorder[:run_idx+1])
        optim_throughputs.append(current_throughput)
    
    speedup_ratio = [t / baseline_throughput for t in optim_throughputs]
    run_numbers = list(range(1, RUN_TIMES+1))
    
    # 绘制图表
    plt.figure(figsize=(10, 5))
    plt.plot(run_numbers, speedup_ratio, 
             marker='o', label='FP16 Optimized', color='#E6550D')
    plt.axhline(y=1, color='#3182BD', linestyle='--', label='FP16 Baseline')
    
    # 图表标注
    plt.title(f'FP16 Throughput Acceleration (Final: {final_score:.2f} pts)')
    plt.xlabel('Run Times')
    plt.ylabel('Speedup Ratio (vs Baseline)')
    plt.xticks(run_numbers)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    # 显示图表
    plt.savefig('fp16_performance.png')
    plt.show()