import torch
import torchvision
import torchvision.transforms as transforms
import time
from tqdm import tqdm
from torch.utils.data import ConcatDataset
from modules.resnet_18 import ResNet18

# --------------------------
# 1. 参数配置
# --------------------------
MODEL_PATH = "./pytorch/model/net_123.pth"
DATA_ROOT = "./pytorch/data"
BATCH_SIZE = 256
NUM_WORKERS = 4
RUN_TIMES = 20  # 新增运行次数参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROGRESS_UNIT = "img"

# --------------------------
# 2. 数据加载（调整为原始测试集）
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

# --------------------------
# 3. 模型加载（保持原样）
# --------------------------
def load_model(model_path):
    model = ResNet18()
    checkpoint = torch.load(model_path, map_location=DEVICE,weights_only=False)
    model.load_state_dict(checkpoint)
    model = model.to(DEVICE).eval()
    return model

model = load_model(MODEL_PATH)

# Calculate the final score (25% accuracy, 75% speed)
def calculate_final_score(accuracy, throughput):
    # Define the weights for accuracy and speed
    accuracy_weight = 0.25
    speed_weight = 0.75
    
    # Normalize the throughput to be on a scale from 0 to 1
    normalized_throughput = throughput / 1000  # This assumes throughput is >1000 images per second as reasonable
    # Calculate the final score (accuracy contributes 25%, speed contributes 75%)
    final_score = accuracy_weight * (accuracy / 100) + speed_weight * normalized_throughput
    return final_score

# Evaluate accuracy
def evaluate_accuracy():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.3f}%')
    return accuracy

def process_single_run(model, dataloader, device, run_num, total_runs):
    total_images = len(dataloader.dataset)
    start_time = time.perf_counter()
    
    # 修正进度条参数配置
    progress_bar = tqdm(
        dataloader,
        desc=f"轮次 {run_num:02d}/{total_runs}",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [吞吐量={postfix}]",  # 保持postfix占位符
        unit='img',  # 关键修正点：使用字符串单位
        unit_scale=BATCH_SIZE,  # 在此处设置缩放比例
        dynamic_ncols=True
    )

    # 预热逻辑保持不变
    with torch.no_grad():
        warmup_tensor = torch.randn(BATCH_SIZE,3,32,32).to(device)
        for _ in range(10):
            _ = model(warmup_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # 正式推理逻辑
    run_start = time.perf_counter()
    with torch.no_grad():
        for images, _ in progress_bar:
            images = images.to(device, non_blocking=True)
            _ = model(images)
            
            current_time = time.perf_counter() - run_start
            current_throughput = (progress_bar.n * BATCH_SIZE) / current_time
            progress_bar.set_postfix_str(f"{current_throughput:.1f} img/s")  # 保持字符串格式

    # 最终计算
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    run_elapsed = time.perf_counter() - run_start
    final_throughput = total_images / run_elapsed

    progress_bar.set_postfix_str(f"{final_throughput:.1f} img/s (最终)")
    progress_bar.close()
    return run_elapsed


# --------------------------
# 5. 修正版主执行逻辑
# --------------------------
if __name__ == "__main__":
    # Define device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化全局计时
    total_processed = len(testset) * RUN_TIMES
    elapsed_recorder = []  # 记录各轮实际耗时
    
    accuracy = evaluate_accuracy()
    # 执行多轮推理
    for run_idx in range(1, RUN_TIMES+1):
        elapsed = process_single_run(model, testloader, DEVICE, run_idx, RUN_TIMES)
        elapsed_recorder.append(elapsed)
    
    # 精确统计（排除进度条渲染等开销）
    total_elapsed = sum(elapsed_recorder)
    avg_throughput = total_processed / total_elapsed
    
    print(f"\n[修正后汇总]")
    print(f"有效总耗时: {total_elapsed:.2f}s")
    print(f"平均吞吐量: {avg_throughput:.2f} img/s")


    final_score = calculate_final_score(accuracy,avg_throughput)
    print(f'最终得分: {final_score:.3f}')

