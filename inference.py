import torch
import torchvision
import torchvision.transforms as transforms
import time
import sys
from tqdm import tqdm
from torch.utils.data import ConcatDataset
from modules.resnet_18 import ResNet18 as ResNet18_optim
from modules.resnet_18_baseline import ResNet18 as ResNet18_baseline

# --------------------------
# 参数配置
# --------------------------
MODEL_PATH = "./pytorch/model/net_123.pth"
DATA_ROOT = "./pytorch/data"
BATCH_SIZE = 256
NUM_WORKERS = 4
RUN_TIMES = 5  # 新增运行次数参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROGRESS_UNIT = "img"

# --------------------------
# 模型加载（保持原样）
# --------------------------
def load_model(model_path, model_type='baseline'):
    # 根据类型选择模型类
    if model_type == 'optim':
        model = ResNet18_optim()
    elif model_type == 'baseline':
        model = ResNet18_baseline()
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Choose ['optim', 'baseline']")

    checkpoint = torch.load(model_path, map_location=DEVICE,weights_only=False)
    model.load_state_dict(checkpoint)
    model = model.to(DEVICE).eval()
    return model
    
model_optim = load_model(MODEL_PATH, model_type='optim')
model_baseline = load_model(MODEL_PATH, model_type='baseline')

def calculate_final_score(accuracy, throughput_optim, throughput_baseline):
    accuracy_weight = 4
    speed_weight = 15
    
    normalized_throughput = throughput_optim / throughput_baseline

    final_score = accuracy_weight * (accuracy / 10) + speed_weight * normalized_throughput
    return min(final_score, 100)

# Evaluate accuracy
def evaluate_accuracy():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model_optim(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.3f}%')
    if accuracy < 87:
        print("[EXIT] Accuracy is intolerable, terminating program")
        sys.exit(1) 

    return accuracy

def process_single_run(model, dataloader, device, run_num, total_runs):
    total_images = len(dataloader.dataset)
    start_time = time.perf_counter()
    
    progress_bar = tqdm(
        dataloader,
        desc=f"轮次 {run_num:02d}/{total_runs}",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",  # 保持postfix占位符
        unit='img',  # 关键修正点：使用字符串单位
        unit_scale=BATCH_SIZE,  # 在此处设置缩放比例
        dynamic_ncols=True
    )

    # 预热逻辑保持不变
    with torch.no_grad():
        warmup_tensor = torch.randn(BATCH_SIZE,3,32,32).to(device)
        for _ in range(3):
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
            #progress_bar.set_postfix_str(f"{current_throughput:.1f} img/s")  # 保持字符串格式

    # 最终计算
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    run_elapsed = time.perf_counter() - run_start
    final_throughput = total_images / run_elapsed

    progress_bar.set_postfix_str(f"{final_throughput:.1f} img/s (最终)")
    progress_bar.close()
    return run_elapsed


# --------------------------
# 数据加载（调整为原始测试集）
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

if __name__ == "__main__":
    # Define device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化全局计时
    processed = len(testset)
    total_processed = processed * RUN_TIMES
    elapsed_recorder = []  # 记录各轮实际耗时
    
    accuracy = evaluate_accuracy()
    
    # 执行多轮推理
    for run_idx in range(1, RUN_TIMES+1):
        elapsed = process_single_run(model_optim, testloader, DEVICE, run_idx, RUN_TIMES)
        elapsed_recorder.append(elapsed)

    total_elapsed = sum(elapsed_recorder)
    avg_throughput = total_processed / total_elapsed
    print(f"手写算子平均吞吐量: {avg_throughput:.2f} img/s\n")

    baseline_elapsed = process_single_run(model_baseline, testloader, DEVICE, 1, 1)
    baseline_throughput = processed / baseline_elapsed
    print(f"基准吞吐量: {baseline_throughput:.2f} img/s\n")
    
    final_score = calculate_final_score(accuracy,avg_throughput, baseline_throughput)
    print(f'最终得分: {final_score:.3f}')

