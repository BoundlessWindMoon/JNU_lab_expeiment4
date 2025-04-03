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
MODEL_DTYPE = "FP16"
BATCH_SIZE = 256
NUM_WORKERS = 4
RUN_TIMES = 5  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROGRESS_UNIT = "img"

# --------------------------
# 模型加载（保持原样）
# --------------------------
def load_model(model_path, model_type='baseline', model_dtype='FP32'):
    
    if model_type == 'optim':
        model = ResNet18_optim()
    elif model_type == 'baseline':
        model = ResNet18_baseline()
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Choose ['optim', 'baseline']")

    checkpoint = torch.load(model_path, map_location=DEVICE,weights_only=False)
    model.load_state_dict(checkpoint)
    dtype = torch.half if model_dtype=="FP16" else torch.float32
    model = model.to(DEVICE, dtype=dtype)
    model.eval()
    return model
    
model_optim = load_model(MODEL_PATH, model_type='optim', model_dtype=MODEL_DTYPE)
model_baseline = load_model(MODEL_PATH, model_type='baseline', model_dtype="FP32")

def calculate_final_score(accuracy, throughput_optim, throughput_baseline):
    accuracy_weight = 4
    speed_weight = 4.5
    
    normalized_throughput = throughput_optim / throughput_baseline

    final_score = accuracy_weight * (accuracy / 10) + speed_weight * normalized_throughput
    return min(final_score, 100)

def evaluate_accuracy():
    correct = 0
    total = 0
    with torch.no_grad():
        model_dtype = next(model_optim.parameters()).dtype
        for data in testloader:
            images, labels = data
            images, labels = images.to(device=device, dtype=model_dtype), labels.to(device)
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

def process_single_run(model, dataloader, device, run_num, total_runs,model_dtype='FP32'):
    total_images = len(dataloader.dataset)
    start_time = time.perf_counter()
    
    progress_bar = tqdm(
        dataloader,
        desc=f"轮次 {run_num:02d}/{total_runs}",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",  
        unit='img', 
        unit_scale=BATCH_SIZE,  
        dynamic_ncols=True
    )

    with torch.no_grad():
        #warmup_tensor = torch.randn(BATCH_SIZE,3,32,32).to(device)
        #warmup_tensor = torch.randn(BATCH_SIZE,3,32,32, dtype=torch.half).to(device)
        warmup_dtype = torch.half if model_dtype=="FP16" else torch.float32
        warmup_tensor = torch.randn(BATCH_SIZE,3,32,32, dtype=warmup_dtype).to(device)
        for _ in range(3):
            _ = model(warmup_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    run_start = time.perf_counter()
    with torch.no_grad():
        for images, _ in progress_bar:
            #images = images.half().to(device, non_blocking=True)
            images = images.to(device=device, dtype=warmup_dtype, non_blocking=True)
            _ = model(images)
            
            current_time = time.perf_counter() - run_start
            current_throughput = (progress_bar.n * BATCH_SIZE) / current_time

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

import matplotlib.pyplot as plt
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed = len(testset)
    total_processed = processed * RUN_TIMES
    elapsed_recorder = []
    
    accuracy = evaluate_accuracy()
    
    for run_idx in range(1, RUN_TIMES+1):
        elapsed = process_single_run(model_optim, testloader, DEVICE, run_idx, RUN_TIMES, model_dtype=MODEL_DTYPE)
        elapsed_recorder.append(elapsed)

    total_elapsed = sum(elapsed_recorder)
    avg_throughput = total_processed / total_elapsed
    print(f"手写算子平均吞吐量: {avg_throughput:.2f} img/s\n")  
    
    baseline_elapsed = process_single_run(model_baseline, testloader, DEVICE, 1, 1,model_dtype="FP32")
    baseline_throughput = processed / baseline_elapsed
    print(f"基准吞吐量: {baseline_throughput:.2f} img/s\n")  
    
    final_score = calculate_final_score(accuracy, avg_throughput, baseline_throughput)
    print(f'最终得分: {final_score:.3f}')  
    
    optim_throughputs = []
    for run_idx in range(RUN_TIMES):
        current_processed = processed * (run_idx + 1)
        current_throughput = current_processed / sum(elapsed_recorder[:run_idx+1])
        optim_throughputs.append(current_throughput)
    
    speedup_ratio = [t / baseline_throughput for t in optim_throughputs]
    run_numbers = list(range(1, RUN_TIMES+1))
    
    plt.figure(figsize=(10, 5))
    plt.plot(run_numbers, speedup_ratio, 
             marker='o', label='Optimized', color='#E6550D')
    plt.axhline(y=1, color='#3182BD', linestyle='--', label='Baseline')
    
    plt.title(f'Throughput Acceleration (Final: {final_score:.2f} pts)')
    plt.xlabel('Run Times')
    plt.ylabel('Speedup Ratio')
    plt.xticks(run_numbers)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    plt.show()

