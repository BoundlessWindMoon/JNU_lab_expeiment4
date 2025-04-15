import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import os
from conv_layer import Conv2d
import torch.ao.quantization as quantization
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            # 替换第一个卷积层
            Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            # 替换第二个卷积层
            Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                # 替换 shortcut 中的卷积层
                Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.inchannel = 64
        self.conv = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        out = self.conv(x)
        out = self.batchnorm(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.dequant(out)
        return out

def ResNet18():

    return ResNet(ResidualBlock)
# Define device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the test set transformations
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load the CIFAR10 test dataset
testset = torchvision.datasets.CIFAR10(root='../pytorch/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8, drop_last=False)

# Load the trained model
model_path = '../pytorch/model/net_005.pth'  # Replace with the path to your trained model
net = ResNet18().to(device)
net.load_state_dict(torch.load(model_path, weights_only=True))
net.eval()  # Set the model to evaluation mode
qconfig = torch.ao.quantization.get_default_qconfig('x86')
qconfig = quantization.QConfig(
    activation=quantization.MinMaxObserver.with_args(dtype=torch.quint8, quant_min=0, quant_max=255),
    weight=quantization.MinMaxObserver.with_args(dtype=torch.qint8, quant_min=-128, quant_max=127)
)
net.qconfig = qconfig
# model_fp32_fused = torch.ao.quantization.fuse_modules(net)
model_fp32_prepared = torch.ao.quantization.prepare(net)
input_fp32 = torch.randn(32, 3, 64, 64).to(device)
model_fp32_prepared(input_fp32)
model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
print("-"*100)
print(model_int8)
print("-"*100)
# Evaluate accuracy
def evaluate_accuracy():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model_int8(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.3f}%')
    return accuracy

# Measure inference speed
def measure_speed():
    # Measure the time it takes for a single forward pass
    start_time = time.time()
    with torch.no_grad():
        for data in testloader:
            images, _ = data
            images = images.to(device)
            outputs = model_int8(images)
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_batch = total_time / len(testloader)
    throughput = len(testloader.dataset) / total_time  # images per second
    print(f'Inference Time per Batch: {avg_time_per_batch:.6f} seconds')
    print(f'Throughput: {throughput:.2f} images/second')
    return throughput

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

if __name__ == "__main__":
    # Evaluate accuracy
    accuracy = evaluate_accuracy()

    # Measure inference speed
    throughput = measure_speed()

    # Calculate the final score
    final_score = calculate_final_score(accuracy, throughput)
    print(f'Final Score: {final_score:.3f}')
