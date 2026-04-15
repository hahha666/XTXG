import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim

# ================= 1. 硬件配置 =================
# 自动检测是否有 GPU，没有就用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的计算设备: {device}")

# ================= 2. 数据准备 =================
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    # 【修复 Bug】：CIFAR10 是 3 通道 RGB 图像，必须传 3 个均值和方差
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    # 【修复 Bug】：CIFAR10 是 3 通道 RGB 图像，必须传 3 个均值和方差
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

print("正在加载数据集...")
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
print(f"数据集加载完成，共有 {len(train_set)} 张图片，分为 {len(train_loader)} 个 Batch。")

# ================= 3. 模型配置 =================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# 【关键补充】：把庞大的模型搬运到 GPU 上
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# ================= 4. 训练循环 =================
# 【关键补充】：通知模型进入训练模式（启用 Dropout 和 BatchNorm）

best_test_loss = float('inf')  # 用于跟踪最佳测试损失

print("开始训练！")
for epoch in range(10):
    model.train()
    running_train_loss = 0.0 # 用于累计一个 Epoch 的 Loss
    train_correct = 0
    train_total = 0
    
    for step, (images, labels) in enumerate(train_loader):
        # 【关键补充】：把数据也搬运到 GPU 上，必须和模型在同一个设备
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
      
        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item()

        # 计算训练准确率
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
                
        # 【关键补充】：每算 100 个 Batch，打印一次进度！
        if step % 100 == 0:
            print(f"Epoch [{epoch+1}/10], Step [{step}/{len(train_loader)}], 当前 Loss: {loss.item():.4f}")

    model.eval()  # 切换到评估模式，关闭 Dropout 和 BatchNorm 的训练行为
    running_test_loss = 0.0 # 用于累计评估阶段的 Loss
    test_correct = 0
    test_total = 0

    with torch.no_grad():  # 评估阶段不需要计算梯度，节省显存和计算资源
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()

            # 计算测试准确率
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # 打印每个 Epoch 的平均 Loss
    epoch_train_loss = running_train_loss / len(train_loader)
    epoch_test_loss = running_test_loss / len(test_loader)

    if test_correct/test_total > best_test_loss:
        best_test_loss = test_correct/test_total
        torch.save(model.state_dict(), 'best_model.pth')

    print(f"====> Epoch {epoch+1} 结束，平均 Loss: {epoch_train_loss:.4f} <====")
    print(f"====> Epoch {epoch+1} 结束，测试 Loss: {epoch_test_loss:.4f} <====")
    print(f"====> Epoch {epoch+1} 结束，训练准确率: {100 * train_correct / train_total:.2f}% <====")
    print(f"====> Epoch {epoch+1} 结束，测试准确率: {100 * test_correct / test_total:.2f}% <====")

print("训练彻底完成！")