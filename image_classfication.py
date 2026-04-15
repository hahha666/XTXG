import torch
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 定义数据转换
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), # 随机裁剪
    transforms.RandomHorizontalFlip(),    # 50%概率水平翻转
    transforms.ToTensor(),  # 将PIL图像转换为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# 加载CIFAR-10训练集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                         shuffle=True, num_workers=4)

# 加载CIFAR-10测试集
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                        shuffle=False, num_workers=4)

# 定义类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层1：输入3通道(RGB)，输出6通道，5x5卷积核
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 池化层：2x2窗口，步长2
        self.pool = nn.MaxPool2d(2, 2)
        # 卷积层2：输入6通道，输出16通道，5x5卷积核
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层1：输入16*5*5，输出120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 全连接层2：输入120，输出84
        self.fc2 = nn.Linear(120, 84)
        # 全连接层3：输入84，输出10(对应10个类别)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 第一层卷积+ReLU+池化
        x = self.pool(F.relu(self.conv1(x)))
        # 第二层卷积+ReLU+池化
        x = self.pool(F.relu(self.conv2(x)))
        # 展平特征图
        x = x.view(-1, 16 * 5 * 5)
        # 全连接层+ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 输出层
        x = self.fc3(x)
        return x

# 实例化网络
net = Net()
net.to(device)

import torch.optim as optim

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 随机梯度下降优化器
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(10):  # 训练10个epoch
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入数据
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = net(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新权重
        optimizer.step()
        
        # 打印统计信息
        running_loss += loss.item()
        if i % 100 == 99:  # 每100个mini-batch打印一次
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():  # 不计算梯度
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test images: {100 * correct / total:.2f}%')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(128):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print(f'Accuracy of {classes[i]:5s}: {100 * class_correct[i] / class_total[i]:.2f}%')

# 保存模型参数
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

