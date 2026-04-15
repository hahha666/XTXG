import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 必须先重新定义一模一样的网络结构
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # 输入层到隐藏层
        self.bn1 = nn.BatchNorm1d(16)  # 批归一化层
        self.fc2 = nn.Linear(16, 32)  # 隐藏层
        self.dropout = nn.Dropout(p=0.5)  # Dropout 层
        self.fc3 = nn.Linear(32, 8)  # 隐藏层到输出层
        self.fc4 = nn.Linear(8, 1)  # 输出层
    
    def forward(self, x):

        x = self.fc1(x)  # 输入层到隐藏层
        x = self.bn1(x)  # 批归一化
        x = F.gelu(x)

        x = self.fc2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.dropout(x)

        x = torch.tanh(self.fc3(x))

        x = self.fc4(x)

        return x

# ==========================================
# 推理阶段 (Inference)
# ==========================================

# 2. 实例化模型，并加载之前保存的权重
model = SimpleNN()
# 假设你之前保存的文件名叫 'simple_nn_weights.pth'
model.load_state_dict(torch.load('simple_nn_weights.pth'))

# 3. 切换到评估模式 (极其重要！)
model.eval()

# 4. 接收或生成全新的输入数据
# 只要保证数据的特征维度与模型输入层一致（这里是 4 个特征）即可。
# 这里我们模拟接收到 10 个全新的随机样本 (Batch Size = 10)
new_X = torch.randn(10, 4) 
print("=== 接收到的新输入数据 (5个样本) ===")
print(new_X)

# 5. 执行推理运算
# 必须使用 torch.no_grad()，因为推理阶段不需要反向传播，
# 关闭梯度追踪可以大幅节省显存并提升计算速度。
with torch.no_grad():
    predictions = model(new_X)

print("\n=== 模型的预测输出 ===")
print(predictions)


