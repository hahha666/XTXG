import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 1. 定义一个简单的神经网络模型
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


# 2. 创建模型实例
model = SimpleNN()

# 直接打印整个状态字典
print(model.state_dict())

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失函数
#optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

# 4. 假设我们有训练数据 X 和 Y
sample_num = 100
batch_size = 10
X = torch.randn(sample_num, 4)  # sample_num 个样本，4 个特征
Y_base = torch.tensor([[1.0], [0.5], [2.0], [1.5], [3.0], [2.5], [4.0], [3.5], [5.0], [4.5]])  # 10 个目标值
Y = Y_base.repeat(10, 1)

dataset = torch.utils.data.TensorDataset(X, Y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

epochs = 1000

learning_rate = 0.001
belta0 = 0.9
belta1 = 0.999
epsilon = 1e-8

m = {}
n = {}
losses = []

for param in model.parameters():
    m[param]  =torch.zeros_like(param)
    n[param]  =torch.zeros_like(param)

global_step = 0

# 5. 训练循环
for epoch in range(epochs): 
    epoch_loss = 0
    # 清空梯度（手动将所有参数的梯度置零）
    for batch_X, batch_Y in dataloader:
        global_step += 1

        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
                
        output = model(batch_X) 
        loss = criterion(output, batch_Y) 
        loss.backward()  # PyTorch 自动计算梯度，并将其存储在 param.grad 中
        epoch_loss += loss.item()
        
        # 手动更新参数 (替代 optimizer.step())
        # 必须使用 torch.no_grad()，防止 PyTorch 将更新操作也记录到计算图中
        

        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    m[param] = belta0*m[param] + (1-belta0)*param.grad
                    n[param] = belta1*n[param] + (1-belta1)*param.grad**2 

                    m_hat = m[param] / (1 - belta0**(global_step+1))
                    n_hat = n[param] / (1 - belta1**(global_step+1))               

                    step_size = learning_rate * m_hat / (torch.sqrt(n_hat) + epsilon)

                    param.sub_(step_size)

    avg_epoch_loss = epoch_loss / len(dataloader)
    losses.append(avg_epoch_loss)

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

    

simple_nn_weights_path = 'simple_nn_weights.pth'
torch.save(model.state_dict(), simple_nn_weights_path)
print(f'Model weights saved to {simple_nn_weights_path}')
        
# 打印所有线性层的权重
'''
for name, param in model.named_parameters():
    if 'fc' in name and 'weight' in name:
        print('--- '+name+' 层的权重 ---')
        print(param)
'''

# 可视化损失变化曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), losses, label='Loss') # 同步 epochs
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid()

# 拿到所有数据的最终预测结果
with torch.no_grad():
    final_output = model(X)

# 可视化预测结果与实际目标值对比
plt.figure(figsize=(8, 5))
# 这里的 range 也要和你的样本总数同步
plt.plot(range(1, sample_num + 1), Y.numpy(), 'o-', label='Actual', color='blue')
plt.plot(range(1, sample_num + 1), final_output.numpy(), 'x--', label='Predicted', color='red')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid()

plt.show()


