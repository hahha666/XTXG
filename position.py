import torch
import math
import matplotlib.pyplot as plt

def get_positional_encoding(max_seq_length, d_model):
    """生成位置编码矩阵"""
    # 1. 初始化全零矩阵
    pe = torch.zeros(max_seq_length, d_model)
    
    # 2. 生成位置列向量 (max_seq_length, 1)
    position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
    
    # 3. 计算频率缩放因子 (d_model/2,)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    
    # 4. 广播乘法并交替赋值
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

# 设置参数：序列长度50，特征维度64
seq_len = 50
d_model = 64

# 获取位置编码矩阵
pe_matrix = get_positional_encoding(seq_len, d_model)

# 转换为 NumPy 格式以便 matplotlib 绘图
pe_numpy = pe_matrix.numpy()

# 设置画布大小
plt.figure(figsize=(10, 8))

# 绘制热力图 (使用 RdBu 颜色映射，从红到蓝表示 -1 到 1)
im = plt.imshow(pe_numpy, cmap='RdBu', aspect='auto', vmin=-1.0, vmax=1.0)

# 添加颜色条
plt.colorbar(im, label='Positional Encoding Value')

# 设置图表标题和坐标轴标签
plt.title(f'Transformer Positional Encoding (seq_len={seq_len}, d_model={d_model})')
plt.xlabel('Embedding Dimension Index (i)')
plt.ylabel('Sequence Position Index (pos)')

# 显示图表
plt.tight_layout()
plt.show()