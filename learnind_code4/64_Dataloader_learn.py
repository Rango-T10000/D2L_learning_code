#我们以一个简单的例子来演示如何使用 DataLoader 加载数据并训练一个神经网络
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn


# 假设我们有两个简单的张量数据
X = torch.randn(100, 3)  # 100个样本，每个样本有3个特征
y = torch.randn(100, 1)  # 100个标签

# 将数据封装成一个数据集
dataset = TensorDataset(X, y)

# 定义DataLoader，batch_size=10 表示每个批次取10个样本
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(3, 1)  # 3个输入特征，1个输出
    
    def forward(self, x):
        return self.fc(x)

model = SimpleNet()

import torch.optim as optim

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(5):  # 假设训练5个epoch
    for batch_x, batch_y in dataloader:
        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/5], Loss: {loss.item():.4f}')
