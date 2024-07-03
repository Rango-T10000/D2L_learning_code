#softmax回归的简洁实现
#通过深度学习框架的高级API能够使实现

import torch
from torch import nn
from d2l import torch as d2l

#读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#初始化模型参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

#定义模型
#在Sequential中添加一个带有10个输出的全连接层
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状，即784个输入，10个输出
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net.apply(init_weights)

#定义损失函数：交叉熵
loss = nn.CrossEntropyLoss(reduction='none')

#定义优化算法：sgd
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

#训练
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)