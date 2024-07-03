#暂退法(drop out)
#希望模型深度挖掘特征，即将其权重分散到许多特征中，而不是过于依赖少数潜在的虚假关联
#与线性模型不同，神经网络并不局限于单独查看每个特征，而是学习特征之间的交互。
#经典泛化理论认为，为了缩小训练和测试性能之间的差距，应该以简单的模型为目标
#简单性的另一个角度是平滑性，即函数不应该对其输入的微小变化敏感。
#暂退法:在计算后续层之前向网络的每一层注入噪声。因为当训练一个有多层的深层网络时，注入噪声只会在输入‐输出映射上增强平滑性。
##暂退法(drop out)：在前向传播过程中，计算每一内部层的同时丢弃一些神经元，从表面上看是在训练过程中丢弃（dropout）一些神经元，在计算下一层之前将当前层中的一些节点置零
#如果通过许多不同的暂退法遮盖后得到的预测结果都是一致的，那么我们可以说网络发挥更稳定


#暂退法可以避免过拟合

import torch
from torch import nn
from d2l import torch as d2l

#深度学习框架的高级API简洁实现
#在每个全连接层之后添加一个Dropout层，将暂退概率作为唯一的参数传递给它的构造函数。
#即以p的概率将隐藏单元置为零
#设置暂退概率
dropout1, dropout2 = 0.2, 0.5

#设置超参数
num_epochs, lr, batch_size = 10, 0.5, 256

#读取数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#定义网络
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

#定义Loss
loss = nn.CrossEntropyLoss(reduction='none')

#定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=lr)

#训练
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)