#批量规范化（batchnormalization）: 一种流行且有效的技术，可持续加速深层网络的收敛速度
#批标准化是一种常用的正则化技术，用于加速深度神经网络的训练过程并提高模型的收敛性和泛化能力。

#批量规范化应用于单个可选层（也可以应用到所有层），其原理如下：
#在每次训练迭代中，我们首先规范化输入，即通过减去其均值并除以其标准差，其中两者均基于当前小批量处理。
# 接下来，我们应用比例系数和比例偏移。

#只有使用足够大的小批量，批量规范化这种方法才是有效且稳定的
#在应用批量规范化时，批量大小的选择很重要
#批量规范化在训练模式和预测模式下的行为通常不同
#在实践中是引入一个批量规范化层在网络中，数学上用批量规范化的运算符为BN表示

#对于MLP,将批量规范化层置于全连接层中的仿射变换和激活函数之间
#对于卷积层，在卷积层之后和非线性激活函数之前应用批量规范化

import torch
from torch import nn
from d2l import torch as d2l

#从零实现批量规范化层
#......省略


#使用深度学习框架中定义的BatchNorm实现，在LeNet网络中加入批量规范化层
#nn.BatchNorm2d()的参数是其输入维度，与前一层的输出维度保持一致
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))


lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())





