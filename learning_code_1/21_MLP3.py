#多层感知机的简洁实现
#通过高级API更简洁地实现多层感知机
#第一层是隐藏层，它包含256个隐藏单元，并使用了ReLU激活函数。第二层是输出层

import torch
from torch import nn
from d2l import torch as d2l

#定义模型：nn.Sequential()的参数从左至右就是一层层网络的小结构
#先把28*28的输入展平，再隐藏层(全连接)，再过激活函数，再输出层(全连接)
#注意，线性神经网络就是线性层，就是全连接层：nn.Linear（）
net = nn.Sequential(nn.Flatten(),nn.Linear(784, 256),nn.ReLU(),nn.Linear(256, 10))

#参数初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        print("全连接层的参数：",m.weight.shape)
    

net.apply(init_weights)

#定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')

#超参数设置
batch_size, lr, num_epochs = 256, 0.1, 10

#参数更新算法
updater = torch.optim.SGD(net.parameters(), lr=lr)

#读取数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#训练
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)



