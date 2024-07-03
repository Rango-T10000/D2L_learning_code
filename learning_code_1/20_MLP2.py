#多层感知机的从零开始实现
#继续使用Fashion‐MNIST图像分类数据集
#Fashion‐MNIST中的每个图像由 28 × 28 = 784个灰度像素值组成。所有图像共分为10个类别。
#将每个图像视为具有784个输入特征和10个类的简单分类数据集


import torch
from torch import nn
from d2l import torch as d2l

#读取出数据集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#初始化模型参数
#将每个图像视为具有784个输入特征和10个类的简单分类数据集
#我们将实现一个具有单隐藏层的多层感知机，它包含256个隐藏单元
#通常，我们选择2的若干次幂作为层的宽度。层数和隐藏单元的个数也是超参数
#注意，对于每一层我们都要记录一个权重矩阵和一个偏置向量
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]


#激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


#定义模型
def net(X):
    X = X.reshape((-1, num_inputs)) #把28*28的图片转为1*784
    H = relu(X@W1 + b1)             # 这里“@”代表矩阵乘法,输入层经过运算通过激活函数作为隐藏层的输入
    return (H@W2 + b2)              # 隐藏层直接运算，不通过激活函数

#定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')

#参数更新算法：选sgd
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)

#训练
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

#预测/推理
d2l.predict_ch3(net, test_iter)




