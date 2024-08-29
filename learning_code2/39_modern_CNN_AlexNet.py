#虽然LeNet在小数据集上取得了很好的效果，但是在更大、更真实的数据集上训练卷积神经网络的性能和可行性还有待研究
#在上世纪90年代初到2012年之间的大部分时间里，神经网络往往被其他机器学习方法超越，如支持向量机（support vector machines）
#训练神经网络的一些关键技巧仍然缺失，包括:
# 启发式参数初始化
# 随机梯度下降的变体
# 非挤压激活函数
# 有效的正则化技术

#经典机器学习方法：在2012年前，图像特征都是机械地计算出来的。
#神经网络：训练端到端（从像素到分类结果）系统， 特征本身应该被学习

#他们认为特征本身应该被学习
#2012年，Alex Krizhevsky、 Ilya Sutskever和Geoff Hinton提出了一种新的卷积神经网络变体AlexNet
#是从浅层网络到深层网络的关键一步，还用了Dropout、 ReLU和预处理
import torch
from torch import nn
from d2l import torch as d2l

#精简版的AlexNet
net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10))
print(net)


#构造一个高度和宽度都为224的单通道数据作为输入
#1个样本，1个输入通道，224x224
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)


#为了快速举例，不用ImageNet数据集
#还是用Fashion‐MNIST数据集，Fashion‐MNIST图像的分辨率（28 × 28像素）低于ImageNet图像。
#所以强行给增加到224 × 224（ImageNet图像），用d2l.load_data_fashion_mnist函数中的resize参数执行此调整
#相当于“裁剪”，往大了剪
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

#训练
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())