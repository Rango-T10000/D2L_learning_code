#LeNet、 AlexNet和VGG都有一个共同的设计模式：
#通过一系列的卷积层与汇聚层来提取空间结构特征；然后通过全连接层对特征的表征进行处理
#AlexNet和VGG对LeNet的改进主要在于如何扩大和加深这两个模块

#如何在这个过程的早期使用全连接层，又不完全放弃表征的空间结构
#网络中的网络（NiN）提供了一个非常简单的解决方案：在每个像素的通道上分别使用多层感知机，即1x1卷积核

#如34_CNN_edge_detection.py中的二维卷积层conv2d
#用的是四维输入和输出格式（批量大小、通道、高度、宽度）
#即卷积层的输入和输出由四维张量组成，张量的每个轴分别对应样本数量/batch_aize、通道、高度和宽度
# 第一个轴（样本轴）：表示输入/输出的样本个数。每个样本可以是一张图像或其他数据样本。
# 第二个轴（通道轴）：表示输入/输出的通道数。对于彩色图像，通道数为3（红、绿、蓝），对于灰度图像，通道数为1。
# 第三个轴（高度轴）：表示输入/输出的图像的高度。
# 第四个轴（宽度轴）：表示输入/输出的图像的宽度。

#MLP的输入和输出通常是分别对应于样本和特征的二维张量，即（样本，特征），就是表格




#NiN的想法是在每个像素位置（针对每个高度和宽度）应用一个全连接层,可以将其视为1 × 1卷积层
#从另一个角度看，即将空间维度中的每个像素视为单个样本，将通道维度视为不同特征（feature）
import torch
from torch import nn
from d2l import torch as d2l

#定义一个NiN块（同样是块的设计思想，就是搭积木，这个积木的大小通过参数调整）
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

#定义网络：NiN和AlexNet之间的一个显著区别是NiN完全取消了全连接层。
#NiN使用一个NiN块，其输出通道数等于标签类别的数量。最后放一个全局平均汇聚层（global average pooling layer），生成一个对数几率（logits）。
#显著减少了模型所需参数的数量
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为(批量大小,10)
    nn.Flatten())

#构建一个高度和宽度为224的单通道数据样本，4维张量作为输入
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:   \t', X.shape)

#训练
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())