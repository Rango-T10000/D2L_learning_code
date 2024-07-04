#从单个神经元的角度思考问题，发展到整个层，现在又转向块，重复层的模式
#使用块的想法首先出现在牛津大学的视觉几何组（visual geometry group）的VGG网络中

#一个VGG块由一系列卷积层组成，后面再加上用于空间下采样的最大汇聚层。

import torch
from torch import nn
from d2l import torch as d2l


#我们定义了一个名为vgg_block的函数来实现一个VGG块
#卷积层的数量num_convs、输入通道的数量in_channels 和输出通道的数量out_channels
#一个VGG块由若干个卷积层组成，后面再加上最大汇聚层
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

#与AlexNet、 LeNet一样， VGG网络可以分为两部分：第一部分主要由卷积层和汇聚层组成，第二部分由全连接层组成
#举例：VGG‐11网络(使用8个卷积层和3个全连接层)
#超参数变量conv_arch。该变量指定了每个VGG块里卷积层个数和输出通道数
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

#定义VGG-11网络
def vgg(conv_arch):
    conv_blks = []   #一个卷积块空列表，后面往里叠加vgg块
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

#实例化这个网络，实例是net
net = vgg(conv_arch)
print(net,"\n")

#构建一个高度和宽度为224的单通道数据样本
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)  #每次只通过net的一层，来观察输出
    print(blk.__class__.__name__,'output shape:\t',X.shape)   


#训练，这里把网络规模缩小一下，将conv_arch 中的输出通道数缩小为原来的 1/4，以减少模型的参数量和计算量
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
#small_conv_arch 变量使用了列表推导（list comprehension）的技术，通过对 conv_arch 进行迭代和计算，创建了一个新的列表。
#对conv_arch中的每个元组进行遍历，并将元组中每一个pair(如（1，32）)的第一个元素保持不变，而将第二个元素除以 ratio，
#然后将结果作为新的元组添加到 small_conv_arch 中

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
