#稠密连接网络（DenseNet） 某种程度上是ResNet的逻辑扩展。
#在跨层连接上，不同于ResNet中将输入与输出相加，稠密连接网络（DenseNet）在通道维上连结输入与输出
#DenseNet这个名字由变量之间的“稠密连接”而得来，最后一层与之前的所有层紧密相连。
#稠密网络主要由2部分构成：稠密块（dense block）和过渡层（transition layer）
#稠密块（dense block）: 定义如何连接输入和输出
#过渡层（transition layer）: 控制通道数量，使其不会太复杂

import torch
from torch import nn
from d2l import torch as d2l


#定义一个卷积块
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

#定义一个稠密块（dense block），由多个卷积块组成
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X
    
#测试稠密块，卷积块的通道数控制了输出通道数相对于输入通道数的增长，因此也被称为增长率（growth rate）
blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8) #输入的通道数为3
Y = blk(X)                  #经过稠密块后的输出的通道数为23
print(Y.shape)

#由于每个稠密块都会带来通道数的增加，使用过多则会过于复杂化模型。所以用过渡层可以用来控制模型复杂度。
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),#通过1 × 1卷积层来减小通道数
        nn.AvgPool2d(kernel_size=2, stride=2))                 #并使用步幅为2的平均汇聚层减半高和宽


blk = transition_block(23, 10)
Y = blk(Y)
print(Y.shape)

#构造DenseNet模型
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# num_channels为当前的通道数
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 上一个稠密块的输出通道数
    num_channels += num_convs * growth_rate
    # 在稠密块之间添加一个转换层，使通道数量减半
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))


lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
