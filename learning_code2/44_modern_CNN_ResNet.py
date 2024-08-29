#“新添加的层如何提升神经网络的性能?”
#"设计网络的能力?"

#对于深度神经网络，如果我们能将新添加的层训练成恒等映射（identity function） f(x) = x，新模型和原模型将同样有效。
#同时，由于新模型可能得出更优的解来拟合训练数据集，因此添加层似乎更容易降低训练误差
#针对这一问题，何恺明等人提出了残差网络（ResNet） 
#核心思想：每个附加层都应该更容易地包含原始函数作为其元素之一。于是，残差块（residual blocks）便诞生了


#输入为x
#之前普通的块，里面的网络希望学出的理想映射为f(x)
#ResNet的基础架构–残差块（residual block）：需要拟合出残差映射f(x) − x
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

#定义残差块（residual blocks）
class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X                  #然后我们通过跨层数据通路，跳过这2个卷积运算，将输入直接加在最后的ReLU激活函数前
        return F.relu(Y)
    

blk = Residual(3,3) #实例化一个残差块，给定输入输出通道数
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
print(Y.shape)

blk = Residual(3,6, use_1x1conv=True, strides=2) #在增加输出通道数的同时，减半输出的高和宽
Y = blk(X)
print(Y.shape,"\n")


# ResNet模型
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = [] #一个空的列表
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

#最终的网络：ResNet‐18
#每个模块有4个卷积层（不包括恒等映射的1 × 1卷积层）。加上第一个7 × 7卷积层和最后一个全连接层，共有18层
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))


#来观察每层的输出形状
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

#训练
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())