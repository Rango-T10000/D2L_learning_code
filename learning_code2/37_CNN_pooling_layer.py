#通常当我们处理图像时，我们希望逐渐降低隐藏表示的空间分辨率、聚集信息，
#这样随着我们在神经网络中层叠的上升，每个神经元对其敏感的感受野（输入）就越大

#汇聚（pooling）层/池化层，它具有双重目的：降低卷积层对位置的敏感性，同时降低对空间降采样表示的敏感性
#与卷积层类似，汇聚层运算符由一个固定形状的窗口组成，该窗口根据其步幅大小在输入的所有区域上滑动，
# 为固定形状窗口（有时称为汇聚窗口）遍历的每个位置计算一个输出。
#汇聚层不包含参数。相反，池运算是确定性的

#输入通过池化层/汇聚层的计算过程为：
#汇聚窗口/池化窗口从输入张量的左上角开始，从左往右、从上往下的在输入张量内滑动
#在汇聚窗口到达的每个位置，它计算该窗口中输入子张量的最大值或平均值。
#如果计算的是汇聚窗口中所有元素的最大值，则该层称之为最大汇聚层（maximum pooling）
#如果计算的是汇聚窗口中所有元素的平均值，则该层称之为平均汇聚层（average pooling）

#汇聚窗口形状为p × q的汇聚层称为p × q汇聚层，汇聚操作称为p × q汇聚

#例子：对象边缘检测示例
import torch
from torch import nn
from d2l import torch as d2l

#实例：池化层参与前向传播计算的实现函数
def pool2d(X, pool_size, mode = None):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X, (2, 2), 'max'))   #用最大汇聚层（maximum pooling）
print(pool2d(X, (2, 2), 'avg'))   #用平均汇聚层（average pooling）


#与卷积层一样，汇聚层也可以通过 填充和步幅 改变输出形状

#对于单通道输入：
#例：一个输入张量X，它有四个维度，其中样本数和通道数都是1
#第一个维度1表示批次大小（batch size），就是样本数，在这里是一个样本。
#第二个维度1表示通道数（channels），在这里只有一个通道。
#第三个和第四个维度4表示高度和宽度，即图像的尺寸。
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print("单通道输入：",X)

#默认情况下，深度学习框架中的步幅与汇聚窗口的大小相同
#例：用深度学习框架中内置的二维最大汇聚层，汇聚窗口的大小为3
pool2d = nn.MaxPool2d(3)
print(pool2d(X))

#填充和步幅可以手动设定
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))

#设定一个任意大小的矩形汇聚窗口，如：(2, 3)，并分别设定填充和步幅的高度和宽度
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
print(pool2d(X))

#对于多通道输入：
#汇聚层在每个输入通道上单独运算，而不是像卷积层一样在通道上对输入进行汇总
#所以，汇聚层的输出通道数与输入通道数相同
#例子：在通道维度上连结张量X和X + 1，以构建具有2个通道的输入
X = torch.cat((X, X + 1), 1)
print("多通道输入，2个通道的输入：\n",X)

#定义池化层
pool2d = nn.MaxPool2d(3, padding=1, stride=2)

#输入经过池化层运算
print("多通道输入的输出：\n",pool2d(X))
