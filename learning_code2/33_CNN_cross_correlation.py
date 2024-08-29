import torch
from torch import nn
from d2l import torch as d2l

#MLP通常用于处理低维数据，例如一维向量或表格数据。
#对于高维数据，需要将其展平为一维向量后才能输入MLP。这种展平操作会忽略数据在原始空间中的空间结构，因为MLP无法显式地学习数据的空间特征。

#卷积神经网络（CNN）在处理高维数据时更具优势。CNN引入了卷积层和池化层，能够有效地处理具有空间结构的数据
#CNN的卷积层通过卷积操作在局部感受野内提取特征，并保留了数据的空间关系。
#池化层则可以对特征图进行空间下采样，减少参数数量，并提取出更加显著的特征。
#CNN的输入是高维数据，如二维或三维张量


def corr2d(X, K):  #@save
    """计算二维互相关运算"""
    h, w = K.shape #K的长宽
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))     #输出大小等于输入大小nh × nw减去卷积核大小kh × kw,即(nh − kh + 1) × (nw − kw + 1)
    
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K).shape)
print(corr2d(X, K))


#基于上面定义的corr2d函数实现二维卷积层
#卷积层对输入和卷积核权重进行互相关运算，并在添加标量偏置之后产生输出。
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


#高度和宽度分别为h和w的卷积核可以被称为h × w卷积或h × w卷积核。
#我们也将带有h × w卷积核的卷积层称为h × w卷积层


