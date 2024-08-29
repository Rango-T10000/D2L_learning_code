#卷积层的一个简单应用：通过找到像素变化的位置，来检测图像中不同颜色的边缘。
import torch
from torch import nn
from d2l import torch as d2l

def corr2d(X, K):  #@save
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))     #输出大小等于输入大小nh × nw减去卷积核大小kh × kw,即(nh − kh + 1) × (nw − kw + 1)
    
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))


#首先，我们构造一个6 × 8像素的黑白图像,中间四列为黑色（0），其余像素为白色（1）
X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)

#构造卷积核K：高度为1、宽度为2，这里卷积核的参数你自己给定的是1和-1
K = torch.tensor([[1.0, -1.0]])

#对参数X（输入）和K（卷积核）执行互相关运算
#输出Y中的1代表从白色到黑色的边缘， ‐1代表从黑色到白色的边缘，其他情况的输出为0
Y = corr2d(X, K)
print(Y)

#现在我们将输入的二维图像转置,再进行如上的互相关运算
#之前检测到的垂直边缘消失了。不出所料，这个卷积核K只可以检测垂直边缘，无法检测水平边缘
Z = corr2d(X.t(), K)
print(Z)




#是否可以通过仅查看“输入‐输出”对来学习由X生成Y的卷积核？废话，卷积核就是卷积层的权重，这个参数当然是从数据中学到的
#这个跟之前的MLP的隐藏层的参数一码事

# 构造一个二维卷积层，它具有1个输入通道，1个输出通道（就是单色图像），和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')
#这里卷积核的参数是学习到的，非常接近之前给定的1，-1
print(conv2d.weight.data.reshape((1, 2)))

