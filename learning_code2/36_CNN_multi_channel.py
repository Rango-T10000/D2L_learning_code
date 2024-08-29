import torch
from d2l import torch as d2l

#只输入单色图像，此时就是单个输入通道，这使得我们可以将输入、卷积核和输出看作二维张量，如：输入形状为nh × nw，卷积核形状为kh × kw
#一般的图像有3个通道，每个RGB输入图像具有3 × h × w的形状。我们将这个大小为3的轴称为通道（channel）维度
#这样的话，输入和隐藏的表示都变成了三维张量
#即：输入形状为nh × nw x 3; 卷积核的形状为ci × kh × kw
#可以理解成：都有ci个通道，每个输入通道将包含形状为kh × kw的二维张量

#对于多通道输入，单通道输出来说
#计算的过程就是：
#有ci个通道，我们可以对每个通道输入的二维张量和卷积核的二维张量进行互相关运算
#再对通道求和（将ci的结果相加）得到二维张量
def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

#计算输入和卷积核的互相关
print("对于多通道输入，单通道输出为：\n",corr2d_multi_in(X, K))


#对于多通道输入，多通道输出来说：
#用ci和co分别表示输入和输出通道的数目，并让kh和kw为卷积核的高度和宽度
#我们可以为每个输出通道创建一个形状为ci × kh × kw的卷积核张量，这样卷积核的形状是co × ci × kh × kw
#计算的过程就是：
#每个输出通道先获取所有输入通道
#再以对应该输出通道的卷积核计算出结果
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

#通过将核张量K与K+1（K中每个元素加1）和K+2连接起来，构造了一个具有3个输出通道的卷积核
K = torch.stack((K, K + 1, K + 2), 0)
print("卷积核的形状",K.shape)

#计算输入和卷积核的互相关
print("对于多通道输入，多通道输出为：\n",corr2d_multi_in_out(X, K))



#1 × 1卷积，即kh = kw = 1
#毕竟，卷积的本质是有效提取相邻像素间的相关特征，而1 × 1卷积显然没有此作用
#因为使用了最小窗口， 1 × 1卷积失去了卷积层的特有能力——在高度和宽度维度上，识别相邻元素间相互作用的能力
#我们可以将1 × 1卷积层看作在每个像素位置应用的全连接层

#用全连接层实现1 × 1卷积
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6