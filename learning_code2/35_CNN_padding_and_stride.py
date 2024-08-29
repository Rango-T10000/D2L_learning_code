import torch
from torch import nn
from d2l import torch as d2l
#填充和步幅可用于有效地调整数据的维度
#填充可以增加输出的高度和宽度
#步幅可以减小输出的高和宽


#卷积的输出形状取决于输入形状和卷积核的形状
#假设输入形状为nh × nw，卷积核形状为kh × kw
#输出形状将是(nh − kh + 1) × (nw − kw + 1)

#在应用多层卷积时，我们常常丢失边缘像素
#解决这个问题的简单方法即为填充（padding）：在输入图像的边界填充元素（通常填充元素是0）
#如果我们添加ph行填充（大约一半在顶部，一半在底部）和pw列填充（左侧大约一半，右侧一半），
#则输出形状将为:(nh − kh + ph + 1) × (nw − kw + pw + 1)
#在许多情况下，我们需要设置ph = kh − 1和pw = kw − 1，使输入和输出具有相同的高度和宽度。



# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])

# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1) #我们创建一个高度和宽度为3的二维卷积层，并在所有侧边填充1个像素
X = torch.rand(size=(8, 8))                        #定高度和宽度为8的输入
print("输出的shape:",comp_conv2d(conv2d, X).shape)


#当卷积核的高度和宽度不同时,如5x3，可以填充不同的高度和宽度，使输出和输入具有相同的高度和宽度。根据上面的公式可以算填充的维度
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print("输出的shape:",comp_conv2d(conv2d, X).shape)


#计算互相关时，卷积窗口从输入张量的左上角开始，向下、向右滑动。
#有时候为了高效计算或是缩减采样次数，卷积窗口可以跳过中间位置，每次滑动多个元素
#将每次滑动元素的数量称为步幅（stride），步幅可以减小输出的高和宽，例如输出的高和宽仅为输入的高和宽的1/n
#例：垂直步幅为3，水平步幅为2的二维互相关运算：卷积窗口分别向下滑动三行和向右滑动两列
#当垂直步幅为sh、水平步幅为sw时，输出的形状：⌊(nh − kh + ph + sh)/sh⌋ × ⌊(nw − kw + pw + sw)/sw⌋
#如果我们设置了ph = kh − 1和pw = kw − 1，则输出形状将简化为⌊(nh + sh − 1)/sh⌋ × ⌊(nw + sw − 1)/sw⌋

#我们将高度和宽度的步幅设置为2，从而将输入的高度和宽度减半
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print("输出的shape:",comp_conv2d(conv2d, X).shape)

#在实践中，我们很少使用不一致的步幅或填充，也就是说，我们通常有ph = pw和sh = sw