#隐变量自回归模型，P (xt | xt−1, . . . , x1) ≈ P (xt | ht−1)
#ht−1是隐状态（hidden state），也称为隐藏变量（hidden variable），它存储了到时间步t − 1的序列信息
#基于当前输入xt和先前隐状态ht−1 来计算时间步t处的任何时间的隐状态：ht = f(xt, ht−1)
#循环神经网络（recurrent neural networks， RNNs）是具有隐状态的神经网络

#普通的MLP没有隐变量，其隐藏层的计算是：H = ϕ(XWxh + bh)
#循环层（recurrent layer）有隐变量，循环层的计算是：Ht = ϕ(XtWxh + Ht−1Whh + bh)
#比原来多了一项：Ht−1Whh，这说的是相邻时间步的隐藏变量Ht和 Ht−1之间的关系
#这些变量捕获并保留了序列直到其当前时间步的历史信息，就如当前时间步下神经网络的状态或记忆，
#因此这样的隐藏变量被称为隐状态（hidden state）
#基于循环计算的隐状态神经网络被命名为 循环神经网络（recurrent neural network）


#注意：隐状态中 XtWxh + Ht−1Whh的计算，相当于Xt和Ht−1的拼接与Wxh和Whh的拼接的矩阵乘法。(废话)
#下面演示：
import torch
from d2l import torch as d2l

X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
print(torch.matmul(X, W_xh) + torch.matmul(H, W_hh))
print(torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0)))


#任务：我们的目标是根据过去的和当前的词元预测下一个词元，因此我们将原始序列移位一个词元作为标签

#基于循环神经网络的字符级语言模型（character‐level language model）
#使用当前的和先前的字符预测下一个字符

#在训练过程中，我们对每个时间步的输出层的输出进行softmax操作，
# 然后利用交叉熵损失计算模型输出和标签之间的误差


#如何度量语言模型的质量：困惑度（Perplexity）
#可以通过一个序列中所有的n个词元的交叉熵损失的平均值来衡量
#困惑度（Perplexity）是上面这个值的exp值







