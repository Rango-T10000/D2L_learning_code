#有了注意力机制之后，我们将词元序列(tokens)输入注意力池化(attention计算)中，以便同一组词元同时充当查询、键和值。(Q,K,V)
# 具体来说，每个查询Q都会关注所有的键－值对(K-V)并生成一个注意力输出O。
# 由于查询、键和值来自同一组输入，因此被称为 自注意力（self‐attention）/内部注意力（intra‐attention） 
# 即q,k,v是用同一组输入得出的

import math
import torch
from torch import nn
from d2l import torch as d2l

#给定一个由词元组成的输入序列x1, . . . , xn，即一组token序列作为输入，其中任意xi ∈ Rd，d维度的向量
#该序列的自注意力输出为一个长度相同的序列 y1, . . . , yn
#所以attention计算可以表示为：yi = f(xi, (x1, x1), . . . , (xn, xn))

#位置编码positional encoding
#在处理词元序列时，循环神经网络是逐个的重复地处理词元的，而自注意力则因为并行计算而放弃了顺序操作。
# 为了使用序列的顺序信息，通过在输入表示中添加 位置编码（positional encoding）来注入绝对的或相对的位置信息。

#假设输入表示X ∈ Rn×d 包含一个序列中n个词元的d维嵌入表示
#位置编码使用相同形状的位置嵌入矩阵P ∈ Rn×d输出X + P

#基于正弦函数和余弦函数的固定位置编码
#在位置嵌入矩阵P中，行代表词元在序列中的位置，列代表位置编码的不同维度。
#@save
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

#不同的行即位置不同，这个位置信息编码位了三角函数的频率！ 
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
d2l.plt.savefig(f"/home2/wzc/d2l_learn/d2l-zh/learning_code/picture/my_plot_61.png", dpi=600)
