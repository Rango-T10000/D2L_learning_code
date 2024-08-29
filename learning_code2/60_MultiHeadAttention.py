#在实践中，当给定相同的查询、键和值的集合时，我们希望模型可以基于相同的注意力机制学习到不同的行为，
# 然后将不同的行为作为知识组合起来，捕获序列内各种范围的依赖关系
#允许注意力机制组合使用查询、键和值的不同 子空间表示（representation subspaces）可能是有益的

#多个注意力汇聚，见图10.5.1
#可以用独立学习得到的h组不同的 线性投影（linear projections）来变换查询、键和值
#这h组变换后的查询、键和值将并行地送到注意力汇聚中。
#最后，将这h个注意力汇聚的输出拼接在一起，并且通过另一个可以学习的线性投影进行变换，以产生最终输出。
#这种设计被称为多头注意力（multihead attention）
#对于h个注意力汇聚输出，每一个注意力汇聚都被称作一个头（head）
#基于这种设计，每个头都可能会关注输入的不同部分，可以表示比简单加权平均值更复杂的函数
import math
import torch
from torch import nn
from d2l import torch as d2l


#在实现过程中通常选择缩放点积注意力作为每一个注意力头,即注意力计算选的是缩放点积注意力的公式
#基于n个查询和m个键－值对计算注意力
#即Q: nxd; K: mxd; V: mxv, (注意：这里写的都是后面的维度，省略了开头的batch_size)
#设定pq = pk = pv = po/h，即 pqh = pkh =pvh = po，即可以并行计算h个头
#po是通过参数num_hiddens指定的
#@save
class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)   #经过O这个线性变换是为了将多头注意力头的输出合并并映射到最终的输出空间

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状: (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状: (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状: (batch_size*num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None: #valid_lens参数用于掩蔽（mask）无效的位置。它指定了每个查询的有效长度
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

#为了能够使多个头并行计算，上面的MultiHeadAttention类将使用下面定义的两个转置函数
#@save
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

#@save
def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


#测试：
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
attention.eval()
print(attention)
print('\n')

batch_size, num_queries = 2, 4
num_kvpairs, valid_lens =  6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens)) #用来计算Q的输入X：（2，4，100）
Y = torch.ones((batch_size, num_kvpairs, num_hiddens)) #用来计算K，V的输入Y: （2，6，100）
print("用来计算Q的输入X_size:", X.shape)
print("用来计算K，V的输入Y_size:", Y.shape)
print("Attention_output_size:",attention(X, Y, Y, valid_lens).shape)








