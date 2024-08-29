import math
import torch
from torch import nn
from d2l import torch as d2l

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$这里对于后续理解非常重要！
#见P388,公式10.2.4！注意力汇聚（attention pooling）公式：f(x) =α(x, xi)yi,
#其中x是查询， (xi, yi)是键值对
#注意力汇聚是yi的加权平均。将查询x和键xi之间的关系建模为 注意力权重（attention weight） α(x, xi)
# 这个权重将被分配给每一个对应值yi

#根据图10.3.1: 计算注意力汇聚的计算过程
#注意力汇聚函数f：先是（q,k）一起经过一个注意力评分函数（attention scoring function），结果再经过softmax函数，最后和V做计算
#也就是说，先是（q,k）经过评分函数计算，计算的结果用softmax映射成概率分布，这个就是计算出来的注意力权重！
#最终，这个注意力权重再去和V计算加权求和！


#1.
#查询q和键ki的注意力权重（标量）是通过注意力评分函数a将两个向量映射成标量，再经过softmax运算得到的
#在某些情况下，并非所有的值都应该被纳入到注意力汇聚中
#例如：某些文本序列被填充了没有意义的特殊词元。为了仅将有意义的词元作为值来获取注意力汇聚，
# 可以指定一个有效序列长度（即词元的个数），以便在计算softmax时过滤掉超出指定范围的位置
#下面的masked_softmax函数实现了这样的掩蔽softmax操作（masked softmax operation），其中任何超出有效长度的位置都被掩蔽并置为0
#@save
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

#测试：
print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))


#2.
#一般来说，当查询和键是不同长度的矢量时，可以使用加性注意力作为评分函数
#加性注意力（additive attention）的评分函数,见公式（10.3.3）
#将查询和键连结起来后输入到一个多层感知机（MLP）中，感知机包含一个隐藏层，
# 其隐藏单元数是一个超参数h。通过使用tanh作为激活函数，并且禁用偏置项
#@save
class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)

#测试：
#查询、键和值：（批量大小，步数或词元序列长度，特征大小）
#例如：q,k,v: (2, 1, 20)
#最终attention计算结果，即注意力汇聚输出的形状为（批量大小，查询的步数，值的维度）
queries = torch.normal(0, 1, (2, 1, 20))
keys = torch.ones((2, 10, 2))
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)

valid_lens = torch.tensor([2, 6])
attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
attention.eval()
print(attention(queries, keys, values, valid_lens))
# d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),xlabel='Keys', ylabel='Queries')



#3.
#使用点积可以得到计算效率更高的评分函数，但是点积操作要求查询和键具有相同的长度d
#缩放点积注意力（scaled dot‐product attention）评分函数为：a(q, k) = q⊤k/√d.
#基于n个查询和m个键－值对计算注意力
#即Q: nxd; K: mxd; V: mxv
#见公式(10.3.5)，这就是Transformer中的attention计算方法
#@save
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

queries = torch.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
print(attention(queries, keys, values, valid_lens))


