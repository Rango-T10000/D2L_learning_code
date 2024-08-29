#循环神经网络的从零开始实现
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

#读取数据集
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

#独热编码（one‐hot encoding）
#在train_iter中，每个词元都表示为一个数字索引，将这些索引直接输入神经网络可能会使学习变得困难
#我们通常将每个词元表示为更具表现力的特征向量。最简单的表示称为独热编码（one‐hot encoding）
#就是对每一个token进行独热编码
#即：将每个索引映射为相互不同的单位向量
#假设词表中不同词元的数目为N（即len(vocab)），词元索引的范围为0到N − 1
#如果词元的索引是整数i，那么我们将创建一个长度为N的全0向量，并将第i处的元素设置为1
#举例：索引为0和2的独热向量如下
print("索引为0的独热向量：\n",F.one_hot(torch.tensor([0]), len(vocab)))
print("索引为2的独热向量：\n",F.one_hot(torch.tensor([2]), len(vocab)))

#.........省略

#tips:
#inputs的形状： (时间步数量，批量大小，词表大小)
#时间步数（time steps）表示输入序列或输出序列的长度。它表示在一个完整的序列中，模型在每个时间点上处理的步骤数量
#假设我们有一个句子： "I love natural language processing."，
# 并且我们希望将其输入到一个 RNN 模型中进行处理。
# 在这种情况下，时间步数就是句子中的单词数。
# 每个时间步上，RNN 模型将一个单词作为输入，并在隐藏状态的基础上生成一个输出。

#预热（warm‐up）期

#梯度裁剪




