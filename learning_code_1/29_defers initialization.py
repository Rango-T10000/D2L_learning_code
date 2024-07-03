#框架的延后初始化（defers initialization），
# 即直到数据第一次通过模型传递时，框架才会动态地推断出每个层的大小
#延后初始化使框架能够自动推断参数形状

import torch
from torch import nn

"""延后初始化"""
# 尚未初始化，此时，因为输入维数是未知的，所以网络不可能知道输入层权重的维数。因此，框架尚未初始化任何参数
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
print(net)

#接下来让我们将数据通过网络，最终使框架初始化参数
X = torch.rand(2, 20)
net(X)
# 根据X自己初始化
print(net)
