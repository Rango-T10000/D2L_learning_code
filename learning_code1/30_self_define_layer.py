#有时我们会遇到或要自己发明一个现在在深度学习框架中还不存在的层。在这些情况下，必须构建自定义层
import torch
import torch.nn.functional as F
from torch import nn

#首先，我们构造一个没有任何参数的自定义层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
    

layer = CenteredLayer()
print(layer)
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

#现在，我们可以将层作为组件合并到更复杂的模型中
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

#作为额外的健全性检查，我们可以在向该网络发送随机数据后，检查均值是否为0
Y = net(torch.rand(4, 8))
print(Y.mean())



#以上我们知道了如何定义简单的层，下面我们继续定义具有参数的层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):   #in_units和units，分别表示输入数和输出数
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
    

#实例化MyLinear类并访问其模型参数
linear = MyLinear(5, 3)
print(linear)
print(linear.weight)

#我们可以使用自定义层直接执行前向传播计算
print(linear(torch.rand(2, 5)))

#还可以使用自定义层构建模型，就像使用内置的全连接层一样使用自定义层
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))

