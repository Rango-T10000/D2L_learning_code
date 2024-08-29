#有时我们希望保存训练的模型，以备将来在各种环境中使用（比如在部署中进行预测）
#如何加载和存储权重向量和整个模型

import os
import torch
from torch import nn
from torch.nn import functional as F


#保存张量
root_path = "/home2/wzc/d2l_learn/d2l-zh/learning_code/model"
x = torch.arange(4)
print(x)
torch.save(x, os.path.join(root_path, 'x-file'))

#读取保存的张量
x2 = torch.load(os.path.join(root_path, 'x-file'))
print(x2)

#我们可以存储一个张量列表，然后把它们读回内存
y = torch.zeros(4)
torch.save([x, y],os.path.join(root_path, 'x-files'))
x2, y2 = torch.load(os.path.join(root_path, 'x-files'))
print((x2, y2))

#我们可以存储一个字典，然后把它们读回内存.   从字符串映射到张量的字典
mydict = {'x': x, 'y': y}
torch.save(mydict, os.path.join(root_path, 'mydict'))
mydict2 = torch.load(os.path.join(root_path, 'mydict'))
print(mydict2,"\n")



#加载和保存模型参数,深度学习框架提供了内置函数来保存和加载整个网络
#注意：将保存模型的参数而不是保存整个模型
#因为模型本身可以包含任意代码，所以模型本身难以序列化
#为了恢复模型，我们需要用代码生成架构，然后从磁盘加载参数。

#举例：从熟悉的多层感知机开始尝试一下
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

#接下来，我们将模型的参数存储在一个叫做“mlp.params”的文件中
torch.save(net.state_dict(), os.path.join(root_path, 'mlp.params'))

#为了恢复模型，我们实例化了原始多层感知机模型的一个备份,即把模型的架构保存下来
clone = MLP()
#为模型加载之前存储好的参数
clone.load_state_dict(torch.load(os.path.join(root_path, 'mlp.params')))
print(clone.eval())

