#多个层被组合成块，形成更大的模型
#从编程的角度来看，块由类（class）表示
#一个块可以由许多层组成；一个块可以由许多块组成

import torch
from torch import nn
from torch.nn import functional as F

#自己定义一个块：
#类的初始化中定义好有那些层
#前向传播中定义好输入中间经过哪些运算得到输出
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module（nn.Module）的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出,就是输入X,中间经过哪些运算得到输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
    

#测试这个函数：
X = torch.rand(2, 20)

net = MLP()   #创建了 MLP 类的一个实例 net，即一个具体的模型对象
print(net(X))


#顺序块，看看Sequential类是如何工作的
class MySequential(nn.Module):
    def __init__(self, *args):  #将每个模块逐个添加到有序字典_modules中
        super().__init__()   ## 调用MLP的父类Module（nn.Module）的构造函数来执行必要的初始化。

        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
    
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net(X))

#然而，并不是所有的架构都是简单的顺序架构。当需要更强的灵活性时，我们需要定义自己的块。

#例如：有时我们可能希望合并既不是上一层的结果也不是可更新参数的项，我们称之为常数参数（constant parameter）
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流：自己定义的操作，即这里举例如何将任意代码集成到神经网络计算的流程中（其实就是在前向传播的计算过程中加进去的就行了）
        while X.abs().sum() > 1: #在L1范数大于1的条件下，将输出向量除以2，直到它满足条件为止
            X /= 2
        return X.sum()

net = FixedHiddenMLP()
print(net(X))

#例如：我们可以混合搭配各种组合块的方法,即在一个定义的类中引用其他的类，来实现块的嵌套
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

#chimera:嵌合体
chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print(chimera(X))
