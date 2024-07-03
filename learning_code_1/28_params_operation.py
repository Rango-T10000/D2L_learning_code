import torch
from torch import nn


#举例：具有单隐藏层的多层感知机
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
print(net)
X = torch.rand(size=(2, 4))
print(net(X))

#参数访问
#当通过Sequential类定义模型时，我们可以通过索引来访问模型的任意层，及其参数
#例：检查第二个全连接层的参数
print(net[2].state_dict())

#参数是复合的对象，包含值、梯度和额外信息。这就是我们需要显式参数值的原因
#要对参数执行任何操作，首先我们需要访问底层的数值
#例：检查第二个全连接层的参数bias的类型，值
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

#除了值之外，我们还可以访问每个参数的梯度。
#由于我们还没有调用反向传播，所以参数的梯度处于初始状态
print(net[2].weight.grad == None)

#一次性访问所有参数
#访问第一个全连接层的参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])

#访问所有层
print(*[(name, param.shape) for name, param in net.named_parameters()])

#另一种访问网络参数的方式
print(net.state_dict()['2.bias'].data) #等同于print(net[2].bias.data)


#将多个块相互嵌套，参数命名约定是如何工作的
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X),"\n")

#设计了网络后，我们看看它是如何工作的,可以直接用print函数打印网络的结构
print(rgnet)

#访问第一个主要的块中、第二个子块的第一层的偏置项
print(rgnet[0][1][0])
print(rgnet[0][1][0].bias.data)
print(rgnet[0][1][0].weight.data)

#参数初始化
#PyTorch的nn.init模块提供了多种预置初始化方法
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
print("\n",net[0].weight.data[0], net[0].bias.data[0])


#你也可以自定义初始化
#我们还可以将所有参数初始化为给定的常数，比如初始化为1。
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print("\n",net[0].weight.data[0], net[0].bias.data[0])
print("\n")

#还可以对某些块应用不同的初始化方法,例如：
#使用Xavier初始化方法初始化第一个神经网络层
#将第三个神经网络层初始化为常量值42
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)

#你也可以自定义初始化
#在下面的例子中，我们使用以下的分布为任意权重参数w定义初始化方法
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
print("\n",net[0].weight[:2])


#注意，我们始终可以直接设置参数。
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]

#参数绑定
#希望在多个层间共享参数：我们可以定义一个稠密层，然后使用它的参数来设置另一个层的参数
# 我们需要给共享层一个名称，以便可以引用它的参数
#当你在神经网络模型中多次使用同一个层对象时，这些层对象会共享相同的参数。
#这是因为在 PyTorch 中，模型的层是对象，层对象内部包含了权重参数等属性。
#当你将同一个层对象多次添加到模型中时，这些层对象实际上是引用同一个对象，因此它们共享相同的参数。
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
print(net)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])

#即第2个和第4个神经网络层的参数是绑定的。不仅值相等，而且由相同的张量表示。
#如果我们改变其中一个参数，另一个参数也会改变。