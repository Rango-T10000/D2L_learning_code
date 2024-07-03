#许多公司、学者和业余爱好者开发了各种成熟的开源框架
#数据迭代器、损失函数、优化器和神经网络层很常用，现代深度学习库也为我们实现了这些组件
#通过深度学习框架的高级API来实现我们的模型只需要相对较少的代码
#通过使用深度学习框架来简洁地实现 3.2节中的线性回归模型

import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

# nn是神经网络的缩写
from torch import nn

#生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

#读取数据集
#调用框架中现有的API来读取数据。我们将features和labels作为API的参数传递，并通过数据迭代器指定batch_size
#布尔值is_train表示是否希望数据迭代器对象在每个迭代周期内打乱数据
def load_array(data_arrays, batch_size, is_train=True): #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays) #*是把数组变成元组，元组是可变数量的
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

#使用next从迭代器中获取第一项
print(next(iter(data_iter)))
print("\n")

#定义模型:可以使用框架的预定义好的层(直接用函数，一句代码实现之前自己写的)
#首先定义一个模型变量net，它是一个Sequential类的实例。 
#Sequential类将多个层串联在一起。当给定输入数据时，Sequential实例将数据传入到第一层，
#然后将第一层的输出作为第二层的输入，以此类推
#在PyTorch中，全连接层在Linear类中定义，第一个指定输入特征形状(即自变量的形状)，第二个指定输出特征形状(即因变量的形状)
net = nn.Sequential(nn.Linear(2, 1))


#初始化模型参数
#在这里，我们指定每个权重参数应该从均值为0、标准差为0.01的正态分布中随机采样，偏置参数将初始化为零
#通过net[0]选择网络中的第一个图层，
#然后使用weight.data和bias.data方法访问参数。
#我们还可以使用替换方法normal_和fill_来重写参数值
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
print("初始的参数w:",net[0].weight.data)
print("初始的参数b:",net[0].bias.data)
print("\n")

#定义损失函数:计算均方误差使用的是MSELoss类，也称为平方L2范数
loss = nn.MSELoss()

#定义优化算法
#小批量随机梯度下降算法（sgd）是一种优化神经网络的标准工具， PyTorch在optim模块中实现了该算法的许多变种
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

#训练：
#在每个迭代周期里，我们将完整遍历一次数据集（train_data），不停地从中获取一个小批量的输入和相应的标签
#对于每一个小批量，我们会进行以下步骤:
#1. 通过调用net(X)生成预测并计算损失l（前向传播）
#2. 通过进行反向传播来计算梯度
#3. 通过调用优化器来更新模型参数
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step() #更新参数
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')


w = net[0].weight.data
print('w的估计误差： ', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差： ', true_b - b)

