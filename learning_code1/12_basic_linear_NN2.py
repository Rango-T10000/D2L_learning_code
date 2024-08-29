#从零开始实现整个方法，包括数据流水线、模型、损失函数和小批量随机梯度下降优化器
#虽然现代的深度学习框架几乎可以自动化地进行所有这些工作，但从零开始实现可以确保我们真正知道自己在做什么

import random
import torch
from d2l import torch as d2l

#构造一个人造数据集
#在下面的代码中，我们生成一个包含1000个样本的数据集，每个样本包含从标准正态分布中采样的2个特征,即2个自变量
def synthetic_data(w, b, num_examples): #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape) #噪声项ϵ服从均值为0,标准差为0.01的正态分布
    return X, y.reshape((-1, 1))

#使用线性模型参数w = [2, −3.4]⊤、 b = 4.2 和噪声项ϵ生成数据集及其标签
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
#注意， features中的每一行都包含一个二维数据样本， labels中的每一行都包含一维标签值（一个标量）
print('features:', features[0],'\nlabel:', labels[0])
d2l.set_figsize((5,4))
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.gca().set_xlabel('The second features[:, 1]')
d2l.plt.gca().set_ylabel('labels')
d2l.plt.savefig("/home/wzc/d2l_learn/d2l-zh/learning_code/my_plot_12.png", dpi = 600)

#定义一个函数，该函数能打乱数据集中的样本并以小批量方式获取数据
#该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量。
#每个小批量包含一组特征和标签
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

print("\n")
print("第一个小批量（batch）：")
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
print("\n")

#初始化模型参数
#我们通过从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重，并将偏置初始化为0
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

#定义模型：将模型的输入和参数同模型的输出关联起来
def linreg(X, w, b): #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b
#注意：torch.mm没有广播，torch.matmul才有广播，两个都是矩阵乘法

#定义损失函数
def squared_loss(y_hat, y): #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

#定义优化算法：小批量随机梯度下降
#lr:learning rate, batch_size每一个小批量的样本数
def sgd(params, lr, batch_size): #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size #向着梯度减小的方向更新参数
            param.grad.zero_()

#训练
#1. 在每次迭代中，我们读取一小批量训练样本，并通过我们的模型来获得一组预测y_hat。
#2. 计算完损失后，我们开始反向传播，存储每个参数的梯度。是计算loss对每个参数的梯度，所以是对loss反向传播
#3. 最后，我们调用优化算法sgd来更新模型参数
#这里的linreg,sgd,squared_loss都是我上面自己定义的函数
#在每个迭代周期（epoch）中，我们使用data_iter函数遍历整个数据集，
# 并将训练数据集中所有样本都使用一次（假设样本数能够被批量大小整除）。
# 这里的迭代周期个数num_epochs和学习率lr都是超参数，分别设为3和0.03
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y) # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。 l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size) # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
print("\n")