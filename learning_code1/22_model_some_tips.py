#作为机器学习科学家，我们的目标是发现模式（pattern）
#我们的目标是发现某些模式，这些模式捕捉到了我们训练集潜在总体的规律
#当我们使用有限的样本时，可能会遇到这样的问题：当收集到更多的数据时，会发现之前找到的明显关系并不成立
#过拟合（overfitting），用于对抗过拟合的技术称为正则化（regularization）


#训练误差（training error）
#泛化误差（generalization error）
#期望泛化误差与训练误差相近

#通常对于神经网络
#我们认为需要更多训练迭代的模型比较复杂
#需要早停（early stopping）的模型（即较少训练迭代周期）就不那么复杂

#模型选择:在评估几个候选模型后选择最终的模型。这个过程叫做模型选择。
#包括：比较本质上是完全不同的模型（比如，决策树与线性模型），比较不同的超参数设置下的同一类模型
#为了确定候选模型中的最佳模型，我们通常会使用验证集（val）

#K折交叉验证：当训练数据稀缺时，我们甚至可能无法提供足够的数据来构成一个合适的验证集

#欠拟合（underfitting）：
#如果模型不能降低训练误差，这可能意味着模型过于简单（即表达能力不足），无法捕获试图学习的模式
#表现为训练和验证误差之间的泛化误差很小

#过拟合（overfitting）：
#表现为训练误差明显低于验证误差时要小心

#举个例子：多项式回归
import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

#自己生成数据集，根据公式(4.4.2)，对以下代码有疑惑去看P144页解释
#为训练集和测试集各生成100个样本
max_degree = 20 # 多项式的最大阶数
n_train, n_test = 100, 100 # 训练和测试数据集大小
true_w = np.zeros(max_degree) # 参数，分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6]) # 见公式(4.4.2)，这是4个真实参数

# 生成多项式特征，就是自变量
features = np.random.normal(size=(n_train + n_test, 1)) #生成一个长度为 n_train + n_test 的随机数数组，作为自变量
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1)) #np.power用于对数组进行幂运算
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1) # gamma(n)=(n-1)!
# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w) #因变量的真实值
labels += np.random.normal(scale=0.1, size=labels.shape) #加上随机噪声

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]]

#从生成的数据集中查看一下前2个样本，即给出两组对应的x,y
#原本是一个x对一个y,后面书里自己定义了多项式特征x^i/i!,所以最高20阶多项式就有20个特征，对应一个y
print("数据集中的自变量/特征\n",features.shape)
print("数据集中的多项式自变量/特征(书里解释了)\n",poly_features.shape)
print("数据集中的因变量/label\n",labels.shape)
print("x1,x2:\n",features[:2])
print("x1,x2对应各自20个多项式自变量/特征:\n",poly_features[:2, :])
print("x1,x2对应的y1,y2\n",labels[:2])

#总结以上过程：
#使用 NumPy 数组生成特征和标签可能是因为 NumPy 在数值计算和数组操作方面具有丰富的功能和灵活性。
#而将 NumPy 数组转换为 PyTorch 张量则是为了能够利用 PyTorch 提供的深度学习框架特性，如自动求导和 GPU 加速等

#定义一个函数来评估模型在给定数据集上的损失
def evaluate_loss(net, data_iter, loss): #@save
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2) # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

#定义训练函数
def train(train_features, test_features, train_labels, test_labels,num_epochs=400, image_index = None):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    i = image_index
    d2l.plt.savefig(f"/home2/wzc/d2l_learn/d2l-zh/learning_code/my_plot_22_{i}.png", dpi=600)
    print('weight:', net[0].weight.data.numpy())

#三阶多项式函数拟合(正常)
# 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
train(poly_features[:n_train, :4], poly_features[n_train:, :4],labels[:n_train], labels[n_train:], image_index =1)

#线性函数拟合(欠拟合)
# 从多项式特征中选择前2个维度，即1和x
train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:],image_index =2)

#高阶多项式函数拟合(过拟合)
# 从多项式特征中选取所有维度
train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:], num_epochs=1500,image_index =3)
