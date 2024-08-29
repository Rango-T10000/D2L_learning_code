#权重衰减（weight decay）是最广泛使用的正则化的技术之一,它通常也被称为L2正则化
#通过函数与零的距离来衡量函数的复杂度，∥w-0∥ = ∥w∥
#即为了惩罚权重向量的大小，我们必须以某种方式在损失函数中添∥w∥2，∥w∥2即L2范数的平方
#就是给loss加个惩罚项：L(w, b) + λ/2*∥w∥2，正则化常数λ是个超参数
#搞个1/2,平方都是为了求导以后的形式简洁

#根据Loss+的惩罚项是L1还是L2
# L2正则化线性模型构成经典的岭回归（ridge regression）算法。（即权重衰减（weight decay））
# L1正则化线性回归是统计学中类似的基本模型，通常被称为套索回归（lasso regression）（即特征选择（feature selection））

#L2惩罚是它对权重向量的大分量施加了巨大的惩罚（权重向量的大分量，平方完更大）。这使得我们的学习算法偏向于在大量特征上均匀分布权重的模型。即权重衰减（weight decay）
#L1惩罚会导致模型将权重集中在一小部分特征上，而将其他权重清除为零。即特征选择（feature selection）

#权重衰减举例
import torch
from torch import nn
from d2l import torch as d2l

#生成数据集
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
train_iter = d2l.load_array(train_data, batch_size)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

#从零开始实现

#初始化模型参数
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

#定义L2范数惩罚
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

#训练
def train(lambd, image_index = None):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)   #lambd = 0就是不用权重衰减
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    i = image_index
    d2l.plt.savefig(f"/home2/wzc/d2l_learn/d2l-zh/learning_code/picture/my_plot_23_{i}.png", dpi=600)
    print('w的L2范数是：', torch.norm(w).item())


#忽略正则化直接训练，lambd = 0就是不用权重衰减
train(lambd=0,image_index = 1)

#使用权重衰减
train(lambd=3,image_index = 2)




#使用深度学习框架简洁实现 
#给Loss加上了L2惩罚，后面参数/权重再优化的时候就得减去他们对参数的梯度的一个值
#所以权重衰减功能在深度学习框架的优化器中提供
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数b没有衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},   #权重衰减集成到优化算法中
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())