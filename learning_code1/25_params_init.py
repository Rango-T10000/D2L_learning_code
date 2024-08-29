import torch
from d2l import torch as d2l

#梯度消失（gradient vanishing）
#神经元要么完全激活要么完全不激活（就像生物神经元）的想法很有吸引力。然而，它却是导致梯度消失问题的一个常见的原因，让我们仔细看看sigmoid函数为什么会导致梯度消失。
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))
d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
d2l.plt.savefig(f"/home2/wzc/d2l_learn/d2l-zh/learning_code/picture/my_plot_25.png", dpi=600)
#更稳定的ReLU系列函数已经成为从业者的默认选择




#梯度爆炸（gradient exploding）
#我们生成100个高斯随机矩阵，并将它们与某个初始矩阵相乘。
M = torch.normal(0, 1, size=(4,4))
print('一个矩阵 \n',M)
for i in range(100):
    M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))

print('乘以100个矩阵后\n', M)

#参数的初始化：选择适当的初始化方法可以解决（或至少减轻）在每一层的隐藏单元之间具有排列对称性，该对称性会使模型的表达能力变弱
#举例：Xavier初始化法
#Xavier初始化从均值为零，方差 σ2 = 2/（nin+nout） 的高斯分布中采样权重，nin和nout是输入和输出的个数


