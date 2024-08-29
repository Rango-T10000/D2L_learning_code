#第一次介绍真正的深度网络
#最简单的深度网络称为多层感知机MLP: 由多层神经元组成，每一层与它的上一层相连，从中接收输入；同时每一层也与它的下一层相连，影响当前层的神经元
#通过在网络中加入一个或多个隐藏层来克服线性模型的限制，使其能处理更普遍的函数关系类型

#这种架构通常称为多层感知机（multilayerperceptron），通常缩写为MLP
#将许多全连接层堆叠在一起。每一层都输出到上面的层，直到生成最后的输出
#把前L−1层看作表示（representation），把最后一层看作线性预测器
#输入层不涉及任何计算，因此使用此网络产生输出只需要实现隐藏层和输出层的计算，所以算层数的时候一般不算输入层

#MLP通常用于处理低维数据，例如一维向量或表格数据。
#对于高维数据，需要将其展平为一维向量后才能输入MLP。这种展平操作会忽略数据在原始空间中的空间结构，因为MLP无法显式地学习数据的空间特征。

#从线性到非线性：仅中间加入隐藏层没用，还是线性模型
#为了发挥多层架构的潜力，我们还需要一个额外的关键要素：在仿射变换之后对每个隐藏单元应用非线性的激活函数（activation function） σ。
#为了构建更通用的多层感知机，我们可以继续堆叠这样的隐藏层，一层叠一层，从而产生更有表达能力的模型。

#常见的几种激活函数（activation function）
import torch
from d2l import torch as d2l

#ReLU函数：修正线性单元（Rectified linear unit， ReLU）, ReLU(x) = max(x, 0)
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
d2l.plt.savefig("/home2/wzc/d2l_learn/d2l-zh/learning_code/my_plot_19_ReLU.png", dpi = 600)
#使用ReLU的原因是，它求导表现得特别好：要么让参数消失，要么让参数通过
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
d2l.plt.savefig("/home2/wzc/d2l_learn/d2l-zh/learning_code/my_plot_19_ReLU_grad.png", dpi = 600)

#sigmoid函数:挤压函数（squashing function）：， sigmoid函数将输入变换为区间(0, 1)上的输出
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
d2l.plt.savefig("/home2/wzc/d2l_learn/d2l-zh/learning_code/my_plot_19_Sigmoid.png", dpi = 600)
# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
d2l.plt.savefig("/home2/wzc/d2l_learn/d2l-zh/learning_code/my_plot_19_Sigmoid_grad.png", dpi = 600)

#tanh函数
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
d2l.plt.savefig("/home2/wzc/d2l_learn/d2l-zh/learning_code/my_plot_19_tanh.png", dpi = 600)
# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
d2l.plt.savefig("/home2/wzc/d2l_learn/d2l-zh/learning_code/my_plot_19_tanh_grad.png", dpi = 600)

