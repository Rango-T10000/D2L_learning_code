#从经典算法————线性神经网络
#经典统计学习技术中的线性回归和softmax回归可以视为线性神经网络

#在开始寻找最好的模型参数（model parameters） w和b之前，我们还需要两个东西：
#（1）一种模型质量的度量方式: 即目标函数/损失函数
#（2）一种能够更新模型以提高模型预测质量的方法：即梯度下降法

#通常会在每次需要计算更新的时候随机抽取一小批样本，这种变体叫做小批量随机梯度下降（minibatch stochastic gradient descent）
#算法的步骤如下：
# （1）初始化模型参数的值，如随机初始化；
# （2）从数据集中随机抽取小批量样本且在负梯度的方向上更新参数，并不断迭代这一步骤。
#每个小批量中的样本数，这也称为批量大小（batch size），属于超参数（hyperparameter）。

#给定特征估计目标的过程通常称为预测（prediction）或推断（inference）/推理
#推断这个词已经成为深度学习的标准术语

#在训练我们的模型时，我们经常希望能够同时处理整个小批量的样本。为了实现这一点，需要我们对计算进行矢量化，而不是使用开销高昂的for循环
import math
import time
import numpy as np
import torch
from d2l import torch as d2l

n = 10000
a = torch.ones([n])
b = torch.ones([n])

#为了进行运行时间的基准测试，所以我们定义一个计时器
class Timer: #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()
    
    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)
    
    def sum(self):
        """返回时间总和"""
        return sum(self.times)
    
    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

#首先，我们使用for循环，每次执行一位的加法
c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{timer.stop():.5f} sec')

#我们使用重载的+运算符来计算按元素的和
timer.start()
d = a + b
print(f'{timer.stop():.5f} sec')
print("结果很明显，第二种方法比第一种方法快得多。矢量化代码通常会带来数量级的加速")


#下面我们定义一个Python函数来计算正态分布,根据正态分布概率密度函数定义
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

# 再次使用numpy进行可视化
x = np.arange(-7, 7, 0.01)
# 均值和标准差对
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], 
         xlabel='x',ylabel='p(x)', 
         figsize=(5, 3),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])

d2l.plt.savefig("/home/wzc/d2l_learn/d2l-zh/learning_code/my_plot_11.png", dpi = 600)

#最小化目标函数和执行极大似然估计等价

#由于模型重点在发生计算的地方，所以通常我们在计算层数时不考虑输入层。也就是说， 图3.1.2中神经网络的层数为1
#将线性回归模型视为仅由单个人工神经元组成的神经网络，或称为单层神经网络
#每个输入都与每个输出相连:将这种变换（图3.1.2中的输出层）称为全连接层（fully‐connected layer）或称为稠密层（dense layer）