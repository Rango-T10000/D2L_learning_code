#MLP处理表格数据，任务是回归，分类
#CNN处理图像数据，任务是分类，卷积神经网络可以有效地处理空间信息

#新的问题：我们不仅仅可以接收一个序列作为输入，而是还可能期望继续猜测这个序列的后续。预测！！！！！！
#RNN,循环神经网络（recurrent neural network，RNN）则可以更好地处理序列信息
#循环神经网络通过引入状态变量存储过去的信息和当前的输入，从而可以确定当前的输出
#许多使用循环网络的例子都是基于文本数据的，因此我们将在本章中重点介绍语言模型

#序列数据：
#音乐、语音、文本和视频都是连续的。如果它们的序列被我们重排，那么就会失去原有的意义。
#预测即：xt ∼ P(xt | xt−1, . . . , x1).

#有以下两种方法：

#(1)自回归模型（autoregressive models）
#实际不需要之前所有的数据，只需要满足某个长度为τ的时间跨度，即使用观测序列xt−1, . . . , xt−τ也行(这种近似是序列满足马尔可夫条件（Markov condition)
#所以实际是预测：xt ∼ P(xt | xt−1, . . . , xt−τ).即考虑过去的一部分历史，不是过去的所有历史
#这种模型被称为自回归模型（autoregressive models），因为它们是对自己执行回归！

#(2)隐变量自回归模型（latent autoregressive models）,见P291图
#保留一些对过去观测的总结ht(隐变量)，并且同时更新预测xˆt和总结ht
#即基于xˆt = P(xt | ht)估计xt
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


#首先，我们生成一些数据：使用正弦函数和一些可加性噪声来生成序列数据，时间步为1, 2, . . . , 1000
import torch
from torch import nn
from d2l import torch as d2l

T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
d2l.plt.savefig(f"/home2/wzc/d2l_learn/d2l-zh/learning_code/picture/my_plot_46.png", dpi=600)

#将这个序列转换为模型的特征－标签（feature‐label）对
tau = 4 #这个tau是嵌入维度（embedding dimension），即本来是（y,x）,现在是（y,x1,x2...,x_tau）
features = torch.zeros((T - tau, tau)) #而且，只要T - tau个数据，不要前tau个
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

print(features.shape)
print(labels.shape)

# 只有前n_train个样本用于训练, 在这里，我们仅使用前600个“特征－标签”对进行训练
batch_size, n_train = 16, 600
train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)

# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')

#训练
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step() #参数更新
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)

#预测：训练的时候只用前600个数据，发现预测数据从4～1000也能和原来吻合
#检查模型预测下一个时间步的能力，也就是单步预测（one‐step‐ahead prediction），一个tau
onestep_preds = net(features)
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))
d2l.plt.savefig(f"/home2/wzc/d2l_learn/d2l-zh/learning_code/picture/my_plot_46_2.png", dpi=600)


#对于直到xt的观测序列，其在时间步t + k处的预测输出xˆt+k 称为k步预测（k‐step‐ahead‐prediction）
#使用每次预测出的数据再去作为网络的输入去预测新的数据，每次用历史的前tau个feature来预测当前的label，这里单步的跨度是tau
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau] #只给0～604的输入features
for i in range(n_train + tau, T): #预测604～1000的label
    multistep_preds[i] = net(multistep_preds[i - tau:i].reshape((1, -1))) #每次用历史的前tau个feature来预测当前的label

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
d2l.plt.savefig(f"/home2/wzc/d2l_learn/d2l-zh/learning_code/picture/my_plot_46_3.png", dpi=600)



#仔细地看一下k步预测的困难
#随着我们对预测时间k值的增加，会造成误差的快速累积和预测质量的极速下降
max_steps = 64
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64) #使用不同的步长tau值
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
d2l.plt.savefig(f"/home2/wzc/d2l_learn/d2l-zh/learning_code/picture/my_plot_46_4.png", dpi=600)


