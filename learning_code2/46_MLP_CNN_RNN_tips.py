#MLP处理表格数据，任务是回归，分类
#CNN处理图像数据，任务是分类，卷积神经网络可以有效地处理空间信息

#新的问题：我们不仅仅可以接收一个序列作为输入，而是还可能期望继续猜测这个序列的后续。预测！！！！！！
#RNN,循环神经网络（recurrent neural network，RNN）则可以更好地处理序列信息
#循环神经网络通过引入状态变量存储过去的信息和当前的输入，从而可以确定当前的输出
#许多使用循环网络的例子都是基于文本数据的，因此我们将在本章中重点介绍语言模型

#序列数据：
#音乐、语音、文本和视频都是连续的。如果它们的序列被我们重排，那么就会失去原有的意义。
#预测即：xt ∼ P(xt | xt−1, . . . , x1).

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
#在这里，我们仅使用前600个“特征－标签”对进行训练
tau = 4 #这个tau是嵌入维度（embedding dimension），即本来是（y,x）,现在是（y,x1,x2...,x_tau）
features = torch.zeros((T - tau, tau)) #而且，只要T - tau个数据，不要前tau个
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

print(features.shape)
print(labels.shape)

batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)

# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
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
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)

#预测
#检查模型预测下一个时间步的能力，也就是单步预测（one‐step‐ahead prediction），一个tau
onestep_preds = net(features)
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))
d2l.plt.savefig(f"/home2/wzc/d2l_learn/d2l-zh/learning_code/picture/my_plot_46_2.png", dpi=600)
