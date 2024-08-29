#使用卷积神经网络，我们可以在图像中保留空间结构。
#同时，用卷积层代替全连接层的另一个好处是：模型更简洁、所需的参数更少

#LeNet，它是最早发布的卷积神经网络之一--------这个模型是由AT&T贝尔实验室的研究员Yann LeCun在1989年提出的
#目的是识别图像(LeCun et al., 1998)中的手写数字
#输入是手写数字，输出为10种可能结果的概率


import torch
from torch import nn
from d2l import torch as d2l

#用深度学习框架实现LeNet-5
#只需要实例化一个Sequential块并将需要的层连接在一起,搭积木
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),   #注：批量大小/样本数：1 通道数：6、高度：5、宽度：5
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

print("小改动的LeNet-5的网络结构：\n",net)
print('\n')

#将一个（即样本数/batch size为1）大小为28 × 28的单通道（黑白）图像通过LeNet，在每一层打印输出的形状
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)


#看看LeNet在Fashion‐MNIST数据集上的表现
#虽然卷积神经网络的参数较少，每个参数都参与更多的乘法,所以计算成本也很高
#故尝试用GPU来加速运算，由于完整的数据集位于内存中，因此在模型使用GPU计算数据集之前，我们需要将其复制到显存中
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

#修改后的精度计算函数，
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

#为了使用GPU，需要修改 3.6节中定义的train_epoch_ch3函数
#在进行正向和反向传播之前，我们需要将每一小批量数据移动到我们指定的设备（例如GPU）上
#使用Xavier随机初始化模型参数
#使用交叉熵作为损失函数
#小批量随机梯度下降作为参数的优化算法
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):  #@save
    """用GPU训练模型(在第六章定义)"""

    #使用Xavier随机初始化模型参数，包括MLP/Linear的参数和卷积层的参数，只有这两种层有参数（权重+偏置）
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)

    #将每一小批量数据移动到我们指定的设备（例如GPU）上
    print('training on', device)
    net.to(device)

    #小批量随机梯度下降作为参数的优化算法
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
    #使用交叉熵作为损失函数
    loss = nn.CrossEntropyLoss()

    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    d2l.plt.savefig(f"/home2/wzc/d2l_learn/d2l-zh/learning_code/picture/my_plot_38.png", dpi=600)
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    

#训练
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

#访问参数：
# print(*[(name, param.shape) for name, param in net.named_parameters()])
print()
for name, param in net.named_parameters():
    print(name, param.shape)
print()











