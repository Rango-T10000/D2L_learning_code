#  在多个GPU上并行化有好几种模式
# 1. 把网络拆分在不同卡上计算
# 2. 数据并行：比方batch_size = 4,就是一个batch有4个mini-batch，把其平分到2个卡上进行计算
#             每个卡运行的都是完整的模型，只不过是本来4个mini-batch在一个卡上走，现在是两个卡
#             最后每个卡算出自己的本地梯度，这些都加起来，再广播给每个卡，用总的梯度更新参数
#             （数据并行是最常见的多卡训练方式）

#以下是几乎不使用pytorch的api的原始实现方式

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

#用原始方式定义一个网络
# 初始化模型参数
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# 定义模型
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# 交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')

#1. 向多个设备分发参数并附加梯度
def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params

#e.g.： 将模型参数复制到GPU[0]
new_params = get_params(params, d2l.try_gpu(0))
print('b1 权重:', new_params[1])
print('b1 梯度:', new_params[1].grad)

#2. 讲求得的每个gpu自己的本地梯度参数(tensor)相加成总的，并广播给所有参与的gpu
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)

#e.g.: 假设有两个gpu: CUDA0;CUDA1
data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('allreduce之前：\n', data[0], '\n', data[1])
allreduce(data)
print('allreduce之后：\n', data[0], '\n', data[1])

#3. 数据分发：比方batch_size = 4,就是一个batch有4个mini-batch，把其平分到2个卡上
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices)    #pytorch的数据分发函数可以直接做这件事
print('input :', data)
print('load into', devices)
print('output:', split)

#4. 定义一个可以同时拆分 feature/X 和 label/Y的函数，用nn.parallel.scatter
#@save
def split_batch(X, y, devices):
    """将X和y拆分到多个设备上"""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))

#-----------------------最终示例：---------------------------
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # 在每个GPU上分别计算损失
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
              X_shards, y_shards, device_params)]
    # 反向传播在每个GPU上分别执行，分别计算各自的本地梯度
    for l in ls:  
        l.backward()
    # 将每个GPU的所有本地梯度相加得到总梯度，并将其广播到所有GPU
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce(
                [device_params[c][i].grad for c in range(len(devices))])
    # 在每个GPU上分别更新模型参数
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # 在这里，我们使用全尺寸的小批量



def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  #这里其实是dataloader?
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # 将模型参数复制到num_gpus个GPU
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:  #--------dataloader中遍历--------
            # 为单个小批量执行多GPU训练
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()  #torch.cuda.synchronize()函数将会等待一个CUDA设备上的所有流中的所有核心的计算完成。
        timer.stop()
        # 在GPU0上评估模型
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'
          f'在{str(devices)}')
    

#启动训练，不同数量gpu运行对比
train(num_gpus=1, batch_size=256, lr=0.2)
train(num_gpus=2, batch_size=256, lr=0.2)