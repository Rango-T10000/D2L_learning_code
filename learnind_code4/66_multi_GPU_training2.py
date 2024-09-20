#多GPU的简洁实现

import torch
from torch import nn
from d2l import torch as d2l

#用Resnet18来举例
#@save
def resnet18(num_classes, in_channels=1):
    """稍加修改的ResNet-18模型"""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(in_channels, out_channels,
                                        use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # 该模型使用了更小的卷积核、步长和填充，而且删除了最大汇聚层
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(
        64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net

#网络实例化
net = resnet18(10)
# 获取GPU列表
devices = d2l.try_all_gpus()

#复习数据并行化的多gpu训练:
# 需要在所有设备上初始化网络参数；
# 在数据集上迭代时，要将小批量数据分配到所有设备上；
# 跨设备并行计算损失及其梯度；
# 聚合梯度，并相应地更新参数。

def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  #dataloader?
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]

    #网络参数初始化
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)

    # 在多个GPU上设置模型
    net = nn.DataParallel(net, device_ids=devices) #把整套网络的参数分发到所有gpu
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()


    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()  #使得网络处于训练模式，就是为了让dropout和BN发挥作用
        timer.start()
        for X, y in train_iter: #--------dataloader中遍历小批量数据（mini-batches）--------
            trainer.zero_grad() #每个iter开始梯度清零

            #需要收到先把数据送到主cpu，默认是cdua0,然后DataParallel 将自动复制输入数据，
            # 并根据 GPU 的数量进行拆分，比如把 batch 分成多个小 batch 送到不同的 GPU 上。
            X, y = X.to(devices[0]), y.to(devices[0])

            l = loss(net(X), y)
            l.backward()    #反向传播计算梯度
            trainer.step()  #参数更新
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'
          f'在{str(devices)}')
    
#所以最关键的就是这里使用的net = nn.DataParallel(net, device_ids=devices)
# ### 1. **DataParallel 的工作机制**
# `nn.DataParallel` 通过以下步骤实现多 GPU 训练：
# - **复制模型**：在每个指定的 GPU 上复制一份模型的副本。
# - **分发输入数据**：自动将输入数据按 batch 大小拆分，分发到每个 GPU 上。例如，如果有 2 个 GPU，输入数据的 batch 可能会分为 2 个小 batch，分别送到每个 GPU 上的模型副本。
# - **并行计算**：每个 GPU 独立计算其对应数据的前向传播和梯度。
# - **梯度聚合**：PyTorch 会自动从所有 GPU 上收集梯度，并在主设备（通常是 `device[0]`）上聚合这些梯度。
# - **更新参数**：优化器在主设备上对聚合的梯度执行参数更新。

# ### 2. **为什么数据先放到 `devices[0]`？**
# 虽然 `nn.DataParallel` 会自动将数据分发到多个 GPU，但它需要输入数据首先位于主 GPU 上（即 `devices[0]`）。然后，`DataParallel` 会根据需要自动将这些数据分割并发送到其他 GPU。因此，在代码中先把 `X` 和 `y` 放到 `devices[0]` 上是一个必要的步骤，这样 `DataParallel` 才能进一步分发这些数据。

# ### 3. **DataParallel 是如何工作的**：
# - **输入数据**：当你将 `X` 和 `y` 放在 `devices[0]`（主 GPU）上时，`DataParallel` 将自动复制输入数据，并根据 GPU 的数量进行拆分，比如把 batch 分成多个小 batch 送到不同的 GPU 上。
# - **输出数据**：多个 GPU 会并行计算各自的小 batch 的输出，然后 `DataParallel` 会把所有输出汇集到主 GPU 上并合并。
  
# 因此，**数据需要首先放在主设备上**，才能让 `DataParallel` 在所有设备上正确地执行并行计算。
