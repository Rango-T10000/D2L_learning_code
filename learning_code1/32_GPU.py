#"内存"（RAM）和硬盘（磁盘）是计算机系统中两种不同类型的存储设备
#主内存（RAM）是计算机系统中的临时存储器，用于存储正在运行的程序和数据。
# 它提供了快速的读写速度，但是数据在断电后会被清除，因此主内存只用于临时存储和计算。
#硬盘（磁盘）是计算机系统中的永久存储设备，用于长期存储数据和文件。
# 硬盘通常具有较大的存储容量，但相对于主内存来说，读写速度较慢。硬盘上存储的数据在断电后不会丢失，可以长期保存。


#我们可以指定用于存储和计算的设备，如CPU和GPU
#默认情况下，张量是在内存中创建的，然后使用CPU计算它。注意：这里说的"内存" 指的是计算机系统的主内存（也称为随机访问存储器，RAM）

#在PyTorch中， 
# CPU用torch.device('cpu')表示   cpu设备意味着所有物理CPU和内存，这意味着PyTorch的计算将尝试使用所有CPU核心
# GPU用torch.device('cuda')表示  gpu设备只代表一个卡和相应的显存
# 第i块GPU（i从0开始）用torch.device(f'cuda:{i}') 来表示，cuda:0和cuda是等价的

#数据可以在不同的设备之间传递
import torch
from torch import nn
import time
import numpy as np

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



print(torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1'))

#可以查询可用gpu的数量
print(torch.cuda.device_count())


#定义两个函数允许我们在不存在所需所有GPU的情况下运行代码
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(try_gpu(), try_gpu(10), try_all_gpus())


#我们可以查询张量所在的设备。默认情况下，张量是在CPU上创建的
x = torch.tensor([1, 2, 3])
print(x.device)

#需要注意的是，无论何时我们要对多个项进行操作，它们都必须在同一个设备上
#例如，如果我们对两个张量求和，我们需要确保两个张量都位于同一个设备上

#有几种方法可以在GPU上存储张量。例如，我们可以在创建张量时指定存储设备
#在GPU上创建的张量只消耗这个GPU的显存。我们可以使用nvidia-smi命令查看显存使用情况。
X = torch.ones(2, 3, device=try_gpu())
Y = torch.rand(2, 3, device=try_gpu(1))
print(Y)

#数据X在cuda,Y在cuda:1
#现在把X复制到cuda:1上去，和Y执行加法,对两个张量求和，我们需要确保两个张量都位于同一个设备上
timer = Timer()
Z = X.cuda(1)
print("把X复制到cuda:1上去用时:",f'{timer.stop():.5f} sec')

timer.start()
Z+Y
print("在cuda:1上计算加法用时:",f'{timer.stop():.5f} sec')

timer.start()
print(Z+Y)
print("在终端打印结果用时，从cuda:1移动到主内存才能打印带终端:",f'{timer.stop():.5f} sec')



#神经网络模型可以指定设备。下面的代码将模型参数放在GPU上
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
print(net(X))
print(net[0].weight.data.device)


#不经意地移动数据可能会显著降低性能。一个典型的错误如下：计算GPU上每个小批量的损失，
#并在命令行中将其报告给用户（或将其记录在NumPy ndarray中）时，将触发全局解释器锁，从而使所有GPU阻塞。
# 最好是为GPU内部的日志分配内存，并且只移动较大的日志。


