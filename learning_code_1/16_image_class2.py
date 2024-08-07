import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

def get_dataloader_workers(): #@save
    """使用4个进程来读取数据"""
    return 4

#整合所有组件
#定义load_data_fashion_mnist函数，用于获取和读取Fashion‐MNIST数据集。这个函数返回训练集和验证集的数据迭代器
#这个函数还接受一个可选参数resize，用来将图像大小调整为另一种形状
def load_data_fashion_mnist(batch_size, resize=None): #@save
    """下载Fashion-MNIST数据集,然后将其加载到内存中"""
    trans = [transforms.ToTensor()]

    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root="/home/wzc/d2l_learn/d2l-zh/learning_code/data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="/home/wzc/d2l_learn/d2l-zh/learning_code/data", train=False, transform=trans, download=True)

    return (data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,num_workers=get_dataloader_workers()))


#通过指定resize参数来测试load_data_fashion_mnist函数的图像大小调整功能
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break

