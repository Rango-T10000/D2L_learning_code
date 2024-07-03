import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

#用于在 Jupyter Notebook 中显示 SVG 格式的图像
d2l.use_svg_display()

#读取数据集：Fashion‐MNIST数据集 (Xiao et al., 2017)
#通过框架中的内置函数将Fashion‐MNIST数据集下载并读取到内存中。
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="/home/wzc/d2l_learn/d2l-zh/learning_code/data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="/home/wzc/d2l_learn/d2l-zh/learning_code/data", train=False, transform=trans, download=True)
print("mnist_train的大小：",len(mnist_train))
print("mnist_test的大小：",len(mnist_test))
print("通道数,像素高度h,像素宽度w：",mnist_train[0][0].shape)

#定义函数用于在数字标签索引及其文本名称之间进行转换
def get_fashion_mnist_labels(labels): #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

#创建一个函数来可视化这些样本
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

#注：next(iter(x))第一次用就是取x中的第一个值
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
d2l.plt.savefig("/home/wzc/d2l_learn/d2l-zh/learning_code/my_plot_15.png", dpi = 600)


#读取小批量,使用内置的数据迭代器data.DataLoader，从训练数据集中按批次获取数据
def get_dataloader_workers(): #@save
    """使用4个进程来读取数据"""
    return 4

batch_size = 256
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=get_dataloader_workers())


#查看读取训练数据所需的时间，数据原本存储在硬盘中，读取数据指的是将数据从硬盘加载到CPU内存中
timer = d2l.Timer()
for X, y in train_iter:
    continue
print("\n")
print("读取训练数据(将数据从硬盘加载到CPU内存中)所需的时间:",f'{timer.stop():.2f} sec')





