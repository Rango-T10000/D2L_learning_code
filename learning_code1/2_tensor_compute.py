# 在Python中，数组（Array）、列表（List）、元组（Tuple）和张量（Tensor）是不同的数据类型，各自具有一些不同的特点和用途。
# 数组（Array）：在Python中，数组通常指的是NumPy库中的多维数组（numpy.ndarray）。数组是一个固定大小、存储相同类型元素的数据结构。它可以是一维、二维或多维的，并且支持高效的数值计算和向量化操作。数组在科学计算、数据分析和机器学习等领域中被广泛使用。
# 列表（List）：列表是Python中最常用的数据结构之一，它是一个有序、可变的集合，可以包含不同类型的元素。列表使用方括号[]表示，元素之间用逗号分隔。列表的长度和内容可以动态改变，可以添加、删除和修改元素。列表适用于存储和处理任意类型的数据。
# 元组（Tuple）：元组是Python中的另一个有序集合，与列表类似，但元组是不可变的，即一旦创建后，不能修改其元素。元组使用圆括号()表示，元素之间用逗号分隔。元组适用于存储不可变的数据，可以作为字典的键、函数返回多个值等。
# 张量（Tensor）：张量是在机器学习和深度学习中使用的数据结构，通常由专门的库（如PyTorch或TensorFlow）提供。张量是一个多维数组，类似于数组，但具有额外的功能和操作，如自动微分、并行计算等。张量广泛应用于深度学习模型中的数据表示和数值计算。
# 总结一下，数组是NumPy库中的多维数组，列表是一种有序、可变的集合，元组是一种有序、不可变的集合，而张量是在机器学习和深度学习中使用的多维数组，具有额外的功能和操作。每种数据类型有其特定的用途和适用范围，根据具体需求选择合适的数据类型。
import torch

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])

#按元素计算，elementwise
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y) # **运算符是求幂运算
print(torch.exp(x)) #e^x次方

#把多个张量连结（concatenate）
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))

#逻辑运算符
print(X == Y)

#对张量中的所有元素进行求和
print(X.sum())
print(Y.sum())

#当两个张量的形状不同，使用广播机制，其实就是自动复制补成一样大，矩阵a将复制列，矩阵b将复制行，然后再按元素相加
a = torch.arange(3).reshape((3, 1)) 
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)
print(a + b)

#索引
print(X[0])
print(X[-1]) #-1是最后一个元素的索引
print(X[1:3])#这个其实选的是第2个和第3个元素
X[1, 2] = 9 #把第1行，第2列的元素指定为9，（都是从0开始数）
print(X)
X[0:2, :] = 12
print(X)

#运行一些操作可能会导致为新结果分配内存
before = id(Y)
print(before)
Y = Y + X
print(id(Y) == before)