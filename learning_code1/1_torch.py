#深度学习存储和操作数据的主要接口是张量（n维数组）。它提供了各种功能，包括基本数学运算、
# 广播、索引、切片、内存节省和转换其他Python对象

import torch

x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())  #number of element


X = x.reshape(3, 4)
print(X)
print(X.shape)
print(X.numel()) 

Y = x.reshape(-1,4) #-1来调用此自动计算出维度的功能。即我们可以用x.reshape(-1,4)或x.reshape(3,-1)来取代x.reshape(3,4)

print(torch.zeros((2, 3, 4)))
print(torch.zeros(2,3,4))
print(torch.ones((2, 3, 4)))
print(torch.randn(3, 4))
print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))
