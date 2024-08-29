import torch

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(Y)

#运行一些操作可能会导致为新结果分配内存
before = id(Y)
print(before)
Y = Y + X
print(id(Y) == before)

#通常情况下，我们希望原地址执行这些操作
#可以使用切片表示法将操作的结果分配给先前分配的数组，例如Y[:]= <expression>
Z = torch.zeros_like(Y)
print(Z)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

#举例：这样操作前后还是存在同一个地址，即原地操作
before = id(X)
X[:] = X + Y #等价X += Y
print(id(X) == before)