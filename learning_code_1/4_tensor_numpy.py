import torch
import numpy

#将深度学习框架定义的张量转换为NumPy张量（ndarray）很容易，反之也同样容易。 torch张量和numpy数将共享它们的底层内存
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
A = X.numpy()
B = torch.tensor(A)
print(X)
print(A)
print(B)

print(type(A))
print(type(B))

#将大小为1的张量转换为Python标量
a = torch.tensor([3.5])
print(a)
print(a.item())
print(float(a))
print(int(a))