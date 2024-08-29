import torch

#仅包含一个数值被称为标量（scalar）
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(x + y, x * y, x / y, x**y)

#向量可以被视为标量值组成的列表,人们通过一维张量表示向量
x = torch.arange(4)
print(x)
print(x[3])
print(len(x))
print(x.numel())
print(x.shape)

#当调用函数来实例化张量时，我们可以通过指定两个分量m和n来创建一个形状为m × n的矩阵
A = torch.arange(20).reshape(5, 4)
print(A)
print(A[1,3])
print(A.T)

#对称矩阵（symmetric matrix） 等于其转置
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B==B.T)

# 张量描述具有任意数量轴的n维数组
X = torch.arange(24).reshape(2, 3, 4)
print(X)

#张量算法的基本性质
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone() # 通过分配新内存，将A的一个副本分配给B
print(A,"\n",
      B,"\n",
       A + B,"\n")

#两个矩阵的按元素乘法称为Hadamard积（Hadamard product）（数学符号⊙）
#显然你想按位乘，那两个矩阵当然是行列数相同！
print(A * B)

#张量乘以或加上一个标量,张量的每个元素都将与标量相加或相乘,shape不变
a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X, (a * X).shape)

#可以指定张量沿哪一个轴来通过求和降低维度
print(A)
A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0, A_sum_axis0.shape)
A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1, A_sum_axis1.shape)
#A.sum(axis=[0, 1]) # 结果和A.sum()相同

#计算任意形状张量的平均值
A.mean() 
A.sum() / A.numel()
A.mean(axis=0)
A.sum(axis=0) / A.shape[0]

#非降维求和
sum_A = A.sum(axis=1, keepdims=True)
print(A)
print(sum_A)

#想沿某个轴计算A元素的累积总和,可以调用cumsum函数。此函数不会沿任何轴降低输入张量的维度
#结果中的每个元素 (i, j) 是原数组 A 中 (i, j) 位置及之前位置（沿着你给定的方向）的元素的累积和
print(A.cumsum(axis=1))

#点积：是相同位置的按元素乘积的和,等价于torch.sum(x * y)
x = torch.arange(4, dtype = torch.float32)
y = torch.ones(4, dtype = torch.float32)
print(x)
print(y)
print(torch.dot(x, y))
print("\n")

#在代码中使用张量表示矩阵‐向量积，我们使用mv函数
#矩阵-向量积（matrix‐vector product）torch.mv()
print(A)
print(x)
print(A.shape, x.shape)
print(torch.mv(A, x))
print(A*x) #这是每个元素按位乘
print("\n")

#在代码中使用张量表示矩阵‐矩阵积，我们使用mm函数
#矩阵‐矩阵乘法（matrix‐matrix multiplication）torch.mm()
B = torch.ones(4, 3)
print(A)
print(B)
print(A.shape, B.shape)
print(torch.mm(A, B))
print("\n")

#范数运算
#非正式地说，向量的范数是表示一个向量有多大
#目标函数，或许是深度学习算法最重要的组成部分（除了数据），通常被表达为范数
#向量范数是将向量映射到标量的函数f
u = torch.tensor([3.0, -4.0])
print(torch.norm(u)) #在L2范数中常常省略下标2，也就是说∥x∥等同于∥x∥2
print(torch.abs(u).sum()) #L1范数，它表示为向量元素的绝对值之和
#矩阵X ∈ Rm×n的Frobenius范数（Frobenius norm）是矩阵元素平方和的平方根
print(torch.norm(torch.ones((4, 9))))
