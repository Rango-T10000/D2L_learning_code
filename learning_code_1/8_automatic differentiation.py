#深度学习框架可以自动计算导数：我们首先将梯度附加到想要对其计算偏导数的变量上，
# 然后记录目标值的计算，执行它的反向传播函数，并访问得到的梯度。
import torch

x = torch.arange(4.0)
print(x)

#我们不会在每次对一个参数求导时都分配新的内存。
# 因为我们经常会成千上万次地更新相同的参数，
# 每次都分配新的内存可能很快就会将内存耗尽。

#requires_grad 属性用于指示张量是否需要梯度计算。这是torch的函数的参数/属性
# 当 requires_grad 设置为 True 时，PyTorch 会自动跟踪张量上的所有操作，以便计算梯度和进行自动微分。
#requires_grad_() 方法是为了原地(in-place)修改张量的 requires_grad 属性
x.requires_grad_(True) # 等价于x=torch.arange(4.0,requires_grad=True)
x.grad # 默认值是None
print(x.grad)

#这里y是一个标量
y = 2 * torch.dot(x, x)
print(y)

#接下来，通过调用反向传播函数来自动计算y关于x每个分量的梯度
y.backward()
print(x.grad)
print(x.grad == 4*x) #说明x.grad梯度求的是对的

# 在默认情况下， PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_() #用于将张量的梯度值归零，即x.grad即张量x的梯度属性grad归零
print(x.grad)
y = x.sum()
y.backward()
print(x.grad)
print("\n")

#非标量变量的反向传播
#当y不是标量时，向量y关于向量x的导数的最自然解释是一个矩阵
#对于高阶和高维的y和x，求导的结果可以是一个高阶张量
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。e.g.: y.backward(参数)，这个参数也必须是一个tensor
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
y.sum().backward() # 等价于y.backward(torch.ones(len(x))),即y.backward()的参数是
#首先，我们对 y 进行求和操作，得到一个标量 y_sum。然后，我们调用 y_sum.backward()，计算 y_sum 相对于与之相关的所有张量的梯度。由于 y_sum 是一个标量，因此不需要传递梯度张量作为参数
print(torch.ones(len(x)))
print(x.grad)

#有时，我们希望将某些计算移动到记录的计算图之外,用.detach()函数拆离
#假设y是作为x的函数计算的，而z则是作为y和x的函数计算的
x.grad.zero_()
y = x * x
u = y.detach() #可以分离y来返回一个新变量u，该变量与y具有相同的值，但丢弃计算图中如何计算y的任何信息
z1 = u * x
z2 = y * x

z1.sum().backward() #这里反向传播函数计算z1=u*x关于x的偏导数，同时将u作为常数处理
print(x.grad == u)
x.grad.zero_()

z2.sum().backward() #这里反向传播函数计算z2=y*x关于x的偏导数，即z=x*x*x关于x的偏导数
print(x.grad == 3*x**2)
x.grad.zero_()


#即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），我们仍然可以计算得到的变量的梯度。
#这个函数是一个对于a来说的分段线性函数，即有对于任何a，存在某个常量标量k，使得f(a)=k*a，其中k的值取决于输入a
#求梯度结果就是斜率k
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad)
print(a.grad == d / a)
