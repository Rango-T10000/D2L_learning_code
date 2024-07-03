import torch

#通过查看API文档学习PyTorch函数和类的用法

#为了知道模块中可以调用哪些函数和类，可以调用dir函数
#通常可以忽略以“__”（双下划线）开始和结束的函数，它们是Python中的特殊对象
#以单个“_”（单下划线）开始的函数，它们通常是内部函数
print(dir(torch.distributions))
print("\n")

#有关如何使用给定函数或类的更具体说明，可以调用help函数
help(torch.ones)
print("\n")

#在Jupyter记事本中，我们可以使用?指令在另一个浏览器窗口中显示文档
#此外，如果我们使用两个问号，如list??，将显示实现该函数的Python代码