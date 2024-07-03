#使用pandas预处理原始数据，并将原始数据转换为张量格式
#就是把数据集中的所有信息表示为数字，方便电脑处理

import torch
import os
import pandas as pd

#我们首先创建一个人工数据集，并存储在CSV文件中
os.makedirs(os.path.join('/home/wzc/d2l_learn/d2l-zh/learn_test_code', 'data'), exist_ok=True)  #将上级目录（".."）和目录名（"data"）连接起来形成新的目录路径
data_file = os.path.join('/home/wzc/d2l_learn/d2l-zh/learn_test_code', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') # 列名
    f.write('NA,Pave,127500\n') # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')


#读取数据集
data = pd.read_csv(data_file)
print(data)
print(type(data))
print("\n")

#通过位置索引iloc，我们将data分成inputs和outputs
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print(inputs)
print("\n")
print(outputs)
print("\n")

#处理缺失值,插值法用一个替代值弥补缺失值
#对于inputs中缺少的数值，我们用同一列的均值替换“NaN”项。
print(inputs.mean())
inputs = inputs.fillna(inputs.mean())
print(inputs)
print("\n")

#对于inputs中的类别值或离散值：本例子中的Alley有两个值：NaN 和 Pave
#dummy独热编码,pandas的get_dummies()函数对inputs进行独热编码的操作
#将其中的分类变量转换为二进制的独热编码表示形式
inputs = pd.get_dummies(inputs, dummy_na=True)
print("现在inputs和outputs中的所有条目都是数值类型:")
print(inputs,"\n")
print(outputs,"\n")

#现在inputs和outputs中的所有条目都是数值类型，它们可以转换为张量格式
#注意：data,inputs,outputs的类型都是DataFrame对象
#DataFrame是Pandas库中最重要的数据结构之一。它提供了一个类似于表格或电子表格的数据结构
#to_numpy()是DataFrame对象的一个方法，用于将其转换为NumPy数组。这将提取inputs中的数据，并以NumPy数组的形式表示
#即从DataFrame对象 -> numpy -> tensor
x = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(x)
print(y)

