#softmax回归: 柔性最大值函数
#分类问题：
#1.我们只对样本的“硬性”类别感兴趣，即属于哪个类别； 
#2.我们希望得到“软性”类别，即得到属于每个类别的概率（即softmax最大，即预测的属于哪个类别的概率最大）

#统计学家很早以前就发明了一种表示分类数据的简单方法：独热编码（one‐hot encoding）
#独热编码是一个向量，它的分量和类别一样多。类别对应的分量设置为1，其他所有分量设置为0
#例：{狗, 猫, 鸡} -> y ∈ {(1, 0, 0), (0, 1, 0), (0, 0, 1)}.

#与线性回归一样， softmax回归也是一个单层神经网络
#softmax回归的输出层也是全连接层
#尽管softmax是一个非线性函数，但softmax回归的输出仍然由输入特征的仿射变换决定。
# 因此， softmax回归是一个线性模型（linear model）

#就是原本的输出o不能作为每个类别的概率，因为可能是负的，还有总和并不为1,这样的输出o是未规范化的预测
#所以想了个办法，让原本的输出通过softmax函数映射成每个类别的概率,y = softmax(o)，即获取一个向量并将其映射为概率
#要将输出视为概率，我们必须保证在任何数据上的输出都是非负的且总和为1
#softmax函数正是这样做的：softmax函数能够将未规范化的预测变换为非负数并且总和为1，同时让模型保持可导的性质。
#1.对每个未规范化的预测求幂，这样可以确保输出非负
#2.为了确保最终输出的概率值总和为1，我们再让每个求幂后的结果除以它们的总和


#损失函数： 使用最大似然估计，对数似然
#在分类问题中： 输入是X,其属于类别Y的概率是一个条件概率：P(Y | X)
#根据最大似然估计，我们最大化P(Y | X)，相当于最小化负对数似然
#总之记住：最大化条件概率，等价于最小化其负对数似然
#最终得出的损失函数l的表达式形似信息论中信息熵的样子
#所以(3.4.8)中的损失函数通常被称为交叉熵损失（cross‐entropy loss），交叉说的是从X到Y,有两个量



