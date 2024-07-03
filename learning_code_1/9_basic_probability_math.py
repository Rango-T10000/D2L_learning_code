import torch
from torch.distributions import multinomial
from d2l import torch as d2l

print(torch.ones([6]))
fair_probs = torch.ones([6]) / 6 #一个概率向量 
a = multinomial.Multinomial(1, fair_probs).sample()
b = multinomial.Multinomial(10, fair_probs).sample()
#把分布（distribution）看作对事件的概率分配
#它在索引i处的值是采样结果中i出现的次数,multinomial.Multinomial是多项分布：将概率分配给一些离散选择的分布称为多项分布
print(a)
print(b)

#现在我们知道如何对骰子进行采样，我们可以模拟1000次投掷
# 将结果存储为32位浮点数以进行除法
counts = multinomial.Multinomial(1000, fair_probs).sample()
print(counts / 1000) # 相对频率作为估计值

#我们也可以看到这些概率如何随着时间的推移收敛到真实概率。让我们进行500组实验，每组抽取10个样本
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
        label=("P(die=" + str(i + 1) + ")"))

d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.savefig('/home/wzc/d2l_learn/d2l-zh/learning_code/my_plot2.png',dpi = 600)
