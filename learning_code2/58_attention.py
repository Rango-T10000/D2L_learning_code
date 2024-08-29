#灵长类动物的视觉系统接受了大量的感官输入，这些感官输入远远超过了大脑能够完全处理的程度。
# 然而，并非所有刺激的影响都是相等的。意识的聚集和专注使灵长类动物能够在复杂的视觉环境中将注意力引向感兴趣的物体

#非自主性提示是基于环境中物体的突出性和易见性
#受到了认知和意识的控制，因此注意力在基于自主性提示去辅助选择时将更为谨慎
#“是否包含自主性提示”将注意力机制与全连接层或汇聚层区别开来
#受试者使用非自主性和自主性提示有选择性地引导注意力。前者基于突出性，后者则依赖于意识

#在注意力机制的背景下，自主性提示被称为查询（query）
#给定任何查询，注意力机制通过注意力汇聚（attention pooling）将选择引导至感官输入（sensory inputs，例如中间特征表示）。
# 在注意力机制中，这些感官输入被称为值（value）。
# 更通俗的解释，每个值都与一个键（key）配对，这可以想象为感官输入的非自主提示。


#见P388,公式10.2.4！注意力汇聚（attention pooling）公式：f(x) =α(x, xi)yi,
#其中x是查询， (xi, yi)是键值对
#注意力汇聚是yi的加权平均。将查询x和键xi之间的关系建模为 注意力权重（attention weight） α(x, xi)
# 这个权重将被分配给每一个对应值yi


#可以通过设计注意力汇聚的方式，便于给定的查询Q（自主性提示）与键K（非自主性提示）进行匹配，这将引导得出最匹配的值V（感官输入）
#注意力机制通过注意力汇聚将查询Q（自主性提示）和键K（非自主性提示）结合在一起，实现对值V（感官输入）的选择倾向
#即选择把注意力放在哪个感官上


#注意力的可视化
import torch
from d2l import torch as d2l

#为了可视化注意力权重，需要定义一个show_heatmaps函数。其输入matrices的形状是（要显示的行数，要显示的列数，查询的数目，键的数目）
#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """显示矩阵热图"""
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    d2l.plt.savefig(f"/home2/wzc/d2l_learn/d2l-zh/learning_code/picture/my_plot_58.png", dpi=600)

#测试：仅当查询Q和键K相同时，注意力权重为1，否则为0
attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')




