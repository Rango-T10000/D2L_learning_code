#Transformer模型完全基于注意力机制，没有任何卷积层或循环神经网络层 
#尽管Transformer最初是应用于在文本数据上的序列到序列学习

#从宏观角度来看， Transformer的编码器和解码器都是由多个相同的层叠加而成的
#并且层中使用了残差连接residual connection和层规范化layer normalization

#Transformer的编码器是由多个相同的层叠加而成的，每个层都有两个子层（子层表示为sublayer）
#第一个子层是多头自注意力（multi‐head self‐attention）汇聚；在计算编码器的自注意力时，查询、键和值都来自前一个编码器层的输出
#第二个子层是基于位置的前馈网络（positionwise feed‐forward network）

#Transformer解码器也是由多个相同的层叠加而成的
#在解码器自注意力中，查询、键和值都来自上一个解码器层的输出。但是，
# 解码器中的每个位置只能考虑该位置之前的所有位置。这种掩蔽（masked）注意力保留了自回归（auto‐regressive）属性，
# 确保预测仅依赖于已生成的输出词元。
#编码器－解码器注意力（encoder‐decoder attention）层。
# 在编码器－解码器注意力中，查询来自前一个解码器层的输出，而键和值来自整个编码器的输出

import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

#基于位置的前馈网络（positionwise feed‐forward network）
#基于位置的前馈网络对序列中的所有位置的表示进行变换时使用的是同一个多层感知机（MLP）
#@save
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
    
#测试：
#输入X: （批量大小，时间步数或序列长度，隐单元数或特征维度）
#输出：（批量大小，时间步数或序列长度， ffn_num_outputs）
ffn = PositionWiseFFN(4, 4, 8)
ffn.eval()
print(ffn(torch.ones((2, 3, 4))).shape)


#加法和规范化（add&norm）
#加法指的是残差连接
#这里是层规范化，和之前学的批量规范化的目标相同
#但层规范化是基于特征维度进行规范化
#@save
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
    
#测试：
add_norm = AddNorm([3, 4], 0.5)
add_norm.eval()
print(add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape)

#编码器中的一个层
#@save
class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


#测试：
X = torch.ones((2, 100, 24))       #输入形状是(batch_size, 时间步数/序列长度，隐单元数或特征维度d)
valid_lens = torch.tensor([3, 2])  #因为输入的batch_size是2，这里给出两个数据各自的有效长度
encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)  #[100, 24]是norm_shape
encoder_blk.eval()
print(encoder_blk(X, valid_lens).shape)

#Transformer编码器
#@save
class TransformerEncoder(d2l.Encoder):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

#测试：
encoder = TransformerEncoder(200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
encoder.eval()
print("输入的形状是（批量大小，时间步数目/序列长度）：", (torch.ones((2, 100)).shape))
print("输出的形状是（批量大小，时间步数目/序列长度，num_hiddens）：")
print(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape)
#理解成输入是X,2个batch的数据，每个数据是序列，长度为100，即100个token组成的序列
#这个num_hiddens为24，就是通过词嵌入把每个token编码为长度为24的向量
#最终Encoder的输出形状就是（2，100，24）即对原本的输入数据（2，100）完成了编码


#解码器
#Transformer解码器也是由多个相同的层组成。
#在DecoderBlock类中实现的每个层包含了三个子层：解码器自注意力、“编码器‐解码器”注意力和基于位置的前馈网络

#关于掩蔽多头解码器自注意力层（第一个子层）：
#在掩蔽多头解码器自注意力层（第一个子层）中，查询、键和值都来自上一个解码器层的输出
#关于序列到序列模型（sequence‐to‐sequence model），在训练阶段，其输出序列的所有位置（时间步）的词元都是已知的；
# 然而，在预测阶段，其输出序列的词元是逐个生成的。因此，在任何解码器时间步中，只有生成的词元才能用于解码器的自注意力计算中。
# 为了在解码器中保留自回归的属性，其掩蔽自注意力设定了参数dec_valid_lens，
# 以便任何查询都只会与解码器中所有已经生成词元的位置（即直到该查询位置为止）进行注意力计算

#再解释：
#为了解决这个问题，我们使用掩蔽技术（masking）。
#在解码器中，我们引入一个掩蔽向量（mask vector）来指示哪些位置是有效的（已生成的词元）和哪些位置是无效的（尚未生成的词元）
#总之，掩蔽技术在解码器中用于限制自注意力计算的范围，
#确保模型在生成每个词元时只依赖于已生成的词元，从而保持自回归的属性。这有助于提高生成的结果的质量和一致性。
class DecoderBlock(nn.Module):
    """解码器中第i个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

#为了便于在“编码器－解码器”注意力中进行缩放点积计算和残差连接中进行加法计算，
#编码器和解码器的特征维度都是num_hiddens
#测试：
decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
decoder_blk.eval()
X = torch.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
print(decoder_blk(X, state)[0].shape)

#Transformer解码器
#由num_layers个DecoderBlock实例组成
#最后，通过一个全连接层计算所有vocab_size个可能的输出词元的预测值
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


#训练：进行序列到序列的学习
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
# print(net)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


#推理/预测：
#使用Transformer模型将一些英语句子翻译成法语，并且计算它们的BLEU分数
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
