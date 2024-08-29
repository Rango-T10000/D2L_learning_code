#典型的序列数据：文本
#解析文本的常见预处理步骤：
# 1. 将文本作为字符串加载到内存中。
# 2. 将字符串拆分为词元（如单词和字符）。
# 3. 建立一个词表，将拆分的词元映射到数字索引。
# 4. 将文本转换为数字索引序列，方便模型操作。

#将文本数据映射为词元，所以将这些词元可以视为一系列离散的观测，例如单词或字符
#假设长度为T的文本序列中的词元依次为x1, x2, . . . , xT。
# 于是， xt（1 ≤ t ≤ T）可以被认为是文本序列在时间步t处的观测或标签。
# 在给定这样的文本序列时，语言模型（language model）的目标是估计序列的联合概率P (x1, x2, . . . , xT )
#利用公式：P (x1, x2, x3 ) = P (x1)P (x1|x2)P (x1,x2|x3)
#例子：P(deep, learning, is, fun) = P(deep)P(learning | deep)P(is | deep, learning)P(fun | deep, learning, is)
#为了训练语言模型，我们需要计算单词的概率，以及给定前面几个单词后出现某个单词的条件概率。这些概率本质上就是语言模型的参数

import collections
import re
from d2l import torch as d2l

#读取数据集：我们从H.G.Well的时光机器99中加载文本
#保存在/home2/wzc/d2l_learn/d2l-zh/learning_code/data中
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])

#词元(token)化：即切成token
#词元（token）是文本的基本单位
#定义一个函数返回一个由词元列表组成的列表，其中的每个词元都是一个字符串（string）
def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
print("\n")

#词表/字典：
#词元token的类型是字符串，而模型需要的输入是数字
#构建一个字典，通常也叫做词表（vocabulary），用来将字符串类型的词元映射到从0开始的数字索引中
# 理解成这个字典(存储的是这个一一对应的一个表，按这个表就可以把一个token用对应的数字表示)
#先将训练集中的所有文档合并在一起，对它们的唯一词元进行统计，得到的统计结果称之为语料（corpus）
#根据每个唯一词元的出现频率，为其分配一个数字索引
#很少出现的词元通常被移除，这可以降低复杂性
#语料库中不存在或已删除的任何词元都将映射到一个特定的未知词元“<unk>”
#增加一个列表，用于保存那些被保留的词元，例如：填充词元（“<pad>”）；序列开始词元（“<bos>”）；序列结束词元（“<eos>”）
class Vocab:  #@save
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


#使用时光机器数据集作为语料库来构建词表，然后打印前几个高频词元及其索引
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])


#利用我们建立好的token和数字的映射关系，即词表/字典
# 将每一条文本行转换成一个数字索引列表
for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])
print("\n")


#在使用上述函数时，我们将所有功能打包到load_corpus_time_machine函数中，
# 该函数返回corpus（词元索引列表）和vocab（时光机器语料库的词表）
def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char') #为了简化后面章节中的训练，我们使用字符（而不是单词）实现文本词元化
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab))
print(vocab.token_to_idx)



