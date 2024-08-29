import torch
from torch import nn
from d2l import torch as d2l

#Splitting an image into patches and linearly projecting these flattened patches 
# can be simplified as a single convolution operation, 
# where both the kernel size and the stride size are set to the patch size.
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size, stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)

img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = torch.zeros(batch_size, 3, img_size, img_size)  #3是图片有3个channel
# d2l.check_shape(patch_emb(X), (batch_size, (img_size//patch_size)**2, num_hiddens))
print(X.shape)
print("理论上的size: ", (batch_size, (img_size//patch_size)**2, num_hiddens))
print("实际的size: ", patch_emb(X).shape)

#Vision Transformer Encoder中的MLP和原本的Transformer中的FFN不一样
#不同之处是：
#1.使用的是Gaussian error linear unit (GELU)作为激活函数，而不是ReLU
#2.每一层FC都用了Dropout
#Vision Transformer中的MLP（类似对应原本的ffn）
class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(self.dense1(x)))))

#原本的Transformer使用的是post-normalization (“add & norm”）在多头注意力计算后
#在vit中，使用pre-normalization design，即在多头注意力计算前进行
class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X, valid_lens=None):
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))

X = torch.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 24, 48, 8, 0.5)
encoder_blk.eval()
# d2l.check_shape(encoder_blk(X), X.shape)
print(X.shape)
print(encoder_blk(X).shape)


#合起来就是：
class ViT(d2l.Classifier):
    """Vision Transformer."""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1,
                 use_bias=False, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))   #这个特殊token就是自己建立的一个参数
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),nn.Linear(num_hiddens, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])

#训练
# img_size, patch_size = 96, 16
# num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
# emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
# model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
#             num_blks, emb_dropout, blk_dropout, lr)
# trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
# data = d2l.FashionMNIST(batch_size=128, resize=(img_size, img_size))
# trainer.fit(model, data)