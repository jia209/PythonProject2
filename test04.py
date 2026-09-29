"""
transformer手动实现
               --JZ
               2026.09.29,大二
"""
# 1.导包
import torch
import torch.nn as nn
import math

# 2.实现多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        :param d_model: 单个token的维度
        :param num_heads: 多头注意力机制的头数
        """
        super().__init__()                     # 初始化
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads        # 每个头的维度数
        self.W_q = nn.Linear(d_model, d_model) # Q
        self.W_k = nn.Linear(d_model, d_model) # K
        self.W_v = nn.Linear(d_model, d_model) # V
        self.W_o = nn.Linear(d_model, d_model) # 合并拼接后进行一次线性变换
    def forward(self, x, y=None,  mask=None):
        """
        :param x: 输入源数据x
        :param y: 根据是否输入y判断是否进行交叉注意力机制
        :param mask: 根据是否输入mask判断是否进行掩码注意力机制
        :return: 经多头注意力机制处理过后的数据
        """
        batch, seq_len, _ = x.shape
        Q = self.W_q(x).view(batch, seq_len, self.num_heads,self.d_k)
        if y is None:
            y=x
        batch_y, seq_len_y ,_= y.shape
        K=self.W_k(y).view(batch_y, seq_len_y, self.num_heads,self.d_k)
        V=self.W_v(y).view(batch_y, seq_len_y, self.num_heads,self.d_k)
        head_outputs=[]
        for head in range(self.num_heads):
            Q_head,K_head,V_head=Q[:,:,head,:],K[:,:,head,:],V[:,:,head,:]
            scores=Q_head@K_head.transpose(-1,-2)/math.sqrt(self.d_k)
            if mask is not None:
                scores=scores.masked_fill(mask==0, float('-inf'))
            weights=torch.softmax(scores,dim=-1)
            head_outputs.append(weights@V_head)
        return self.W_o(torch.cat(head_outputs,dim=-1))

# 3.定义前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        """
        :param d_model: 单个token的维度
        :param d_ff: 前馈神经网络的中间维度数
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff) # (batch,seq_len,d_model)->(batch,seq_len,d_ff)
        self.fc2 = nn.Linear(d_ff, d_model) # (batch,seq_len,d_ff)->(batch,seq_len,d_model)
        self.relu = nn.ReLU()               # Relu激活函数,非线性
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# 4.实现位置编码
def positional_encoding(seq_len, d_model):
    """
    :param seq_len: 序列长度（token数）
    :param d_model: 单个token的维度
    :return: 位置编码矩阵 (seq_len, d_model)
    """
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(seq_len).unsqueeze(1)     # (seq_len,1) ,增加了一个维度，方便下一步的广播机制
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))    # div_term是(d_model/2)，相乘自动广播得到(seq_len, d_model/2)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# 5.定义编码器层 （两个子层，注意力机制+前馈神经网络）
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        """
        :param d_model: 单个token的维度
        :param num_heads: 多头注意力机制的头数
        :param d_ff: 前馈神经网络的中间维度数
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)   # 注意力机制
        self.feed_forward = FeedForward(d_model, d_ff)            # 前馈神经网路
        self.layer_norm_1 = nn.LayerNorm(d_model)                 # 层归一化-1
        self.layer_norm_2 = nn.LayerNorm(d_model)                 # 层归一化-2
    def forward(self, x):
        x=self.layer_norm_1(x+self.attention(x))                  # 子层1：注意力机制+残差连接+层归一化
        x=self.layer_norm_2(x+self.feed_forward(x))               # 子层2：前馈神经网络+残差连接+层归一化
        return x

# 6.实现编码器
class Encoder(nn.Module):
    """
    Encoder 负责把一串离散的 token 编号，转换成连续的语义向量序列，供 Decoder 使用
    """
    def __init__(self, vocab_size,num_layers, d_model, num_heads, d_ff):
        """
        :param vocab_size:词表总大小，所有 token 编号范围 0 ~ vocab_size-1
        :param num_layers:一共要堆叠多少个 EncoderLayer
        :param d_model:单个token的维度
        :param num_heads:多头注意力机制的头数
        :param d_ff:前馈神经网络的中间维度数
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        # nn.ModuleList：专门存放多个 nn.Module 的列表。普通 list 里的子模块不会被 PyTorch 自动注册，参数不会被优化器更新。ModuleList 会注册所有子层参数。

    def forward(self, token_ids):
        """
        :param token_ids: 整数序列，shape [batch, seq_len]，就是一句话的 token 编号。
        :return: 经编码器处理过后的矩阵，编码器记忆
        """
        seq_len = token_ids.shape[-1]
        x = self.embedding(token_ids) + positional_encoding(seq_len, self.d_model)
        for layer in self.layers:
            x = layer(x)
        return x

# 7.定义解码器层，三层（掩码注意力，交叉注意力，前馈神经网络）
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        """
        :param d_model: 单个token的维度
        :param num_heads: 多头注意力机制的头数
        :param d_ff: 前馈神经网络的中间维度数
        """
        super().__init__()
        self.masked_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.layer_norm_1 = nn.LayerNorm(d_model)     # 子层1：掩码注意力机制+残差连接+层归一化
        self.layer_norm_2 = nn.LayerNorm(d_model)     # 子层2：交叉注意力机制+残差连接+层归一化
        self.layer_norm_3 = nn.LayerNorm(d_model)     # 子层3：前馈神经网络+残差连接+层归一化

    def forward(self, x, enc_output, mask):
        """
        :param x: 输入矩阵
        :param enc_output: 编码器记忆
        :param mask: 掩码标志
        :return: 处理过后的矩阵
        """
        x = self.layer_norm_1(x + self.masked_attn(x, mask=mask))
        x = self.layer_norm_2(x + self.cross_attn(x, enc_output))
        x = self.layer_norm_3(x + self.ffn(x))
        return x

# 8.实现解码器
class Decoder(nn.Module):
    """
    Decoder 负责把编码器记忆，转换成目标语言的 token 概率分布"""
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers):
        """
        :param vocab_size: 词表总大小，所有 token 编号范围 0 ~ vocab_size-1
        :param d_model: 单个token的维度
        :param num_heads: 多头注意力机制的头数
        :param d_ff: 前馈神经网络的中间维度数
        :param num_layers: 一共要堆叠多少个 DecoderLayer
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

    def forward(self, token_ids, enc_output, mask):
        """
        :param token_ids: 整数序列，shape [batch, seq_len]，就是一句话的 token 编号。
        :param enc_output: 编码器记忆
        :param mask: 掩码标志
        :return: 解码器输出
        """
        seq_len = token_ids.shape[-1]
        x = self.embedding(token_ids) + positional_encoding(seq_len, self.d_model)
        for layer in self.layers:
            x = layer(x, enc_output, mask)
        return x

# 9.正式组装transformer
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers):
        """
        :param vocab_size: 词表总大小，所有 token 编号范围 0 ~ vocab_size-1
        :param d_model: 单个token的维度
        :param num_heads: 多头注意力机制的头数
        :param d_ff: 前馈神经网络的中间维度数
        :param num_layers: 一共要堆叠多少个 DecoderLayer
        """
        super().__init__()
        self.encoder = Encoder(vocab_size, num_layers, d_model, num_heads, d_ff)
        self.decoder = Decoder(vocab_size, d_model, num_heads, d_ff, num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)  # 向量 -> 词的概率

    def forward(self, src_ids, tgt_ids, tgt_mask):
        """
        :param src_ids: 源语言token编号
        :param tgt_ids: 目标语言token编号
        :param tgt_mask: 目标语言掩码
        :return: 预测的目标语言token概率
        """
        enc_output = self.encoder(src_ids)
        dec_output = self.decoder(tgt_ids, enc_output, tgt_mask)
        return self.output_layer(dec_output)  # (batch, tgt_len, vocab_size)












