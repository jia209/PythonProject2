# Transformer 手动实现

从零手动实现 Transformer 模型（基于 PyTorch），不依赖任何现成的 Transformer 库。本项目用于深度学习学习，帮助理解 Transformer 每个组件的工作原理。

> 作者：JZ（大二）　日期：2026.09.29

## 项目简介

用 PyTorch 手动实现了 Transformer 的核心结构：多头注意力、位置编码、前馈网络、编码器、解码器等，逐层搭建，适合入门理解 Transformer 的内部原理。

## 已实现的组件

| 组件 | 说明 |
| --- | --- |
| `MultiHeadAttention` | 多头注意力机制，支持自注意力 / 交叉注意力 / 掩码注意力 |
| `FeedForward` | 前馈神经网络（两层线性 + ReLU） |
| `positional_encoding` | 正弦 / 余弦位置编码 |
| `EncoderLayer` / `Encoder` | 编码器层与编码器（自注意力 + 前馈 + 残差连接 + LayerNorm） |
| `DecoderLayer` / `Decoder` | 解码器层与解码器（掩码自注意力 + 交叉注意力 + 前馈） |
| `Transformer` | 组装完整的 Transformer 模型 |

## 环境依赖

- Python 3.11
- [PyTorch](https://pytorch.org/)

```bash
pip install torch
```

## 文件结构

```
.
└── test04.py    # Transformer 完整实现
```

## 使用说明

```python
import torch
from test04 import Transformer

# 注意：d_model 需要能被 num_heads 整除
model = Transformer(
    vocab_size=13,   # 词表大小
    d_model=512,     # token 维度
    num_heads=8,     # 注意力头数
    d_ff=2048,       # 前馈网络中间维度
    num_layers=6,    # 层数
)

src_ids = torch.randint(0, 13, (2, 10))   # 源序列 [batch, seq_len]
tgt_ids = torch.randint(0, 13, (2, 10))   # 目标序列 [batch, seq_len]
tgt_mask = ...                             # 解码器掩码（上三角矩阵）

out = model(src_ids, tgt_ids, tgt_mask)   # 输出 [batch, tgt_len, vocab_size]
```

> 注：本项目为结构实现的学习项目，暂未包含训练脚本和测试用例。

## 参考资料

- 论文：*Attention Is All You Need*

## 最后

本人计算机专业大二学生，从一开始的机器学习到如今的Transformer框架，全部基于自学，感谢B站上的人工智能相关网课以及各类AI的辅助学习，让自己的能力得到不断提高。
由于本人是初学者，对于Transformer框架的一些细节理解并不深入，项目存在很多改进与完善的空间，欢迎各位前来指正，谢谢！


