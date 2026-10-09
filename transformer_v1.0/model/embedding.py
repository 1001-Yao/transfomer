"""
model/embedding.py —— 输入表示

Transformer 没有 RNN 也没有卷积，对序列顺序**完全没有感知**：
把输入顺序打乱，自注意力的输出只是跟着做同样的置换。
所以必须显式地把位置信息注入进去，也就是"位置编码"（论文 3.5 节）。

    TokenEmbedding     : id -> 稠密向量，并乘以 √d_model
    PositionalEncoding : 用 sin/cos 生成固定的位置向量，和词向量相加
    TransformerEmbedding: 上面两步的组合，后面加一层 dropout
"""

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    词嵌入。

    用 nn.Embedding 把 (B, L) 的整数 id 映射成 (B, L, d_model) 的向量。

    padding_idx：指定 <pad> 对应的行。该行的向量恒为 0，且不参与梯度更新。
    这对我们很有利——padding 位置的表示天然为零向量，不会给注意力贡献信息。

    为什么要乘以 √d_model ？
      因为位置编码的值域是 [-1, 1]，量级比随机初始化的词向量小。
      论文里把词向量放大 √d_model 倍（512 -> 22.6）来平衡两者的量级，
      让相加之后的表示中词义和位置都很明显。
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    正弦位置编码（论文 3.5 节，式 (3)(4)）：

        PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
        PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )

    直觉理解：
      - 每个维度 i 对应一个不同频率的正弦波，从高频到低频排列，
        拼起来像"二进制计数"的连续版本，能唯一地编码每个位置。
      - 由于 sin(a+b) 可以用 sin/cos 的线性组合表示，
        模型可以比较容易地学到"相对位置"的概念（PE[pos+k] 是 PE[pos] 的线性变换）。
      - 它不是学习出来的参数，而是固定的常量表，所以外推到更长序列
        也不会增加参数（当然外推效果有限）。

    实现要点：
      预先把 (max_len, d_model) 的表格算好，注册成 buffer。
      buffer 不是参数（不参与优化），但会跟着 .to(device) 一起搬移，
      也会被存进 state_dict（方便复现）。
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        assert d_model % 2 == 0, "d_model 必须是偶数：sin/cos 成对占用维度 0,1 / 2,3 / ..."

        self.dropout = nn.Dropout(dropout)

        # pe[pos, i] : 位置 pos 在第 i 个维度上的编码值,pe即为一个位置矩阵
        pe = torch.zeros(max_len, d_model)

        # position: (max_len, 1)，每个位置的下标 0,1,2,...
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 10000^(2i/d_model) 用 exp/log 形式算，避免数值溢出。
        # 等价于 1 / 10000^(2i/d_model)，i = 0..d_model/2-1
        div_term = torch.exp(
            # torch.arange(start, end, step)， math.log(10000.0)，其为自然对数lnx
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维用 sin
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维用 cos

        pe = pe.unsqueeze(0)                            # (1, max_len, d_model)，方便广播,因为会应用一批的矩阵，新增对应batch
        self.register_buffer("pe", pe)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数: x : (B, L, d_model) 已经过词嵌入（且乘以 √d_model）的表示
        返回:   (B, L, d_model)
        """
        L = x.size(1)
        assert L <= self.max_len, f"序列长度 {L} 超过了位置编码表的最大长度 {self.max_len}"

        # 词向量 + 位置向量（论文式 (3) 的前半步），然后做 dropout。
        # 论文 5.4 节："apply dropout to the sums of the embeddings and the
        # positional encodings"，所以 dropout 加在相加之后。
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """
    把"词嵌入 -> 乘 √d_model -> 加位置编码 -> dropout"打包，
    编码器和解码器各自持有一个实例（词表可以不同，但共享同一套位置编码公式）。
    """

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 5000,
                 dropout: float = 0.1, padding_idx: int = 0):
        super().__init__()
        self.tok_embed = TokenEmbedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pos_enc(self.tok_embed(x))
