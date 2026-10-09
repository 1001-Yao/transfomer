"""
model/decoder.py —— 解码器（论文 3.1 节右半部分）

结构（Figure 1 右侧）：

    ┌────────────────────────────────────┐
    │  Masked Multi-Head Self-Attention  │ ← 只能看到已生成的前缀（因果掩码 + padding 掩码）
    │  Add & Norm                        │
    ├────────────────────────────────────┤
    │  Multi-Head Cross-Attention        │ ← Q 来自解码器，K/V 来自编码器输出 memory
    │  Add & Norm                        │    （需要屏蔽源句的 <pad>）
    ├────────────────────────────────────┤
    │  Feed Forward                      │
    │  Add & Norm                        │
    └────────────────────────────────────┘   × N 层

和第 1 个子层的关键区别：
    自注意力：Q = K = V = 解码器输入，必须加**因果掩码**，
              否则预测第 t 个词时会"偷看"到答案（标签泄漏）。
    交叉注意力：Q 是解码器当前状态，K/V 是编码器的 memory。
              它的掩码用的是**源句的 padding 掩码**，形状 (B,1,1,Ls)。
"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .layers import LayerNorm, PositionwiseFeedForward, SublayerConnection, clones


class DecoderLayer(nn.Module):
    """单个解码器层 = 自注意力 + 交叉注意力 + 前馈，三个子层。"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, norm_first: bool = False):
        super().__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        # 三个子层，各配一个 SublayerConnection
        self.sublayer = clones(SublayerConnection(d_model, dropout, norm_first), 3)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x        : (B, Lt, d_model)，目标句（已右移）的嵌入表示
            memory   : (B, Ls, d_model)，编码器的输出
            src_mask : (B, 1, 1, Ls)，源句 padding 掩码 —— 给交叉注意力用
            tgt_mask : (B, 1, Lt, Lt)，因果 + 目标 padding 掩码 —— 给自注意力用

        返回:
            (B, Lt, d_model)
        """
        # 子层 1：掩码自注意力（Masked Self-Attention）
        x = self.sublayer[0](x, lambda t: self.self_attn(t, t, t, tgt_mask))

        # 子层 2：交叉注意力（Encoder-Decoder Attention）
        #   注意参数顺序是 (query, key, value)：
        #   query 用 x（解码器状态），key 和 value 都用 memory（编码器输出）。
        #   这就是"解码器从源句中检索信息"的地方，也是机器翻译里对齐关系所在之处。
        x = self.sublayer[1](x, lambda t: self.cross_attn(t, memory, memory, src_mask))

        # 子层 3：前馈网络
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """N 层解码器堆叠，末尾的 LayerNorm 说明同 Encoder。"""

    def __init__(self, layer: DecoderLayer, n_layers: int, norm_first: bool = False):
        super().__init__()
        self.layers = clones(layer, n_layers)
        self.final_norm = LayerNorm(layer.d_model) if norm_first else None

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x
