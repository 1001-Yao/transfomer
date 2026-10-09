"""
model/encoder.py —— 编码器（论文 3.1 节左半部分）

结构（Figure 1 左侧）：

    ┌──────────────────────────────┐
    │  Multi-Head Self-Attention   │  ← 源句自己注意自己（需要 padding mask）
    │  Add & Norm                  │
    ├──────────────────────────────┤
    │  Feed Forward                │
    │  Add & Norm                  │
    └──────────────────────────────┘   × N 层

编码器的任务是：把源句 (B, Ls, D) 编码成一组同样长度的、富含上下文信息的
表示 memory (B, Ls, D)，供解码器做交叉注意力时查询。
"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .layers import LayerNorm, PositionwiseFeedForward, SublayerConnection, clones


class EncoderLayer(nn.Module):
    """
    单个编码器层 = 自注意力子层 + 前馈子层，每个子层外面套一层
    "残差连接 + LayerNorm"（即 SublayerConnection）。
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, norm_first: bool = False):
        super().__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        # 两个子层各需要一个 SublayerConnection（参数独立，所以不能共用同一个实例）
        self.sublayer = clones(SublayerConnection(d_model, dropout, norm_first), 2)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x        : (B, Ls, d_model)，源句的嵌入表示
            src_mask : (B, 1, 1, Ls)，源句的 padding 掩码（True = 屏蔽 <pad>）

        返回:
            (B, Ls, d_model)
        """
        # 子层 1：自注意力。Q = K = V = x（这就是"自"注意力的含义）
        # 用 lambda 把 mask 固定进去，交给 SublayerConnection 去做残差和归一化
        x = self.sublayer[0](x, lambda t: self.self_attn(t, t, t, src_mask))
        # 在语法上 会调用sublayer（SublayerConnection）的forward方法
        # lambda 是 Python 里的"匿名函数"——一个没有名字的小函数：lambda 参数: 返回值表达式
        # 子层 2：逐位置前馈网络
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """
    N 层编码器堆叠。

    论文在最后一层之后还加了一次 LayerNorm（Figure 1 里 Encoder 框顶部那个 Norm）。
    在 Post-LN 模式下，每个子层结尾已经归一化过了，再补一次是冗余的
    （很多实现仍然保留它，效果几乎一样）；在 Pre-LN 模式下，这个最终归一化是
    **必须**的，否则最后一层的输出没有被归一化过。
    所以这里只在 norm_first=True 时才添加。
    """

    def __init__(self, layer: EncoderLayer, n_layers: int, norm_first: bool = False):
        super().__init__()
        self.layers = clones(layer, n_layers)
        self.final_norm = LayerNorm(layer.d_model) if norm_first else None

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x
