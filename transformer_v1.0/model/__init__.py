"""
model 包 —— Transformer 模型的全部组件

模块划分（每个文件只负责一件事）：
    masks.py       掩码的生成（Padding Mask / Causal Mask）
    attention.py   缩放点积注意力、多头注意力
    layers.py      LayerNorm、前馈网络、残差连接（子层连接器）、clones 工具
    embedding.py   词嵌入 + 位置编码
    encoder.py     编码器层 + 编码器
    decoder.py     解码器层 + 解码器
    transformer.py 把上面所有东西组装成完整的 Transformer

数据在上图中的流向（B = batch, L = 序列长度, D = d_model）：

    源句 ids (B, Ls)
        └─> TransformerEmbedding ─> (B, Ls, D)
              └─> Encoder(6 × EncoderLayer) ─> memory (B, Ls, D)
                                                  │
    目标句 ids (B, Lt)                             │
        └─> TransformerEmbedding ─> (B, Lt, D)     │
              └─> Decoder(6 × DecoderLayer) <──────┘  (cross-attention 读 memory)
                    └─> Linear(D -> V) ─> logits (B, Lt, V)
"""

from .masks import make_padding_mask, make_causal_mask, combine_masks
from .attention import MultiHeadAttention, scaled_dot_product_attention, NEG_INF
from .layers import LayerNorm, PositionwiseFeedForward, SublayerConnection, clones
from .embedding import TokenEmbedding, PositionalEncoding, TransformerEmbedding
from .encoder import EncoderLayer, Encoder
from .decoder import DecoderLayer, Decoder
from .transformer import Transformer

__all__ = [
    "make_padding_mask",
    "make_causal_mask",
    "combine_masks",
    "MultiHeadAttention",
    "scaled_dot_product_attention",
    "NEG_INF",
    "LayerNorm",
    "PositionwiseFeedForward",
    "SublayerConnection",
    "clones",
    "TokenEmbedding",
    "PositionalEncoding",
    "TransformerEmbedding",
    "EncoderLayer",
    "Encoder",
    "DecoderLayer",
    "Decoder",
    "Transformer",
]
