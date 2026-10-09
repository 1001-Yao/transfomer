"""
model/transformer.py —— 完整的 Transformer（论文 Figure 1）

把 embedding / encoder / decoder / 输出层组装在一起：

    源句 ids ──> Embedding ──> Encoder ──┐
                                         │  memory
    目标句 ids ─> Embedding ─> Decoder <──┘ ──> Linear(d_model -> tgt_vocab) ──> logits

两个实现细节值得注意：

1) 输出层与词嵌入共享权重（论文 3.4 节 "weight tying"）
   输出投影的权重矩阵形状是 (tgt_vocab_size, d_model)，
   目标词嵌入矩阵的形状正好也是 (tgt_vocab_size, d_model)，二者直接共享。
   好处：省参数、缓解小词表/低频词的过拟合。

2) 参数初始化
   论文没有详细写，官方代码用的是 Xavier 均匀初始化，
   并且所有偏置初始化为 0，LayerNorm 的 γ=1, β=0（已在 LayerNorm 里设定）。
"""

import torch
import torch.nn as nn

from .decoder import Decoder, DecoderLayer
from .embedding import TransformerEmbedding
from .encoder import Encoder, EncoderLayer
from .layers import LayerNorm
from .masks import combine_masks, make_causal_mask, make_padding_mask


class Transformer(nn.Module):
    """
    seq2seq Transformer。

    参数:
        src_vocab_size / tgt_vocab_size : 源/目标词表大小
        d_model      : 所有子层的隐藏维度（论文 512）
        n_heads      : 多头注意力的头数（论文 8）
        n_encoder_layers / n_decoder_layers : 编码器/解码器堆叠层数 N（论文 6）
        d_ff         : 前馈网络的中间维度（论文 2048）
        dropout      : dropout 概率（论文 0.1）
        max_pos_len  : 位置编码支持的最大长度
        padding_idx  : <pad> 的 id
        norm_first   : False = Post-LN（论文原版），True = Pre-LN
        tie_weights  : 是否共享输出层与目标词嵌入的权重
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_pos_len: int = 5000,
        padding_idx: int = 0,
        norm_first: bool = False,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.tgt_vocab_size = tgt_vocab_size

        # ---------- 1. 输入表示 ----------
        self.src_embed = TransformerEmbedding(
            src_vocab_size, d_model, max_len=max_pos_len,
            dropout=dropout, padding_idx=padding_idx,
        )
        self.tgt_embed = TransformerEmbedding(
            tgt_vocab_size, d_model, max_len=max_pos_len,
            dropout=dropout, padding_idx=padding_idx,
        )

        # ---------- 2. 编码器 / 解码器 ----------
        # 注意：clones 是在 Encoder/Decoder 内部做的，这里只传"一层"作为模板
        enc_layer = EncoderLayer(d_model, n_heads, d_ff, dropout, norm_first)
        dec_layer = DecoderLayer(d_model, n_heads, d_ff, dropout, norm_first)
        self.encoder = Encoder(enc_layer, n_encoder_layers, norm_first)
        self.decoder = Decoder(dec_layer, n_decoder_layers, norm_first)

        # ---------- 3. 输出层 ----------
        # 把 d_model 维的隐状态投影到词表大小的 logits 上
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # ---------- 4. 权重共享 ----------
        # 让输出层的权重矩阵就是目标词嵌入矩阵本身（同一个 Parameter 对象）
        if tie_weights:
            assert d_model == self.tgt_embed.tok_embed.embed.embedding_dim
            self.generator.weight = self.tgt_embed.tok_embed.embed.weight

        # ---------- 5. 初始化 ----------
        self._init_parameters()

    # ------------------------------------------------------------------
    # 掩码构造（对外暴露成方法，训练/推理脚本直接调用）
    # ------------------------------------------------------------------
    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """源句掩码：只屏蔽 <pad>。输入 (B, Ls) -> 输出 (B, 1, 1, Ls)"""
        return make_padding_mask(src, self.padding_idx)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        目标句掩码：屏蔽 <pad> **并且**屏蔽未来位置。
        输入 (B, Lt) -> 输出 (B, 1, Lt, Lt)
        """
        L = tgt.size(1)
        pad_mask = make_padding_mask(tgt, self.padding_idx)          # (B, 1, 1, L)
        causal = make_causal_mask(L, device=tgt.device)              # (1, 1, L, L)
        return combine_masks(pad_mask, causal)                       # (B, 1, L, L)

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """源句 -> memory。推理时只跑一次，然后复用（这是相对 RNN 的巨大优势）。"""
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor,
               tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """memory + 目标句前缀 -> 每个位置的词表 logits (B, Lt, V)"""
        x = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        return self.generator(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        训练时的前向传播。

        参数:
            src      : (B, Ls) 源句 id
            tgt      : (B, Lt) 目标句 id，**已经右移**（以 <bos> 开头，不含最后一个词）
            src_mask : (B, 1, 1, Ls)
            tgt_mask : (B, 1, Lt, Lt)

        返回:
            logits : (B, Lt, V)，第 t 个位置是"给定前 t 个词后，第 t+1 个词"的预测
                     （Teacher Forcing：训练时所有位置并行预测，
                      因为因果掩码保证了每个位置只能看到它左边的词）
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    # ------------------------------------------------------------------
    def _init_parameters(self):
        """
        Xavier 均匀初始化 + 偏置置零（对齐论文官方代码）。

        原理：让每一层输出的方差 ≈ 输入的方差，
        避免信号在深层网络里逐层放大或衰减。
        bias 初始化为 0 是因为对称性由权重打破就够了。
        """
        for p in self.parameters():
            if p.dim() > 1:                 # 权重矩阵（Linear / Embedding）
                nn.init.xavier_uniform_(p)
            else:                           # 偏置、LayerNorm 的 γ/β
                nn.init.constant_(p, 0.0)

        # 再单独处理 LayerNorm：γ 必须是 1，否则初始时归一化后的输出被乘成 0。
        # （上面那个循环把 γ 也置零了，这一步把它改回来。）
        for module in self.modules():
            if isinstance(module, LayerNorm):
                nn.init.constant_(module.gamma, 1.0)
                nn.init.constant_(module.beta, 0.0)
