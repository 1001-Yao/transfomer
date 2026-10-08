"""
model/masks.py —— 掩码（Mask）的生成

Transformer 里一共有两种掩码，它们的唯一作用就是在 softmax 之前把某些位置
的注意力分数置成极小的数（-inf 的近似），使得 softmax 之后这些位置的权重约等于 0。

1) Padding Mask（填充掩码）
   batch 里不同句子长度不一样，短的句子要补 <pad> 对齐。
   这些 <pad> 是没有意义的，不能参与注意力计算。
   形状 (B, 1, 1, L)：对每个样本的每一个被查询位置，屏蔽掉 K/V 里的 pad 位置。

2) Causal Mask / Look-ahead Mask（因果掩码 / 未来掩码）
   解码器是自回归的：预测第 t 个词时只能看到第 0..t 个词，不能看到未来。
   形状 (1, 1, L, L)：一个上三角矩阵，True 表示"这个位置要屏蔽"。
   论文 3.2.3 节的 "masking ... to prevent positions from attending to
   subsequent positions"。

约定（整个项目统一）：
    mask 是 torch.bool 张量，**True 表示该位置需要被屏蔽（不参与注意力）**。
    这个约定和 PyTorch 官方 nn.MultiheadAttention 的 attn_mask 一致。

注意力分数张量的形状是 (B, n_heads, L_query, L_key)，
所以掩码只要能广播到这个形状即可：
    padding mask : (B, 1, 1, L_key)
    causal  mask : (1, 1, L_query, L_key)
    两者按位或   : (B, 1, L_query, L_key)
"""

import torch


def make_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    生成填充掩码。

    参数:
        seq     : (B, L) 的 token id 张量
        pad_idx : <pad> 的 id

    返回:
        (B, 1, 1, L) 的 bool 张量，True 表示该位置是 <pad>，需要屏蔽。

    为什么中间要补两个维度？
        因为要和 (B, n_heads, L_q, L_k) 的分数矩阵广播相乘/填充，
        补成 (B, 1, 1, L) 后，第 2 维(查询位置)和第 3 维(键位置)会自动广播。
    """
    # seq == pad_idx 得到 (B, L) 的 bool，再在倒数第二、第三维各插一个维度
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)


def make_causal_mask(size: int, device=None) -> torch.Tensor:
    """
    生成因果（未来）掩码，形状 (1, 1, size, size)。

    矩阵的第 i 行第 j 列表示：查询位置 i 能否看到键位置 j。
    当 j > i 时（即未来位置）需要屏蔽，所以是**严格上三角**。

    torch.triu(x, diagonal=1) 保留主对角线以上的元素（不含主对角线），
    正好就是 j > i 的部分。

    例：size = 4 时
        [[F, T, T, T],
         [F, F, T, T],
         [F, F, F, T],
         [F, F, F, F]]     F = False = 允许，T = True = 屏蔽
    """
    # 先用 float 造矩阵再转 bool，兼容性最好
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)


def combine_masks(*masks: torch.Tensor) -> torch.Tensor:
    """
    把多个掩码按位或合并（任意一个为 True 就屏蔽）。

    典型用法：解码器自注意力需要同时屏蔽 <pad> 和未来位置：
        tgt_mask = combine_masks(make_padding_mask(tgt, pad), make_causal_mask(L))

    利用广播，(B,1,1,L) | (1,1,L,L) 会得到 (B,1,L,L)。
    """
    result = masks[0]
    for m in masks[1:]:
        result = result | m
    return result
