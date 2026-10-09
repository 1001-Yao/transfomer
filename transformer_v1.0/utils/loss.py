"""
utils/loss.py —— 标签平滑损失（论文 5.4 节）

普通交叉熵把真实类别的目标概率设成 1（one-hot 硬标签）：

    L = -log p(y_true)

标签平滑把它改成"留一点概率给其他类别"：

    y_smooth = (1 - ε)           当 k = 真实类别
             = ε / (V - 2)       当 k 是其他类别
             = 0                 当 k 是 <pad>（<pad> 不参与）

    这里 V-2 是因为要排除 <pad> 和真实类别本身：
    剩下 V-2 个类别均分 ε 的平滑质量。

    L = -Σ_k y_smooth[k] · log p(k)

论文为什么用它（原文 5.4 节）：
    1. 让模型不要对自己的预测过于自信，缓解过拟合；
    2. 提升 BLEU 分数，虽然让模型的"困惑度(perplexity)"变差——
       因为困惑度的定义和硬标签绑在一起，平滑后的模型本来就不该追求极低的困惑度。

注意 <pad> 的处理要仔细：
    - 目标分布里 <pad> 那一列必须是 0（不能让模型去学习预测 padding）；
    - 目标本身就是 <pad> 的那些位置，整行置 0，彻底不计入损失。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    参数:
        vocab_size  : 词表大小 V
        padding_idx : <pad> 的 id
        smoothing   : 平滑系数 ε，论文用 0.1；设为 0 就退化成普通交叉熵
    """

    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing   # 真实类别保留的概率 1 - ε

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        参数:
            logits : (B, L, V) 模型输出，**未经过 softmax** 的原始分数
            target : (B, L)    每个位置的真实词 id/标签

        返回:
            标量损失（对非 <pad> 位置取平均）
        """
        V = logits.size(-1) # 词表大小

        # 把 batch 和长度两维压平，问题就变成"一堆独立的 V 分类任务"
        logits = logits.reshape(-1, V)              # (B*L, V)
        target = target.reshape(-1)                 # (B*L,)

        # log_softmax 比 "softmax 再取 log" 在数值上稳定得多
        log_probs = F.log_softmax(logits, dim=-1)   # (B*L, V)

        with torch.no_grad():
            # 构造平滑后的目标分布，形状同样为 (B*L, V)
            dist = torch.full_like(log_probs, self.smoothing / (V - 2))
            # scatter_ 把真实类别那一列填成 1 - ε
            dist.scatter_(1, target.unsqueeze(1), self.confidence)
            # <pad> 列永远为 0（模型不该学着去预测 padding）
            dist[:, self.padding_idx] = 0.0
            # 真实标签本身就是 <pad> 的位置，整行置 0（这些位置不计损失）
            dist[target == self.padding_idx] = 0.0

        # 逐位置、逐类别加权求和，再取负号
        loss = -(dist * log_probs).sum()

        # 除以"有效 token 数"做平均，避免损失大小受 padding 比例影响。
        # clamp(min=1) 防止极端情况下（整个 batch 全是 pad）除以 0。
        num_valid = (target != self.padding_idx).sum().clamp(min=1)
        return loss / num_valid
