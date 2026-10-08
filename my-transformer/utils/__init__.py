"""
utils 包 —— 训练与推理的辅助工具

    loss.py       标签平滑损失（论文 5.4 节）
    scheduler.py  Noam 学习率调度器（论文 5.3 节）
    decode.py     贪心解码与束搜索解码（推理阶段逐词生成）
"""

from .loss import LabelSmoothingLoss
from .scheduler import NoamScheduler, build_noam_optimizer
from .decode import greedy_decode, beam_search

__all__ = [
    "LabelSmoothingLoss",
    "NoamScheduler",
    "build_noam_optimizer",
    "greedy_decode",
    "beam_search",
]
