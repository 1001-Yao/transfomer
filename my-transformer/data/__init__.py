"""
data 包 —— 词表与数据集

    vocab.py     词表：token <-> id 的双向映射，负责处理 4 个特殊符号
    toy_data.py  玩具平行语料生成器（copy / reverse / sort 三个任务）
    dataset.py   PyTorch Dataset 与 collate_fn（把变长句子 padding 成 batch）
"""

from .vocab import Vocabulary
from .toy_data import build_toy_corpus
from .dataset import TranslationDataset, make_collate_fn

__all__ = ["Vocabulary", "build_toy_corpus", "TranslationDataset", "make_collate_fn"]
