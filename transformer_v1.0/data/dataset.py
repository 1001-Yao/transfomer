"""
data/dataset.py —— 数据集与批处理

包含：
    TranslationDataset : 把 (源句 tokens, 目标句 tokens) 转成 id 张量
    make_collate_fn    : 把一批变长句子 padding 到同一长度

关于 padding 和"右移"（shifted right）
--------------------------------------
源句：  a b c <eos>
目标句：<bos> d c b a <eos>      （以 reverse 任务为例）

训练时我们把目标句拆成两段：
    decoder 输入 (tgt_in)  : <bos> d c b a     —— 前缀，不含最后一个词
    decoder 标签 (tgt_out) : d c b a <eos>     —— 右移一位，作为每个位置的答案

这正是论文 3.1 节说的 "we shift the output embedding by one position"，
对应代码里的 tgt[:, :-1] 和 tgt[:, 1:]。
这样一次前向传播就能对**所有位置**并行计算损失（Teacher Forcing），
而因果掩码保证了位置 t 只能看到它左边的内容，不会提前看到答案。
"""

import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    """
    PyTorch Dataset：__len__ 返回样本数，__getitem__ 返回第 i 个样本。

    源句和目标句使用各自的词表（真实翻译任务里中英词表不同；
    本项目的玩具任务里传同一个词表对象即可）。
    """

    def __init__(self, pairs, src_vocab, tgt_vocab):
        """
        参数:
            pairs     : [(源句 token 列表, 目标句 token 列表), ...]
            src_vocab : 源端词表
            tgt_vocab : 目标端词表
        """
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        src_tokens, tgt_tokens = self.pairs[index]

        # 源句：末尾加 <eos>，不加 <bos>（编码器不需要起始符号）
        src_ids = self.src_vocab.encode(src_tokens, add_bos=False, add_eos=True)
        # 目标句：两头都加，后面再切成 tgt_in / tgt_out
        tgt_ids = self.tgt_vocab.encode(tgt_tokens, add_bos=True, add_eos=True)

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


def make_collate_fn(pad_idx: int = 0):
    """
    生成一个 collate_fn。DataLoader 会用它把一批样本拼成一个 batch。

    collate_fn 接收的是 __getitem__ 返回值组成的列表 [(src1,tgt1), (src2,tgt2), ...]，
    我们要把长度不一的句子补齐到本 batch 内的最大长度。

    补到"batch 内最大长度"而不是全局最大长度，是为了省显存：
    同一个 batch 里最长的那条决定矩阵形状，多余的列全是 <pad>，纯属浪费算力。

    返回:
        src : (B, Ls_max) 源句 id，不足处补 pad_idx
        tgt : (B, Lt_max) 目标句 id，不足处补 pad_idx
    """
    def collate_fn(batch):
        src_list, tgt_list = zip(*batch)         # unzip 成两个 tuple
        batch_size = len(src_list)
        max_src_len = max(s.size(0) for s in src_list)
        max_tgt_len = max(t.size(0) for t in tgt_list)

        # 先全部填成 <pad>，再把真实内容拷进去
        src = torch.full((batch_size, max_src_len), pad_idx, dtype=torch.long)
        tgt = torch.full((batch_size, max_tgt_len), pad_idx, dtype=torch.long)

        for i, (s, t) in enumerate(zip(src_list, tgt_list)):
            src[i, :s.size(0)] = s
            tgt[i, :t.size(0)] = t

        return src, tgt

    return collate_fn
