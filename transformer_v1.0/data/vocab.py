"""
data/vocab.py —— 词表

负责 token（字符串）和 id（整数）之间的双向映射。

4 个特殊符号固定占用 0~3 号 id，顺序不能乱：

    id 0 : <pad>  填充符号。为了让 nn.Embedding(padding_idx=0) 生效，
                   也为了让掩码函数可以直接用 (x == 0) 判断，必须是 0。
    id 1 : <unk>  未登录词
    id 2 : <bos>  Begin Of Sentence，解码器的起始输入
    id 3 : <eos>  End Of Sentence，解码器的终止输出

真实的 NLP 项目里词表会有几万到十几万个 token，这里因为用的是玩具任务，
词表只有十几个字符。
"""

from collections import Counter


class Vocabulary:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    def __init__(self):
        # idx2token 是列表，token2idx 是字典，两者互为逆映射。
        # 用列表而不是字典来存 id->token，是因为 id 就是从 0 开始的连续整数。
        self.idx2token = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        self.token2idx = {t: i for i, t in enumerate(self.idx2token)}

    # ------------------------------------------------------------------
    # 特殊符号的 id（用 property 暴露，读起来更清楚）
    # ------------------------------------------------------------------
    @property
    def pad_idx(self) -> int:
        return self.token2idx[self.PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.token2idx[self.UNK_TOKEN]

    @property
    def bos_idx(self) -> int:
        return self.token2idx[self.BOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[self.EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.idx2token)

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    def add_token(self, token: str) -> int:
        """把一个 token 加进词表，返回它的 id；已存在则直接返回。"""
        if token not in self.token2idx:
            self.token2idx[token] = len(self.idx2token)
            self.idx2token.append(token)
        return self.token2idx[token]

    def build(self, token_seqs, min_freq: int = 1) -> "Vocabulary":
        """
        从若干条 token 序列中统计并构建词表。

        参数:
            token_seqs : 可迭代对象，每个元素是一句话的 token 列表
            min_freq   : 出现次数低于该值的 token 会被丢弃（变成 <unk>）

        为了结果可复现，按 （频次降序, 首次出现顺序） 排序后再加入，
        避免因为 set/dict 的遍历顺序不同导致两次运行得到不同的 id。
        """
        counter = Counter()
        for seq in token_seqs:
            counter.update(seq)

        # 按频次降序排序；频次相同的保持原有相对顺序（Counter 保序，sorted 稳定）
        for token, freq in sorted(counter.items(), key=lambda kv: -kv[1]):
            if freq >= min_freq:
                self.add_token(token)
        return self

    # ------------------------------------------------------------------
    # 编解码
    # ------------------------------------------------------------------
    def encode(self, tokens, add_bos: bool = False, add_eos: bool = False):
        """
        token 列表 -> id 列表。

        add_bos: 在开头插入 <bos>（只对解码器输入需要）
        add_eos: 在结尾插入 <eos>（源句和目标句都需要，作为序列结束标志，
                 也是训练时"停止生成"的监督信号）
        """
        ids = [self.token2idx.get(t, self.unk_idx) for t in tokens]
        if add_bos:
            ids = [self.bos_idx] + ids
        if add_eos:
            ids = ids + [self.eos_idx]
        return ids

    def decode(self, ids, skip_special: bool = True):
        """
        id 列表 -> token 列表。

        skip_special=True 时会跳过 <pad>/<bos>/<eos>，但保留 <unk>
        （因为 <unk> 是"模型不认识这个词"，属于有信息量的输出）。
        """
        tokens = []
        for i in ids:
            token = self.idx2token[i]
            if skip_special and token in (self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN):
                continue
            tokens.append(token)
        return tokens

    # ------------------------------------------------------------------
    # 存档 / 读档（训练完要先把词表存下来，推理时才能对齐 id）
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"idx2token": list(self.idx2token)}

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        vocab = cls()
        # 直接覆盖，保证 id 顺序和保存时完全一致
        vocab.idx2token = list(data["idx2token"])
        vocab.token2idx = {t: i for i, t in enumerate(vocab.idx2token)}
        return vocab
