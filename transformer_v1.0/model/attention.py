"""
model/attention.py —— 注意力机制

包含两部分：
    1. scaled_dot_product_attention : 论文 3.2.1 节的"缩放点积注意力"（单个头）
    2. MultiHeadAttention           : 论文 3.2.2 节的"多头注意力"

核心公式（论文式 (1)）：

                    Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V

    Q (Query)  : 我在找什么
    K (Key)    : 我有什么标签
    V (Value)  : 我实际携带的内容
    除以 √d_k    : 防止点积随维度增大而变得很大，把 softmax 推入梯度极小的饱和区。

多头注意力（论文式 (2)(3)）：

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

    h 个头各自在 d_k = d_v = d_model / h 的子空间里独立做注意力，
    最后拼接起来再过一个线性层 W^O 融合。
    "多头"让模型能在不同的表示子空间里同时关注不同位置的不同信息。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 用来代替 -inf 的一个很大的负数。
# 为什么不直接用 float('-inf') ？
#   如果某个查询位置的所有键都被屏蔽，softmax 的输入全是 -inf，
#   分母为 0，结果会变成 NaN，并顺着残差连接污染整个 batch 的损失。
#   用 -1e9 则 softmax 退化成均匀分布（值为 1/L_k），虽然物理意义不大，
#   但不会产生 NaN，而且这些位置本来就是 padding，输出会被丢弃。
NEG_INF = -1e9


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    缩放点积注意力（单个头）。
    参数:
        query : (B, n_heads, L_q, d_k)
        key   : (B, n_heads, L_k, d_k)
        value : (B, n_heads, L_k, d_v)
        mask  : 可广播到 (B, n_heads, L_q, L_k) 的 bool 张量，True 表示屏蔽
        dropout: 可选的 nn.Dropout，作用在注意力权重上（论文 5.4 节）

    返回:
        output : (B, n_heads, L_q, d_v)
        attn   : (B, n_heads, L_q, L_k) 的注意力权重，可用于可视化
    """
    d_k = query.size(-1)

    # (B, h, L_q, d_k) @ (B, h, d_k, L_k) -> (B, h, L_q, L_k)
    # 分数：第B批的第h头，l_q（query）对l_k(key)的注意力分数
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 掩码：把需要屏蔽的位置的分数压到极小，softmax 后权重≈0
    if mask is not None:
        scores = scores.masked_fill(mask, NEG_INF)
        # 把 scores 里 mask 为 True 的位置，全部替换成负无穷大

    # 最后一个向量：第 i 个 query 对所有 key 的相似度
    # 在最后一维（键的位置）做 softmax，得到每条 query 对所有 key 的注意力分布
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)

    # 用注意力权重对 value 加权求和
    output = torch.matmul(attn, value)  # (B, h, L_q, d_v)
    return output, attn


class MultiHeadAttention(nn.Module):
    """
    多头注意力。

    这里用最直白的方式实现：4 个独立的 nn.Linear 做 Q/K/V/O 投影，
    然后把 d_model 拆成 n_heads 个 d_k = d_model / n_heads 的子空间。
    （工程上常用一个大 Linear 切分来提速，但那是优化，不是原理。）
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):

        # d_model:模型的主维度，每一个token被表示多长的向量
        # n_heads: 多头注意力的头数

        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除，因为每个头要分到 d_k = d_model / n_heads 的维度"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads   # 论文里 d_v 也取 d_k
        self.d_v = d_model // n_heads

        # 4 个投影矩阵（论文式 (2) 的 W_i^Q, W_i^K, W_i^V 和式 (3) 的 W^O）。
        # 把 h 个头的参数"打包"成一个 (d_model, d_model) 矩阵，靠后续的 reshape 拆开。
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # 保存最近一次的注意力权重，方便事后做可视化分析
        self.attn_weights = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        把 (B, L, d_model) 拆成 (B, n_heads, L, d_k)。

        view 把最后一维 d_model 拆成 (n_heads, d_k)，
        再 transpose 把"头"这个维度换到前面，方便后面做批量矩阵乘法。
        """
        B, L, _ = x.size()
        x = x.view(B, L, self.n_heads, self.d_k)   # (B, L, h, d_k).view:改变形状，但元素总数不能变。
        return x.transpose(1, 2)                   # (B, h, L, d_k)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # X和它的类型，返回值tensor
        """
        把 (B, n_heads, L, d_k) 还原回 (B, L, d_model)，即论文里的 Concat。

        transpose 之后内存通常不连续，必须先 contiguous() 才能 view。
        """
        B, _, L, _ = x.size()
        x = x.transpose(1, 2).contiguous()         # 让内存顺序和逻辑形状一致，恢复"行优先"的标准布局 (B, L, h, d_k)
        return x.view(B, L, self.d_model)          # (B, L, d_model)

    def forward(self, query, key, value, mask=None):
        self.model_ = """
        参数:
            query : (B, L_q, d_model)
            key   : (B, L_k, d_model)
            value : (B, L_k, d_model)
            mask  : 可广播到 (B, h, L_q, L_k) 的 bool 张量，True = 屏蔽

            三种使用场景：
              编码器自注意力 : query = key = value = 编码器输入, mask = src_padding_mask
              解码器自注意力 : query = key = value = 解码器输入, mask = 因果+填充掩码
              交叉注意力     : query = 解码器状态, key = value = 编码器输出,
                               mask = src_padding_mask（屏蔽源句的 <pad>）

        返回:
            (B, L_q, d_model)
        """
        B = query.size(0)

        # 1) 线性投影 + 拆成多头
        q = self._split_heads(self.w_q(query))   # (B, h, L_q, d_k)
        k = self._split_heads(self.w_k(key))     # (B, h, L_k, d_k)
        v = self._split_heads(self.w_v(value))   # (B, h, L_k, d_v)

        # 2) 缩放点积注意力（h 个头并行计算）
        out, attn = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)
        self.attn_weights = attn.detach()

        # 3) 拼回多头，再过输出投影 W^O
        out = self._merge_heads(out)             # (B, L_q, d_model)
        return self.w_o(out)
