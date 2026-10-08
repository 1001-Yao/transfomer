"""
model/layers.py —— 基础积木

包含：
    LayerNorm                : 层归一化（自己实现，不依赖 nn.LayerNorm）
    PositionwiseFeedForward  : 论文 3.3 节的逐位置前馈网络 FFN
    SublayerConnection       : "残差连接 + LayerNorm" 的包装器（论文 3.1 节 Add & Norm）
    clones                   : 把一个层深拷贝 N 份的工具函数

关于 Post-LN 与 Pre-LN
----------------------
论文原文（Figure 1）用的是：
        x = LayerNorm(x + Sublayer(x))          <- Post-LN
即"先做子层，再残差相加，最后归一化"。
但 Post-LN 在层数变多时需要精心设计的学习率预热才训得动，
后来的实践（GPT 系列等）更常用：
        x = x + Sublayer(LayerNorm(x))          <- Pre-LN
本文件两种都实现了，用 config.norm_first 切换，默认 False（忠于论文）。
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    层归一化（论文 3.1 节）。

    对**单个样本的单个位置**的所有特征维度（最后一维 d_model）做标准化：

        y = γ * (x - μ) / √(σ² + ε) + β

    μ, σ² 是沿着最后一维统计的均值和方差；γ, β 是可学习的缩放/平移参数，
    维度都是 d_model（初始化为 1 和 0，即初始时是"恒等映射"）。

    为什么用 LayerNorm 而不是 BatchNorm ？
      BatchNorm 在 batch 维上统计，依赖 batch 大小、对变长序列不友好；
      LayerNorm 与 batch 无关，每个位置独立归一化，适合序列建模。
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # keepdim=True 保持形状，方便广播
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False：这里算的是总体方差（除以 N 而不是 N-1），
        # 与 PyTorch 官方 LayerNorm 的行为一致
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class PositionwiseFeedForward(nn.Module):
    """
    逐位置前馈网络（论文 3.3 节，式 (2)）：

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    - "逐位置"(position-wise)：对每个位置**独立地、参数共享地**应用同一个 MLP，
      所以等价于两个卷积核大小为 1 的卷积。
    - 中间维度 d_ff 通常是 4 * d_model（论文 512 -> 2048）。
    - 论文用的激活函数是 ReLU；换成 GELU 就是 BERT/GPT 的做法。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)   # 升维，给中间层更大的空间做非线性特征组合
        self.w_2 = nn.Linear(d_ff, d_model)   # 降回 d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

    # y = self.w_1(x)          # 1. 升维
    # y = F.relu(y)            # 2. 激活
    # y = self.dropout(y)      # 3. dropout
    # y = self.w_2(y)          # 4. 降维
    # return y


class SublayerConnection(nn.Module):
    """
    "Add & Norm" 子层包装器（论文 3.1 节）：

        Post-LN: x = LayerNorm(x + Dropout(Sublayer(x)))
        Pre-LN : x = x + Dropout(Sublayer(LayerNorm(x)))

    用法是把子层写成一个可调用对象传进来：

        x = self.sublayer[0](x, lambda t: self.self_attn(t, t, t, mask))

    这样每个 EncoderLayer 只要写三行代码就能表达三个子层，
    不用把残差和归一化重复写三遍。

    注意：Dropout 作用在**子层的输出**上，然后再和 x 相加，
    这是论文 5.4 节 "We apply dropout to the output of each sub-layer" 的意思。
    """

    def __init__(self, d_model: int, dropout: float = 0.1, norm_first: bool = False):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        if self.norm_first:
            return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer(x)))


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    把同一个层深拷贝 n 份，构成论文里的 "N = 6" 层堆叠。

    必须用 deepcopy 而不是 [module] * n：
      列表重复只是同一份参数的多个引用，梯度会互相覆盖；
      deepcopy 得到的是参数各自独立的 N 个层。

    用 nn.ModuleList 而不是普通 list，是为了让 PyTorch 能正确注册这些子模块，
    从而被 model.parameters() 收集、被 .to(device) 搬移、被 state_dict 保存。

    注意：deepcopy 会复制这一层的**初始参数**，所以千万不要写成
        clones(some_layer, N) 之后再修改 some_layer，那不会影响拷贝。
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
