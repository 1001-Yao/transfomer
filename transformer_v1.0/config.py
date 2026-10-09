"""
config.py —— 全局超参数配置

本文件把整个项目用到的所有超参数集中在一个 dataclass 里，方便统一修改。
命名尽量对齐《Attention Is All You Need》(Vaswani et al., 2017) 论文中的记号。

论文里 "base" 模型的配置是：
    d_model = 512, n_heads = 8, d_ff = 2048,
    n_encoder_layers = n_decoder_layers = 6, dropout = 0.1
本项目的默认值为了在玩具任务上快速训练，把维度缩小了，但结构完全一致。
"""

from dataclasses import dataclass


@dataclass
class Config:
    # ------------------------------------------------------------------
    # 一、数据相关
    # ------------------------------------------------------------------
    task: str = "copy"          # 玩具任务类型: "copy"(复制) / "reverse"(逆序) / "sort"(排序)
    train_size: int = 1000      # 训练集句子对数
    valid_size: int = 200       # 验证集句子对数
    min_len: int = 3            # 源句最短长度
    max_len: int = 8            # 源句最长长度
    seed: int = 42              # 随机种子

    # ------------------------------------------------------------------
    # 二、词表 / 嵌入
    # ------------------------------------------------------------------
    # 特殊符号固定占用 0~3 号 id，详见 data/vocab.py
    padding_idx: int = 0        # <pad> 的 id，注意力掩码和损失函数都要用它
    max_pos_len: int = 256      # 位置编码的最大长度（要 >= 句长+2）

    # ------------------------------------------------------------------
    # 三、模型结构（对应论文 3.1 节 Table 3 的 base 模型）
    # ------------------------------------------------------------------
    d_model: int = 64           # 论文 512。所有子层的输入输出维度都是它
    n_heads: int = 4            # 论文 8。多头注意力的头数，必须能整除 d_model
    n_encoder_layers: int = 2   # 论文 6。编码器层堆叠数量 N
    n_decoder_layers: int = 2   # 论文 6。解码器层堆叠数量 N
    d_ff: int = 256             # 论文 2048。前馈网络的隐藏层维度，一般为 4*d_model
    dropout: float = 0.1        # 论文 0.1
    norm_first: bool = False    # False = 论文原始的 Post-LN；True = Pre-LN（训练更稳，见 layers.py）
    tie_weights: bool = True    # 是否让输出层权重与目标词嵌入共享（论文 3.4 节）

    # ------------------------------------------------------------------
    # 四、训练相关
    # ------------------------------------------------------------------
    batch_size: int = 64
    epochs: int = 30
    warmup_steps: int = 100     # Noam 学习率调度器的预热步数（论文 4000，玩具任务要小很多）
    lr_factor: float = 1.0      # 学习率缩放系数
    label_smoothing: float = 0.1  # 论文 5.4 节标签平滑
    clip_grad: float = 1.0      # 梯度裁剪阈值（论文正文没写，但官方代码里用了）
    log_every: int = 20         # 每多少个 batch 打印一次训练日志
    demo_every: int = 5         # 每多少个 epoch 打印一次生成样例

    def __post_init__(self):
        # 一些结构上的硬性约束，早发现早报错
        assert self.d_model % self.n_heads == 0, "d_model 必须能被 n_heads 整除"
        assert self.d_model % 2 == 0, "d_model 必须是偶数（位置编码的 sin/cos 成对使用）"

    @property
    def max_decode_len(self) -> int:
        """推理时最多生成多少个 token（源句最长长度 + <bos> + <eos>）"""
        return self.max_len + 2


