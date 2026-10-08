"""
utils/scheduler.py —— 学习率调度（论文 5.3 节）

论文用的学习率公式（式 (3)）：

    lrate = d_model^(-0.5) · min( step^(-0.5), step · warmup_steps^(-1.5) )

它由两段组成：

    warmup 阶段 (step < warmup_steps)：
        lr 随 step **线性增长** —— 对应 min 里的第二项 step · warmup^(-1.5)
    衰减阶段 (step >= warmup_steps)：
        lr 随 step 的 -0.5 次方**缓慢衰减** —— 对应 min 里的第一项 step^(-0.5)

为什么要预热？
    Transformer 的 Post-LN 结构在初始化时，深层子层的梯度方向很不稳定，
    一上来就用大学习率容易发散。先用小学习率"热机"几十到几千步再正常训练，
    是它能够收敛的关键之一。

论文同时给出了优化器设置（5.3 节末）：
    Adam(β₁=0.9, β₂=0.98, ε=1e-9)
注意 β₂ 比默认的 0.999 小、ε 比默认的 1e-8 大，都是论文特意调的。
"""

import torch


class NoamScheduler:
    """
    论文式 (3) 的学习率调度器。

    刻意**没有**继承 torch.optim.lr_scheduler._LRScheduler，
    因为那个基类有一些隐式规则（比如第一次 step 会被跳过）容易让人迷惑。
    这里就把公式直接写出来，完全可控、完全透明。

    正确用法（顺序很重要，和 PyTorch 官方 scheduler 一致）：

        optimizer.step()     # 先更新参数
        scheduler.step()     # 再更新学习率
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000, factor: float = 1.0):
        """
        参数:
            optimizer    : 要控制的优化器
            d_model      : 模型隐藏维度（出现在公式里的那个 d_model）
            warmup_steps : 预热步数，论文 base 模型用 4000
            factor       : 额外的整体缩放系数，方便做实验时统一调大/调小
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = max(1, warmup_steps)   # 防止传 0 导致除零
        self.factor = factor
        self.step_num = 0
        # 立刻把初始学习率写进优化器，保证在第一次 optimizer.step() 时就生效
        self._set_lr(self._compute_lr())

    def _compute_lr(self) -> float:
        """按照论文式 (3) 计算当前步的学习率。"""
        step = max(self.step_num, 1)   # 避免 step=0 时出现 0^(-0.5) = inf
        return self.factor * (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5),
        )

    def _set_lr(self, lr: float):
        """把学习率写进优化器的所有参数组（不同参数组可以有不同的 base_lr）。"""
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self) -> float:
        """走一步，更新学习率，返回新的 lr。"""
        self.step_num += 1
        lr = self._compute_lr()
        self._set_lr(lr)
        return lr

    def get_last_lr(self):
        """兼容 PyTorch scheduler 的接口，方便打印日志。"""
        return [group["lr"] for group in self.optimizer.param_groups]


def build_noam_optimizer(model, d_model: int, warmup_steps: int = 4000,
                         factor: float = 1.0, betas=(0.9, 0.98), eps: float = 1e-9):
    """
    按照论文 5.3 节的配置创建 Adam 优化器 + Noam 调度器。

    Adam 的三个超参完全对齐论文：
        betas = (0.9, 0.98)
        eps   = 1e-9
    （学习率不用在这里设，调度器会在初始化时覆盖掉。）
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=betas, eps=eps)
    scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps, factor=factor)
    return optimizer, scheduler
