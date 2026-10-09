"""
data/toy_data.py —— 玩具平行语料的生成

为了让项目开箱即用（不依赖任何下载），这里用程序随机生成"平行语料"。

源句是从字母表里随机抽出的字符序列，目标句是源句经过某种变换的结果：

    copy    : 目标 = 源句原样复制          （最简单的 seq2seq 任务）
    reverse : 目标 = 源句逆序              （需要交叉注意力真正地"对齐"两端）
    sort    : 目标 = 源句按字母升序排列     （位置信息要重新组合）

这三个任务都不需要外部数据，却能完整地驱动"编码器 -> 交叉注意力 -> 解码器"
这条路走通：模型必须学会读源句、找到正确的源位置、再按目标顺序吐出来。
想验证代码是否正确，看训练日志里的"完全匹配准确率"能不能涨到 100% 即可。
"""

import random

# 字母表（相当于词表里的"普通 token"）
LETTERS = list("abcdefghij")


def transform(tokens, task: str):
    """对源句 token 列表做变换，得到目标句 token 列表。"""
    if task == "copy":
        return list(tokens)
    if task == "reverse":
        return list(reversed(tokens))
    if task == "sort":
        return sorted(tokens)
    raise ValueError(f"未知任务类型: {task}（可选: copy / reverse / sort）")


def build_toy_corpus(size: int, min_len: int = 3, max_len: int = 8,
                     task: str = "copy", seed: int = 0, letters=None):
    """
    生成 size 条 (源句 token 列表, 目标句 token 列表) 数据对。

    参数:
        size    : 生成多少条
        min_len / max_len : 源句长度范围（均匀采样）
        task    : 变换类型，见 transform
        seed    : 随机种子，保证每次运行数据一致
        letters : 字母表，默认 LETTERS

    返回:
        list[tuple[list[str], list[str]]]

    去重：因为字母表小、句子短，随机采样会产生重复样本。
    这里用 dict（有序、天然去重）过滤掉完全相同的 (源, 目标) 对，
    避免同一句话在训练集里出现几十次导致过拟合。
    """
    if letters is None:
        letters = LETTERS

    rng = random.Random(seed)   # 独立的 Random 实例，不污染全局随机状态
    unique = {}                 # 用 dict 当作有序集合来去重

    # 循环直到凑够需要的条数。加一个次数上限防止参数不合理时死循环。
    attempts = 0
    max_attempts = size * 100
    while len(unique) < size and attempts < max_attempts:
        attempts += 1
        n = rng.randint(min_len, max_len)
        src = [rng.choice(letters) for _ in range(n)]
        tgt = transform(src, task)
        unique[tuple(src)] = (src, tgt)

    if len(unique) < size:
        # 组合数不够时（比如 min_len=max_len=1、字母表只有 2 个）如实报出来
        print(f"[警告] 只生成了 {len(unique)} 条唯一数据（请求 {size} 条），"
              f"请增大 max_len 或字母表规模。")

    return list(unique.values())
