# 从零实现 Transformer（Attention Is All You Need）

一个**完全手写**的 Transformer 实现，用于学习目的：

- 不调用 `nn.Transformer`、`nn.MultiheadAttention`、`nn.TransformerEncoderLayer` 等任何封装好的模块
- 只使用 `nn.Linear`、`nn.Embedding`、`nn.Dropout`、`nn.Parameter` 这些最基础的积木
- `LayerNorm`、多头注意力、掩码、残差连接、位置编码、学习率调度、标签平滑**全部自己实现**
- 文件按功能拆分，每个文件只做一件事，注释解释"为什么"而不只是"是什么"

结构严格对应论文的 Figure 1 / Figure 2。

---

## 一、目录结构

```
transfomer/
├── README.md                  本文件
├── config.py                  所有超参数集中在这里
├── main.py                    一键入口：训练 + 推理演示
├── train.py                   训练脚本（可独立运行）
├── inference.py               推理脚本：贪心 / 束搜索（可独立运行）
│
├── model/                     ← 模型本体
│   ├── __init__.py
│   ├── masks.py               Padding Mask / Causal Mask 的生成
│   ├── attention.py           缩放点积注意力 + 多头注意力（论文 3.2 节）
│   ├── layers.py              LayerNorm / 前馈网络 FFN / 残差连接 / clones
│   ├── embedding.py           词嵌入 + 正弦位置编码（论文 3.4、3.5 节）
│   ├── encoder.py             编码器层 + 编码器（论文 3.1 节左侧）
│   ├── decoder.py             解码器层 + 解码器（论文 3.1 节右侧，含交叉注意力）
│   └── transformer.py         完整模型：组装 + 掩码接口 + 权重共享 + 初始化
│
├── data/                      ← 数据
│   ├── __init__.py
│   ├── vocab.py               词表：token ↔ id，含 4 个特殊符号
│   ├── toy_data.py            玩具平行语料生成（copy / reverse / sort）
│   └── dataset.py             Dataset + collate_fn（变长句子 padding 成 batch）
│
└── utils/                     ← 训练与推理工具
    ├── __init__.py
    ├── loss.py                标签平滑损失（论文 5.4 节）
    ├── scheduler.py           Noam 学习率调度器（论文 5.3 节，式 (3)）
    └── decode.py              贪心解码 + 束搜索
```

---

## 二、环境与运行

环境：Python 3.14.6，PyTorch 2.12.1+cu130。除 PyTorch 外**没有任何额外依赖**（不需要 tqdm、numpy 等）。

```bash
# 一键跑通：训练 + 推理演示
python main.py

# 或者分开跑
python train.py                              # 默认 copy 任务，30 个 epoch
python train.py --task reverse --epochs 60   # 换任务
python train.py --task copy --d-model 128 --n-heads 8 --epochs 50
python train.py --norm-first                 # 用 Pre-LN 代替论文原版的 Post-LN

python inference.py                          # 跑内置演示样例
python inference.py --src "a b c d"          # 指定源句
python inference.py --src "a b c d" --beam 4 # 用束搜索
```

训练产物保存在 `checkpoints/model.pt`（含模型参数、超参数、两个词表）。

---

## 三、数据流：一次前向传播发生了什么

记号：`B` = batch size，`Ls`/`Lt` = 源/目标序列长度，`D` = d_model，`V` = 词表大小。

```
源句 ids            (B, Ls)
   │  TokenEmbedding ×√D  →  + PositionalEncoding  →  Dropout
   ▼
源句表示            (B, Ls, D)
   │
   │  ┌─────────────── EncoderLayer × N ───────────────┐
   │  │  ① Multi-Head Self-Attention   (mask: src pad)  │
   │  │     Add & Norm                                  │
   │  │  ② Position-wise Feed-Forward                   │
   │  │     Add & Norm                                  │
   │  └─────────────────────────────────────────────────┘
   ▼
memory              (B, Ls, D)   ← 编码器输出，只算一次
   │
   │                                    目标句 ids      (B, Lt)
   │                                       │ 同样的嵌入 + 位置编码
   │                                       ▼
   │                                  (B, Lt, D)
   │                                       │
   │      ┌───────────── DecoderLayer × N ──────────────┐
   └─────>│  ① Masked Multi-Head Self-Attention          │  mask: tgt pad + 因果
          │     (Q=K=V=解码器状态)                        │
          │     Add & Norm                               │
          │  ② Multi-Head Cross-Attention                │  mask: src pad
          │     (Q=解码器状态, K=V=memory)  ←─── 交叉注意力
          │     Add & Norm                               │
          │  ③ Position-wise Feed-Forward                │
          │     Add & Norm                               │
          └──────────────────────────────────────────────┘
                          ▼
                    (B, Lt, D)
                          │  Linear(D → V)（与目标词嵌入共享权重）
                          ▼
                    logits (B, Lt, V)  →  标签平滑交叉熵损失
```

---

## 四、几个关键实现点

### 1. 两种掩码（`model/masks.py`）

| 掩码 | 形状 | 作用 | 用在哪 |
|------|------|------|--------|
| Padding Mask | `(B, 1, 1, Lk)` | 屏蔽 `<pad>`，不让它参与注意力 | 编码器自注意力、交叉注意力、解码器自注意力 |
| Causal Mask | `(1, 1, Lq, Lk)` | 严格上三角，屏蔽未来位置，防止标签泄漏 | **只用在解码器自注意力** |

统一约定：**`True` 表示该位置需要被屏蔽**（和 `nn.MultiheadAttention` 的 `attn_mask` 一致）。
实现方式是在 softmax 之前 `scores.masked_fill(mask, -1e9)`。

> 为什么用 `-1e9` 而不是 `float('-inf')`？
> 如果某个 query 位置的所有 key 都被屏蔽，softmax 输入全是 `-inf`，分母为 0，结果变成
> `NaN`，并顺着残差连接污染整个 batch 的损失。用 `-1e9` 则 softmax 退化为均匀分布，
> 数值安全（这些位置本来就是 padding，输出会被丢弃）。

### 2. 多头注意力（`model/attention.py`）

```
Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V
MultiHead(Q,K,V) = Concat(head_1..head_h)·Wᴼ,  head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

实现上用 4 个 `(D, D)` 的 `nn.Linear` 一次算完所有头的投影，
再通过 `view + transpose` 把 `d_model` 拆成 `n_heads × d_k`：

```
(B, L, D) --view--> (B, L, h, d_k) --transpose--> (B, h, L, d_k)
```

这样 `h` 个头就变成了 batch 维的一部分，一次 `torch.matmul` 全部并行算完。
计算完再 `transpose + contiguous + view` 拼回 `(B, L, D)`，过输出投影 `Wᴼ`。

### 3. 位置编码（`model/embedding.py`）

```
PE(pos, 2i)   = sin(pos / 10000^(2i/D))
PE(pos, 2i+1) = cos(pos / 10000^(2i/D))
```

固定的、不学习的常量表，注册成 `register_buffer`（跟着 `.to(device)` 走，也会进 `state_dict`）。
词嵌入先乘 `√d_model` 再和它相加，用来平衡两者量级。

### 4. 残差 + LayerNorm（`model/layers.py`）

```
Post-LN（论文原版，默认）: x = LayerNorm(x + Dropout(Sublayer(x)))
Pre-LN （训练更稳，可选）: x = x + Dropout(Sublayer(LayerNorm(x)))
```

用 `--norm-first` 切换。Pre-LN 模式下编码器/解码器末尾会补一次最终 LayerNorm（必需），
Post-LN 模式下则不加（每个子层结尾已经归一化过）。

### 5. 学习率调度（`utils/scheduler.py`，论文式 (3)）

```
lrate = d_model^(-0.5) · min( step^(-0.5), step · warmup_steps^(-1.5) )
```

前 `warmup_steps` 步**线性上升**，之后按 `step^(-0.5)` **缓慢衰减**。
优化器用论文指定的 `Adam(β₁=0.9, β₂=0.98, ε=1e-9)`。

### 6. 标签平滑（`utils/loss.py`，论文 5.4 节）

真实类别置信度设为 `1-ε`，其余 `ε/(V-2)` 平摊（排除 `<pad>` 和真实类别本身），
`<pad>` 那一列恒为 0，标签本身是 `<pad>` 的行整行置 0、完全不计损失。

### 7. 权重共享（论文 3.4 节）

```python
self.generator.weight = self.tgt_embed.tok_embed.embed.weight
```

输出投影矩阵形状 `(V, D)` 与目标词嵌入矩阵形状完全相同，直接共享同一个 `Parameter`。
省参数，也能缓解低频词的过拟合。

### 8. 右移（Shifted Right）

```
目标句：      <bos> w1 w2 w3 <eos>
解码器输入：  <bos> w1 w2 w3        ← tgt[:, :-1]
预测标签：         w1 w2 w3 <eos>   ← tgt[:, 1:]
```

一次性并行计算所有位置的损失（Teacher Forcing），因果掩码保证位置 `t` 看不到答案。

---

## 五、玩具任务说明

`data/toy_data.py` 会程序化生成"平行语料"，源句是从字母表 `a~j` 里随机抽出的字符序列，
目标句是源句经过某种确定变换的结果：

| 任务 | 变换 | 难度 |
|------|------|------|
| `copy` | 原样复制（默认） | 容易，最快收敛到 100% |
| `reverse` | 逆序输出 | 中等，能直观看出交叉注意力的"对齐"行为 |
| `sort` | 按字母升序 | 中等 |

这三个任务都不需要下载任何数据，却能完整地驱动「编码器 → 交叉注意力 → 解码器」这条路。
训练日志里的**完全匹配准确率**能涨到接近 100%，就说明整个实现是正确的。

想换成真实数据（比如中英翻译），只需要：
1. 准备 `源句\t目标句` 格式的文本文件；
2. 用 `Vocabulary().build(...)` 分别构建源/目标词表；
3. 把 `build_dataloaders` 里的 `build_toy_corpus(...)` 换成读文件的逻辑（返回 `[(src_tokens, tgt_tokens), ...]` 即可）。

`TranslationDataset`、`make_collate_fn`、整个模型都不需要改动。

---

## 六、超参数对照（论文 base 模型 → 本项目默认值）

| 超参数 | 论文 | 本项目默认 | 说明 |
|--------|------|-----------|------|
| `d_model` | 512 | 64 | 玩具任务不需要那么大 |
| `n_heads` | 8 | 4 | 必须整除 `d_model` |
| `n_encoder_layers` / `n_decoder_layers` | 6 / 6 | 2 / 2 | |
| `d_ff` | 2048 | 256 | 通常取 `4 × d_model` |
| `dropout` | 0.1 | 0.1 | 一致 |
| `warmup_steps` | 4000 | 100 | 玩具任务总步数只有几百 |
| 优化器 | Adam(0.9, 0.98, 1e-9) | 同 | 一致 |
| 标签平滑 | 0.1 | 0.1 | 一致 |

---

## 七、建议的阅读顺序

1. `config.py` —— 先看有哪些旋钮
2. `model/masks.py` —— 掩码是理解整个模型的钥匙，先搞懂 `True` 代表什么
3. `model/attention.py` —— 核心中的核心，公式和代码一一对应
4. `model/layers.py` —— 归一化、前馈、残差这三块积木
5. `model/embedding.py` —— 位置编码
6. `model/encoder.py` / `model/decoder.py` —— 把积木拼成层
7. `model/transformer.py` —— 组装成完整模型
8. `utils/scheduler.py` + `utils/loss.py` —— 论文的两个训练技巧
9. `train.py` —— 训练循环
10. `utils/decode.py` + `inference.py` —— 推理时是怎么逐词生成的

调试小技巧：在 `utils/decode.py` 里，`model.encode()` 之后的 `memory` 全程复用，
而 `model.decode()` 每生成一个词就要重跑一次——这正是 Transformer 在推理时的计算特征。
