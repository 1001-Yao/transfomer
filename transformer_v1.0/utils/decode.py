"""
utils/decode.py —— 解码（自回归生成）

训练时是 Teacher Forcing（一次性喂入整句真值），
推理时没有真值可用，只能一个一个词地往外吐，每生成一个词就把它接到输入后面，
重新跑一次解码器。这叫"自回归生成"（autoregressive decoding）。

    步骤 1：输入 <bos>                -> 预测第 1 个词
    步骤 2：输入 <bos> w1             -> 预测第 2 个词
    步骤 3：输入 <bos> w1 w2          -> 预测第 3 个词
    ...直到所有句子都吐出 <eos> 或达到长度上限

性能提示：编码器只需要跑**一次**（encode 的结果 memory 全程复用），
        每次循环只重跑解码器。这是 Transformer 相对 RNN 的一大优势
        （RNN 的编码器状态必须一路传下去）。

提供两种搜索策略：
    greedy_decode : 每步只保留概率最大的那个词。快，但容易一步错步步错。
    beam_search   : 每步保留 beam_size 条候选路径。慢，但通常质量更好。
"""

import torch
import torch.nn.functional as F

from model.attention import NEG_INF


@torch.no_grad()
def greedy_decode(model, src: torch.Tensor, src_mask: torch.Tensor,
                  max_len: int, bos_idx: int, eos_idx: int) -> torch.Tensor:
    """
    贪心解码（Greedy Search）。

    参数:
        model    : Transformer 实例（内部会切换成 eval 模式）
        src      : (B, Ls) 源句 id
        src_mask : (B, 1, 1, Ls)
        max_len  : 最多生成多少个 token（含 <bos>）
        bos_idx / eos_idx : 起始/结束符号 id

    返回:
        (B, L) 的 id 张量，第 0 列是 <bos>，后面是生成的词。

    @torch.no_grad() 装饰器：推理不需要梯度，关掉自动求导能省大量显存和时间。
    """
    model.eval()
    device = src.device
    batch_size = src.size(0)

    # 编码器只跑一次，得到 memory 全程复用
    memory = model.encode(src, src_mask)

    # 已生成的序列，初始只有 <bos>：(B, 1)
    generated = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
    # 标记每条句子是否已经生成了 <eos>
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        # 每一步都要重新构造因果掩码，因为序列长度在增长
        tgt_mask = model.make_tgt_mask(generated)
        # (B, L, V) —— 我们只需要最后一个位置的预测
        logits = model.decode(memory, src_mask, generated, tgt_mask)

        # 取最后一个位置的分布，选概率最大的词（贪心）
        next_token = logits[:, -1, :].argmax(dim=-1)          # (B,)

        # 把新词接到序列末尾
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        # 已经结束的句子做个标记；所有句子都结束了就提前退出
        finished = finished | (next_token == eos_idx)
        if finished.all():
            break

    return generated


@torch.no_grad()
def beam_search(model, src: torch.Tensor, src_mask: torch.Tensor,
                max_len: int, bos_idx: int, eos_idx: int,
                beam_size: int = 4, alpha: float = 0.6):
    """
    束搜索（Beam Search）。

    贪心解码每步只看"当前这一步概率最大的词"，但"当前最优"不等于"全局最优"——
    前面选错一个词，后面就再也回不来了。
    束搜索每步同时保留得分最高的 beam_size 条**完整路径**，最后再选整体得分最高的。

    打分细节：
      - 一条路径的得分 = 它上面每个词的对数概率之和（log 概率相加 = 概率相乘）。
      - 直接比总分对短句有利（词少、加的负数少），所以要除以长度做归一化，
        这就是长度惩罚（length penalty）： score / length^alpha。

    如何防止所有 beam 都变成同一个结果？
      第 0 步如果让每个 beam 都从 <bos> 出发，它们会给出完全相同的分布，
      于是 topk 会选出同一个词，beam 就退化了。
      解法：把 1 号及以后的 beam 初始得分设成 -inf，第一步只允许一个候选。

    参数:
        beam_size : 每步保留多少条路径，常用 4~10
        alpha     : 长度惩罚指数，0 = 不惩罚，0.6~1.0 是常用值

    返回:
        list[list[int]]，长度为 B，每个元素是一条生成序列的 id 列表。

    说明：这里为了代码清晰，是"逐样本"处理已完成假设的，
          适合 batch 较小的推理场景（评估/演示足够了）。
    """
    model.eval()
    device = src.device
    B = src.size(0)
    V = model.tgt_vocab_size

    # 编码器只跑一次；把 memory 和 mask 按 beam_size 复制，
    # 这样 (B*beam) 条路径可以共享同一次矩阵运算
    memory = model.encode(src, src_mask)                       # (B, Ls, D)
    memory = memory.repeat_interleave(beam_size, dim=0)        # (B*b, Ls, D)
    src_mask_b = src_mask.repeat_interleave(beam_size, dim=0)  # (B*b, 1, 1, Ls)

    # 每条路径当前已生成的内容，初始都是 <bos>
    generated = torch.full((B * beam_size, 1), bos_idx, dtype=torch.long, device=device)

    # 每条路径的累计对数概率。(B, b) 展平成 (B*b,)
    beam_scores = torch.zeros((B, beam_size), device=device)
    beam_scores[:, 1:] = NEG_INF     # 第一步只让 0 号 beam 有效，避免路径重复
    beam_scores = beam_scores.reshape(-1)

    # 每条样本已经"以 <eos> 结尾"的候选，元素为 (归一化得分, id 列表)
    finalized = [[] for _ in range(B)]

    for _ in range(max_len - 1):
        tgt_mask = model.make_tgt_mask(generated)
        logits = model.decode(memory, src_mask_b, generated, tgt_mask)[:, -1, :]  # (B*b, V)
        log_probs = F.log_softmax(logits, dim=-1)                                 # (B*b, V)

        # 累计得分：当前路径得分 + 下一步每个词的对数概率
        # 展平成 (B, b*V)，让 topk 只在**同一条样本内部**竞争
        next_scores = (beam_scores.unsqueeze(1) + log_probs).view(B, beam_size * V)

        # 取 2b 个候选：其中一部分会以 <eos> 结束，需要归档；
        # 剩下不足 b 个继续往下扩展，所以要多取一些备用
        topk_scores, topk_idx = next_scores.topk(2 * beam_size, dim=-1)  # (B, 2b)
        beam_idx = topk_idx // V       # 这个候选是从第几条 beam 扩展来的
        token_idx = topk_idx % V       # 追加的新词

        # 把历史序列按 beam_idx 重排，再拼上新词
        gen = generated.view(B, beam_size, -1)                            # (B, b, L)
        gen = gen[torch.arange(B, device=device).unsqueeze(1), beam_idx]  # (B, 2b, L)
        gen = torch.cat([gen, token_idx.unsqueeze(-1)], dim=-1)           # (B, 2b, L+1)

        # ---------- 归档已经结束（生成了 <eos>）的候选 ----------
        eos_mask = (token_idx == eos_idx)                                 # (B, 2b)
        if eos_mask.any():
            for b in range(B):
                for k in torch.nonzero(eos_mask[b]).flatten().tolist():
                    # gen 的最后一列是刚追加的 <eos>，真正的长度是 (总长 - 1 - 1)？
                    # 不：序列是 <bos> w1 ... <eos>，长度就是 gen.size(-1)，
                    # 打分时按"生成了多少个词"来归一化，这里用 L+1 即可（常数差异不影响排序）
                    length = gen.size(-1)
                    score = topk_scores[b, k].item() / (length ** alpha)
                    finalized[b].append((score, gen[b, k].tolist()))

        # ---------- 只保留尚未结束的候选继续搜索 ----------
        keep_scores = topk_scores.masked_fill(eos_mask, NEG_INF)          # 结束的置 -inf
        keep_scores, keep_pos = keep_scores.topk(beam_size, dim=-1)       # (B, b)

        generated = gen[
            torch.arange(B, device=device).unsqueeze(1), keep_pos
        ].reshape(B * beam_size, -1)                                      # (B*b, L+1)
        beam_scores = keep_scores.reshape(-1)                             # (B*b,)

        # 每条样本都攒够 beam_size 条完整假设就可以收工了
        if all(len(f) >= beam_size for f in finalized):
            break

    # ---------- 收尾：每条样本挑出最好的结果 ----------
    results = []
    for b in range(B):
        if finalized[b]:
            # 按归一化得分选最优
            best = max(finalized[b], key=lambda item: item[0])[1]
        else:
            # 一条都没生成 <eos>（比如 max_len 太小），退而求其次：
            # 直接取当前得分最高的那条路径
            start = b * beam_size
            scores = beam_scores[start:start + beam_size]
            k = int(scores.argmax().item())
            best = generated[start + k].tolist()
        results.append(best)

    return results
