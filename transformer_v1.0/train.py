"""
train.py —— 训练脚本

用法：
    python train.py                          # 用 config.py 里的默认配置训练（copy 任务）
    python train.py --task reverse --epochs 40
    python train.py --d-model 128 --n-heads 8 --epochs 50

训练流程（和任何 seq2seq 模型一样）：
    1. 取一个 batch 的 (源句, 目标句)，都补到 batch 内最大长度
    2. 把目标句切成 tgt_in = tgt[:, :-1]（解码器输入）和 tgt_out = tgt[:, 1:]（标签）
    3. 构造掩码：src_mask 屏蔽源句的 <pad>；tgt_mask 屏蔽目标句的 <pad> 和未来位置
    4. 前向传播得到 logits (B, Lt, V)
    5. 用带标签平滑的损失算 loss
    6. 反向传播 -> 梯度裁剪 -> 更新参数 -> 调整学习率
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from config import Config
from data.dataset import TranslationDataset, make_collate_fn
from data.toy_data import build_toy_corpus
from data.vocab import Vocabulary
from model.transformer import Transformer
from utils.decode import greedy_decode
from utils.loss import LabelSmoothingLoss
from utils.scheduler import build_noam_optimizer


# ======================================================================
# 命令行参数
# ======================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="训练一个从零实现的 Transformer")
    p.add_argument("--task", default=None, choices=["copy", "reverse", "sort"],
                   help="玩具任务类型（默认取 config.py 里的值）")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--d-model", type=int, default=None, help="隐藏维度，必须是 n_heads 的倍数且为偶数")
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--enc-layers", type=int, default=None)
    p.add_argument("--dec-layers", type=int, default=None)
    p.add_argument("--d-ff", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--norm-first", action="store_true", help="使用 Pre-LN（默认 Post-LN，忠于论文）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="cpu / cuda，默认自动选择")
    p.add_argument("--save", type=str, default="checkpoints/model.pt")
    return p.parse_args(argv)


def build_config(args) -> Config:
    """把命令行里显式给出的参数覆盖到默认 Config 上。"""
    cfg = Config()
    # 名字一一对应的直接覆盖
    for name in ("task", "epochs", "dropout", "warmup_steps"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    # 命令行用连字符，Config 用下划线，这里做映射
    mapping = {
        "batch_size": args.batch_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_encoder_layers": args.enc_layers,
        "n_decoder_layers": args.dec_layers,
        "d_ff": args.d_ff,
    }
    for name, value in mapping.items():
        if value is not None:
            setattr(cfg, name, value)
    if args.norm_first:
        cfg.norm_first = True
    cfg.seed = args.seed
    return cfg


# ======================================================================
# 数据准备
# ======================================================================
def build_dataloaders(cfg: Config):
    """
    生成玩具语料 -> 建词表 -> 建 Dataset / DataLoader。

    注意：词表只能从**训练集**统计，验证集里出现的新词要当 <unk> 处理，
    否则就是数据泄漏。本项目里训练集和验证集用同一个字母表，所以不会有 <unk>。
    """
    # 一次性生成训练+验证所需的全部数据对
    pairs = build_toy_corpus(
        size=cfg.train_size + cfg.valid_size,
        min_len=cfg.min_len, max_len=cfg.max_len,
        task=cfg.task, seed=cfg.seed,
    )
    train_pairs = pairs[: cfg.train_size]
    valid_pairs = pairs[cfg.train_size:]

    # 源端和目标端各自建一个词表
    # （真实翻译任务里这两个词表完全不同；这里碰巧是同一套字母，但结构上保持独立）
    src_vocab = Vocabulary().build([src for src, _ in train_pairs])
    tgt_vocab = Vocabulary().build([tgt for _, tgt in train_pairs])

    train_set = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    valid_set = TranslationDataset(valid_pairs, src_vocab, tgt_vocab)

    collate = make_collate_fn(cfg.padding_idx)
    # 训练时 shuffle 打乱顺序；验证集不需要
    # 注意 collate_fn 必须传进去，否则变长句子无法拼成 batch
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=collate, drop_last=False)

    return train_loader, valid_loader, src_vocab, tgt_vocab


# ======================================================================
# 训练 / 评估
# ======================================================================
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, cfg, epoch):
    """跑一个 epoch，返回平均损失（每个非 padding token 的平均损失）。"""
    model.train()   # 打开 dropout
    total_loss, total_tokens = 0.0, 0
    start_time = time.time()

    for step, (src, tgt) in enumerate(loader, start=1):
        src, tgt = src.to(device), tgt.to(device)

        # 右移：输入是 <bos> ... w_{n-1}，标签是 w_0 ... <eos>
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        # 构造掩码
        src_mask = model.make_src_mask(src)     # (B, 1, 1, Ls)
        tgt_mask = model.make_tgt_mask(tgt_in)  # (B, 1, Lt, Lt)

        # 前向传播 -> (B, Lt, V)
        logits = model(src, tgt_in, src_mask, tgt_mask)

        # 损失：注意标签平滑只作用在**非 padding** 位置上（LabelSmoothingLoss 内部处理）
        loss = criterion(logits, tgt_out)

        # 反向传播三连
        optimizer.zero_grad()                    # 清空上一轮的梯度
        loss.backward()                          # 反传
        # 梯度裁剪：防止个别 batch 产生爆炸的梯度把参数带飞。
        # 论文正文没提，但官方实现里有这一步，实践中几乎必需。
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        optimizer.step()                         # 更新参数
        lr = scheduler.step()                    # 更新学习率（必须在 optimizer.step 之后）

        # 统计（loss 是"平均每个 token"，乘回 token 数得到总损失，方便加权平均）
        n_tokens = (tgt_out != cfg.padding_idx).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        if step % cfg.log_every == 0:
            print(f"   epoch {epoch:>3d} | step {step:>4d}/{len(loader)} "
                  f"| loss {loss.item():.4f} | lr {lr:.2e}")

    avg_loss = total_loss / max(total_tokens, 1)
    print(f"   epoch {epoch:>3d} 完成 | 平均 loss {avg_loss:.4f} | 用时 {time.time() - start_time:.1f}s")
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg):
    """在验证集上算平均损失。"""
    model.eval()    # 关掉 dropout
    total_loss, total_tokens = 0.0, 0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        logits = model(src, tgt_in, model.make_src_mask(src), model.make_tgt_mask(tgt_in))
        loss = criterion(logits, tgt_out)
        n_tokens = (tgt_out != cfg.padding_idx).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


def _trim_ids(ids, bos_idx, eos_idx, pad_idx):
    """去掉 <bos>/<eos>/<pad>，只留下真正的词，方便逐条比较。"""
    out = []
    for t in ids:
        if t == bos_idx:
            continue
        if t in (eos_idx, pad_idx):
            break          # 遇到结束符就停止
        out.append(t)
    return out


@torch.no_grad()
def evaluate_accuracy(model, loader, device, cfg, src_vocab, tgt_vocab, max_batches=4):
    """
    用贪心解码生成结果，和标准答案逐条比较，算"完全匹配准确率"。

    只看前 max_batches 个 batch（贪心解码要逐词循环，比较慢），
    对数以千计的玩具数据来说，这个采样已经足够反映趋势了。
    """
    model.eval()
    correct, total = 0, 0
    examples = []

    for i, (src, tgt) in enumerate(loader):
        if i >= max_batches:
            break
        src, tgt = src.to(device), tgt.to(device)
        pred_ids = greedy_decode(model, src, model.make_src_mask(src),
                                 max_len=cfg.max_decode_len,
                                 bos_idx=tgt_vocab.bos_idx, eos_idx=tgt_vocab.eos_idx)

        for b in range(src.size(0)):
            pred = _trim_ids(pred_ids[b].tolist(), tgt_vocab.bos_idx,
                             tgt_vocab.eos_idx, cfg.padding_idx)
            gold = _trim_ids(tgt[b].tolist(), tgt_vocab.bos_idx,
                             tgt_vocab.eos_idx, cfg.padding_idx)
            is_correct = (pred == gold)
            correct += int(is_correct)
            total += 1
            # 留几条样本用于打印展示
            if len(examples) < 3:
                examples.append((
                    " ".join(src_vocab.decode(src[b].tolist())),
                    " ".join(tgt_vocab.decode(pred)),
                    " ".join(tgt_vocab.decode(gold)),
                    is_correct,
                ))

    return correct / max(total, 1), examples


# ======================================================================
# 主流程
# ======================================================================
def main(argv=None, cfg: Config = None):
    args = parse_args(argv) if cfg is None else None
    if cfg is None:
        cfg = build_config(args)

    # ---------- 随机种子 ----------
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # ---------- 设备 ----------
    if args is not None and args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"任务: {cfg.task} | 设备: {device} | torch: {torch.__version__}")
    print(f"d_model={cfg.d_model} n_heads={cfg.n_heads} "
          f"enc={cfg.n_encoder_layers} dec={cfg.n_decoder_layers} "
          f"d_ff={cfg.d_ff} dropout={cfg.dropout} "
          f"norm={'Pre-LN' if cfg.norm_first else 'Post-LN'}")
    print("=" * 70)

    # ---------- 数据 ----------
    train_loader, valid_loader, src_vocab, tgt_vocab = build_dataloaders(cfg)
    print(f"训练样本 {len(train_loader.dataset)} 条 / 验证样本 {len(valid_loader.dataset)} 条")
    print(f"源词表 {len(src_vocab)} / 目标词表 {len(tgt_vocab)}")
    print(f"源词表内容: {src_vocab.idx2token}")

    # ---------- 模型 ----------
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_encoder_layers=cfg.n_encoder_layers,
        n_decoder_layers=cfg.n_decoder_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_pos_len=cfg.max_pos_len,
        padding_idx=cfg.padding_idx,
        norm_first=cfg.norm_first,
        tie_weights=cfg.tie_weights,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数量: {n_params:,}")

    # ---------- 优化器 / 调度器 / 损失 ----------
    optimizer, scheduler = build_noam_optimizer(
        model, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps, factor=cfg.lr_factor,
    )
    # 注意：<unk> 和 <bos> 也会出现在平滑分布里（论文只排除了 <pad>），
    # 如果想让它们也不参与，把 vocab_size 改成 len(vocab)-1 之类的技巧要谨慎，这里保持论文原样。
    criterion = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab), padding_idx=cfg.padding_idx, smoothing=cfg.label_smoothing,
    )

    # ---------- 训练循环 ----------
    best_acc = 0.0
    save_path = args.save if args is not None else "checkpoints/model.pt"

    for epoch in range(1, cfg.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, cfg, epoch)
        valid_loss = evaluate(model, valid_loader, criterion, device, cfg)

        acc, examples = evaluate_accuracy(model, valid_loader, device, cfg,
                                          src_vocab, tgt_vocab)
        print(f"   >>> 验证 loss {valid_loss:.4f} | 完全匹配准确率 {acc * 100:.1f}%")

        if epoch % cfg.demo_every == 0 or epoch == 1:
            for src_s, pred_s, gold_s, ok in examples:
                flag = "✓" if ok else "✗"
                print(f"       [{flag}] 源: {src_s:<28s} 预测: {pred_s:<28s} 标准: {gold_s}")

        # 保存最优模型
        if acc >= best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg.__dict__,          # dataclass 的字段字典
                "src_vocab": src_vocab.to_dict(),
                "tgt_vocab": tgt_vocab.to_dict(),
                "best_acc": best_acc,
            }, save_path)

    print("=" * 70)
    print(f"训练结束，最佳完全匹配准确率 {best_acc * 100:.1f}%，模型已保存到 {save_path}")
    print("=" * 70)
    return model, src_vocab, tgt_vocab, cfg


if __name__ == "__main__":
    main()
