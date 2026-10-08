"""
inference.py —— 推理 / 生成

用法：
    python inference.py                              # 跑内置的几个演示样例
    python inference.py --src "a b c d"              # 指定一条源句
    python inference.py --src "a b c d" --beam 4     # 用束搜索
    python inference.py --ckpt checkpoints/model.pt

流程：加载 checkpoint -> 重建词表和模型 -> 逐词生成 -> 把 id 还原成 token。
"""

import argparse

import torch

from config import Config
from data.vocab import Vocabulary
from model.transformer import Transformer
from utils.decode import beam_search, greedy_decode


def load_model(ckpt_path: str, device: torch.device):
    """
    从 checkpoint 恢复模型、词表和配置。

    checkpoint 里存了 4 样东西（见 train.py 的保存逻辑）：
        model_state_dict : 模型参数
        config           : 超参数字典（重建模型结构需要）
        src_vocab / tgt_vocab : 词表（id 必须和训练时完全一致）

    返回: (model, src_vocab, tgt_vocab, cfg)
    """
    # weights_only=False：checkpoint 里有 dict/list 等普通 Python 对象，
    # 而且这个文件是我们自己训练时写出来的，可以安全反序列化。
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = Config(**{k: v for k, v in ckpt["config"].items()
                    if k in Config.__dataclass_fields__})
    src_vocab = Vocabulary.from_dict(ckpt["src_vocab"])
    tgt_vocab = Vocabulary.from_dict(ckpt["tgt_vocab"])

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

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, src_vocab, tgt_vocab, cfg


def translate(model, src_vocab, tgt_vocab, cfg, sentence: str,
              device: torch.device, beam: int = 1):
    """
    把一条源句转成目标句（token 列表 -> token 列表）。

    参数:
        sentence : 空格分隔的 token 字符串，例如 "a b c d"
        beam     : 1 表示贪心解码，>1 表示束搜索
    """
    tokens = sentence.split()
    if not tokens:
        return []

    # 编码：加 <eos>，不加 <bos>（和 dataset.py 里的处理保持一致）
    src_ids = src_vocab.encode(tokens, add_bos=False, add_eos=True)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = model.make_src_mask(src)

    if beam > 1:
        # beam_search 返回 list[list[int]]
        out_ids = beam_search(model, src, src_mask,
                              max_len=cfg.max_decode_len,
                              bos_idx=tgt_vocab.bos_idx,
                              eos_idx=tgt_vocab.eos_idx,
                              beam_size=beam)[0]
    else:
        # greedy_decode 返回 (B, L) 张量
        out_ids = greedy_decode(model, src, src_mask,
                                max_len=cfg.max_decode_len,
                                bos_idx=tgt_vocab.bos_idx,
                                eos_idx=tgt_vocab.eos_idx)[0].tolist()

    return tgt_vocab.decode(out_ids)


def demo(ckpt_path: str = "checkpoints/model.pt", beam: int = 1):
    """跑几个内置样例并把结果打印出来。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model, src_vocab, tgt_vocab, cfg = load_model(ckpt_path, device)
    except FileNotFoundError:
        print(f"[跳过] 找不到 {ckpt_path}，请先运行 `python train.py` 训练一个模型。")
        return

    print("=" * 70)
    print(f"任务: {cfg.task} | 解码方式: {'贪心' if beam == 1 else f'束搜索(beam={beam})'}")
    print("=" * 70)

    # 几个固定的演示样例（用词表里的前几个字母，保证不会出现 <unk>）
    samples = ["a b c", "a b c d e", "e d c b a", "a a b c", "j i h g f e d c"]

    for s in samples:
        out = translate(model, src_vocab, tgt_vocab, cfg, s, device, beam=beam)

        # 顺便算一下"标准答案"（玩具任务的变换是确定的）便于对照
        from data.toy_data import transform
        gold = transform(s.split(), cfg.task)

        flag = "✓" if out == gold else "✗"
        print(f"  [{flag}] 源: {s:<24s} 预测: {' '.join(out):<24s} 标准: {' '.join(gold)}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="用训练好的 Transformer 做推理")
    p.add_argument("--ckpt", default="checkpoints/model.pt", help="checkpoint 路径")
    p.add_argument("--src", default=None, help='一条源句，例如 "a b c d"；不填则跑内置演示样例')
    p.add_argument("--beam", type=int, default=1, help="1=贪心解码，>1=束搜索宽度")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.src is None:
        demo(args.ckpt, beam=args.beam)
    else:
        model, src_vocab, tgt_vocab, cfg = load_model(args.ckpt, device)
        out = translate(model, src_vocab, tgt_vocab, cfg, args.src, device, beam=args.beam)
        print(f"源句  : {args.src}")
        print(f"生成  : {' '.join(out)}")
