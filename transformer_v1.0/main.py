"""
main.py —— 一键跑通整条链路

    python main.py

等价于依次执行：
    python train.py       （训练并保存到 checkpoints/model.pt）
    python inference.py   （加载模型，跑几个生成样例）

想改超参数就直接改 config.py，或者用 train.py 的命令行参数。
"""

from config import Config
from inference import demo
from train import main as train_main


def main():
    # 用默认配置训练。想换任务/加大模型，改这里或者直接跑 train.py 的命令行。
    cfg = Config()

    train_main(cfg=cfg)               # 训练并保存 checkpoint
    demo("checkpoints/model.pt")      # 加载刚训练的模型做推理演示


if __name__ == "__main__":
    main()
