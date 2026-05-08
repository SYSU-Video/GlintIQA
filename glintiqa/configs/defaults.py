from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GlintIQA training and evaluation")
    parser.add_argument("--dataset", default="live")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--label-path", dest="label_path", type=Path, default=None)
    parser.add_argument("--similarity-csv", dest="similarity_csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/glintiqa"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--model", default="glintiqa")
    parser.add_argument("--arch", default="vit_small_patch16_224")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--vit-patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--train-patch-num", type=int, default=1)
    parser.add_argument("--test-patch-num", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--t-max", type=int, default=30)
    parser.add_argument("--eta-min", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda")
    return parser
