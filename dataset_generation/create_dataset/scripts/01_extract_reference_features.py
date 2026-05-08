from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saqt.config import REFERENCE_FOLDERS
from saqt.feature_extraction import extract_features_to_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract semantic features for IQA reference images.")
    parser.add_argument("--dataset", default="kadid-10k", choices=sorted(REFERENCE_FOLDERS))
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root directory of the IQA dataset.")
    parser.add_argument("--image-root", type=Path, default=None, help="Direct image directory. Overrides dataset-root/reference folder.")
    parser.add_argument("--output-root", type=Path, default=Path("./outputs"))
    parser.add_argument("--model-name", default="resnet101")
    parser.add_argument("--weights", default="DEFAULT")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--image-list", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_root = args.image_root or args.dataset_root / REFERENCE_FOLDERS[args.dataset]
    output_dir = args.output_root / "features" / args.dataset / args.model_name
    extract_features_to_disk(
        image_root=image_root,
        output_dir=output_dir,
        model_name=args.model_name,
        weights=args.weights,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        recursive=args.recursive,
        image_list=args.image_list,
        device=args.device,
    )
    print(f"Saved features to: {output_dir}")


if __name__ == "__main__":
    main()

