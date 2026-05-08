from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saqt.config import SAQTLabelConfig
from saqt.label_transfer import generate_saqt_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SAQT-IQA label CSV files from semantic matches.")
    parser.add_argument("--kadid-dmos", type=Path, required=True, help="Path to KADID-10K dmos.csv.")
    parser.add_argument("--matches-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/labels/saqt-iqa/top1"))
    parser.add_argument("--batch-size", type=int, default=1000, help="Number of source images per label batch.")
    parser.add_argument("--distortion-types", type=int, default=25)
    parser.add_argument("--distortion-levels", type=int, default=5)
    parser.add_argument("--rank", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SAQTLabelConfig(
        kadid_dmos_path=args.kadid_dmos,
        matches_csv_path=args.matches_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        distortion_types=args.distortion_types,
        distortion_levels=args.distortion_levels,
        rank=args.rank,
    )
    generate_saqt_labels(config)
    print(f"Saved SAQT labels to: {args.output_dir}")


if __name__ == "__main__":
    main()

