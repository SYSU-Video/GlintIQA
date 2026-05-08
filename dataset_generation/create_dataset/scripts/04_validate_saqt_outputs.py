from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saqt.config import MATCH_COLUMNS, REFERENCE_IMAGE_COLUMNS, SAQT_LABEL_COLUMNS
from saqt.csv_io import require_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SAQT-IQA generation outputs.")
    parser.add_argument("--feature-dir", type=Path, default=None)
    parser.add_argument("--matches-csv", type=Path, default=None)
    parser.add_argument("--label-dir", type=Path, default=None)
    return parser.parse_args()


def validate_feature_dir(feature_dir: Path) -> None:
    features = feature_dir / "reference_features.npy"
    images = feature_dir / "reference_images.csv"
    if not features.exists() or not images.exists():
        raise FileNotFoundError(f"Feature directory is incomplete: {feature_dir}")
    feature_array = np.load(features)
    image_table = require_columns(images, REFERENCE_IMAGE_COLUMNS)
    if len(feature_array) != len(image_table):
        raise ValueError("Feature count does not match reference_images.csv row count.")
    print(f"Feature directory OK: {feature_array.shape}")


def validate_matches_csv(matches_csv: Path) -> None:
    df = require_columns(matches_csv, MATCH_COLUMNS)
    print(f"Matches CSV OK: {len(df)} rows")


def validate_label_dir(label_dir: Path) -> None:
    manifest_path = label_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {label_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    label_files = sorted(label_dir.glob("labels_batch_*.csv"))
    if not label_files:
        raise FileNotFoundError(f"No labels_batch_*.csv files found in {label_dir}")

    row_count = 0
    for csv_file in label_files:
        df = require_columns(csv_file, SAQT_LABEL_COLUMNS)
        row_count += len(df)

    expected = manifest.get("label_rows")
    if expected is not None and row_count != expected:
        raise ValueError(f"Label row count mismatch: {row_count} rows vs manifest {expected}")
    print(f"Label directory OK: {len(label_files)} files, {row_count} rows")


def main() -> None:
    args = parse_args()
    if args.feature_dir:
        validate_feature_dir(args.feature_dir)
    if args.matches_csv:
        validate_matches_csv(args.matches_csv)
    if args.label_dir:
        validate_label_dir(args.label_dir)


if __name__ == "__main__":
    main()
