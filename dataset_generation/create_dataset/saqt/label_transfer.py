from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import KADID_DMOS_COLUMNS, SAQT_LABEL_COLUMNS, SAQTLabelConfig
from .csv_io import require_columns, write_csv


def normalize_to_range(values: np.ndarray, target_min: float = 0.0, target_max: float = 9.0) -> np.ndarray:
    values = values.astype(np.float32)
    value_min = float(values.min())
    value_max = float(values.max())
    if value_max == value_min:
        return np.full_like(values, target_min)
    return (values - value_min) / (value_max - value_min) * (target_max - target_min) + target_min


def parse_distortion_from_name(image_name: str) -> Tuple[int, int]:
    stem = Path(image_name).stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse distortion type/level from '{image_name}'")
    return int(parts[-2]), int(parts[-1])


def load_kadid_scores(kadid_dmos_path: Path) -> Dict[Tuple[str, int, int], float]:
    dmos_df = require_columns(kadid_dmos_path, KADID_DMOS_COLUMNS)

    normalized_scores = normalize_to_range(dmos_df["dmos"].to_numpy())
    score_lookup: Dict[Tuple[str, int, int], float] = {}
    for row, score in zip(dmos_df.itertuples(index=False), normalized_scores):
        dist_type, dist_level = parse_distortion_from_name(row.dist_img)
        score_lookup[(row.ref_img, dist_type, dist_level)] = float(score)
    return score_lookup


def _distorted_image_name(source_image: str, source_index: int, batch_size: int, dist_type: int, dist_level: int) -> str:
    folder_idx = int(np.ceil((source_index + 1) / batch_size))
    source_stem = Path(source_image).stem
    return f"{folder_idx:03d}/{source_stem}_{dist_type:02d}_{dist_level:02d}.bmp"


def generate_saqt_labels(config: SAQTLabelConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if config.distortion_types <= 0:
        raise ValueError("distortion_types must be greater than 0.")
    if config.distortion_levels <= 0:
        raise ValueError("distortion_levels must be greater than 0.")
    if config.rank <= 0:
        raise ValueError("rank must be greater than 0.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    score_lookup = load_kadid_scores(config.kadid_dmos_path)
    matches = require_columns(config.matches_csv_path, ("source_image", "target_reference"))

    if "rank" in matches.columns:
        matches = matches[matches["rank"] == config.rank]

    matches = matches.reset_index(drop=True)
    if matches.empty:
        raise ValueError("No semantic matches available for label transfer.")

    num_batches = int(np.ceil(len(matches) / config.batch_size))
    for batch_idx in tqdm(range(num_batches), desc="Generating SAQT label batches"):
        start = batch_idx * config.batch_size
        end = min(start + config.batch_size, len(matches))
        batch_rows = []

        for local_idx, match in enumerate(matches.iloc[start:end].itertuples(index=False)):
            source_index = start + local_idx
            source_image = match.source_image
            target_reference = match.target_reference
            for dist_type in range(1, config.distortion_types + 1):
                for dist_level in range(1, config.distortion_levels + 1):
                    key = (target_reference, dist_type, dist_level)
                    if key not in score_lookup:
                        raise KeyError(f"No KADID score for {key}")
                    batch_rows.append(
                        {
                            "distorted_image": _distorted_image_name(
                                source_image,
                                source_index,
                                config.batch_size,
                                dist_type,
                                dist_level,
                            ),
                            "quality_score": score_lookup[key],
                            "source_image": source_image,
                            "matched_reference": target_reference,
                            "distortion_type": dist_type,
                            "distortion_level": dist_level,
                        }
                    )

        output_file = config.output_dir / f"labels_batch_{batch_idx + 1:04d}.csv"
        write_csv(pd.DataFrame(batch_rows), output_file, columns=SAQT_LABEL_COLUMNS)

    manifest = {
        "dataset": "SAQT-IQA",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "matches_csv": str(config.matches_csv_path),
        "kadid_dmos": str(config.kadid_dmos_path),
        "rank": config.rank,
        "source_images": len(matches),
        "distortion_types": config.distortion_types,
        "distortion_levels": config.distortion_levels,
        "label_rows": len(matches) * config.distortion_types * config.distortion_levels,
        "batch_size": config.batch_size,
    }
    (config.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
