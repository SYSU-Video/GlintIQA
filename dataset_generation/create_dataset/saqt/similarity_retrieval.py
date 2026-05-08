from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import FeatureFileSet, MATCH_COLUMNS, REFERENCE_IMAGE_COLUMNS
from .csv_io import require_columns, write_csv
from .feature_extraction import extract_features
from .image_io import list_images, relative_or_name

LEGACY_MATCH_COLUMNS = ("kadis", "kadid-10k", "sim_value")
LEGACY_TOPK_MATCH_COLUMNS = ("kadis", "kadid-10k", "sim_value", "rank")


def load_reference_features(
    feature_dir: Path,
    legacy_reference_decimals: Optional[int] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    files = FeatureFileSet.from_dir(Path(feature_dir))
    if not files.feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {files.feature_path}")
    if not files.image_table_path.exists():
        raise FileNotFoundError(f"Image table not found: {files.image_table_path}")

    features = np.load(files.feature_path).astype(np.float32)
    if legacy_reference_decimals is not None:
        features = np.round(features, decimals=legacy_reference_decimals).astype(np.float32)
    image_table = require_columns(files.image_table_path, REFERENCE_IMAGE_COLUMNS)
    if len(features) != len(image_table):
        raise ValueError(
            f"Feature/image count mismatch: {len(features)} features vs {len(image_table)} images"
        )
    return features, image_table


def retrieve_matches_to_csv(
    source_image_root: Path,
    reference_feature_dir: Path,
    output_csv: Path,
    model_name: str = "resnet101",
    weights: str = "DEFAULT",
    batch_size: int = 32,
    num_workers: int = 0,
    top_k: int = 1,
    recursive: bool = False,
    source_list: Optional[Path] = None,
    device: Optional[str] = None,
    legacy_reference_decimals: Optional[int] = None,
) -> None:
    source_image_root = Path(source_image_root)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    source_paths = list_images(source_image_root, recursive=recursive, image_list=source_list)
    source_features = extract_features(
        image_paths=source_paths,
        model_name=model_name,
        weights=weights,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    reference_features, reference_table = load_reference_features(
        reference_feature_dir,
        legacy_reference_decimals=legacy_reference_decimals,
    )
    if top_k > len(reference_table):
        raise ValueError(f"top_k={top_k} exceeds reference image count {len(reference_table)}.")

    source_tensor = torch.from_numpy(source_features)
    reference_tensor = torch.from_numpy(reference_features).T
    similarities = torch.mm(source_tensor, reference_tensor)
    values, indices = torch.topk(similarities, k=top_k, dim=1)

    rows = []
    for source_idx in tqdm(range(len(source_paths)), desc="Writing semantic matches"):
        source_path = source_paths[source_idx]
        for rank_idx in range(top_k):
            target_idx = int(indices[source_idx, rank_idx].item())
            cosine_similarity = float(values[source_idx, rank_idx].item())
            target_row = reference_table.iloc[target_idx]
            source_image = relative_or_name(source_path, source_image_root)
           
            rows.append(
                {
                    "source_image": source_image,
                    "source_path": str(source_path),
                    "target_reference": target_row["image"],
                    "target_reference_path": target_row.get("path", ""),
                    "semantic_distance": 1.0 - cosine_similarity,
                    "cosine_similarity": cosine_similarity,
                    "rank": rank_idx + 1,
                }
            )

    
    output_columns = MATCH_COLUMNS
    write_csv(pd.DataFrame(rows), output_csv, columns=output_columns)
