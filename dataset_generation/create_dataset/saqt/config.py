from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


CSV_ENCODING = "utf-8-sig"

IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".png",
    ".bmp",
    ".jpeg",
    ".jpg",
    ".gif",
    ".tiff",
)


REFERENCE_FOLDERS: Dict[str, str] = {
    "live": "refimgs",
    "csiq": "src_imgs",
    "tid2013": "reference_images",
    "kadid-10k": "ref_imgs",
}


@dataclass(frozen=True)
class FeatureFileSet:
    feature_path: Path
    image_table_path: Path

    @classmethod
    def from_dir(cls, feature_dir: Path) -> "FeatureFileSet":
        return cls(
            feature_path=feature_dir / "reference_features.npy",
            image_table_path=feature_dir / "reference_images.csv",
        )


@dataclass(frozen=True)
class SAQTLabelConfig:
    kadid_dmos_path: Path
    matches_csv_path: Path
    output_dir: Path
    batch_size: int = 1000
    distortion_types: int = 25
    distortion_levels: int = 5
    rank: int = 1


REFERENCE_IMAGE_COLUMNS = ("image", "path")
MATCH_COLUMNS = (
    "source_image",
    "source_path",
    "target_reference",
    "target_reference_path",
    "semantic_distance",
    "cosine_similarity",
    "rank",
)
KADID_DMOS_COLUMNS = ("dist_img", "ref_img", "dmos")
SAQT_LABEL_COLUMNS = (
    "distorted_image",
    "quality_score",
    "source_image",
    "matched_reference",
    "distortion_type",
    "distortion_level",
)
