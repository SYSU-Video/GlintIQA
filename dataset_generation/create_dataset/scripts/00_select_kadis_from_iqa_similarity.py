from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saqt.csv_io import require_columns, write_csv


SOURCE_IMAGE_COLUMN = "kadis"
KADID_REFERENCE_COLUMN = "kadid-10k"
KADID_SIMILARITY_COLUMN = "sim_value_kadid-10k"
OUTPUT_COLUMNS = ("kadis", "kadid-10k", "sim_value")


@dataclass(frozen=True)
class SelectionConfig:
    input_csv: Path
    output_csv: Path
    dataset_selection_counts: Dict[str, int]
    total_images: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SelectionConfig":
        dataset_selection_counts = {
            "tid2013": args.tid_count,
            "csiq": args.csiq_count,
            "pipal": args.pipal_count,
        }
        return cls(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            dataset_selection_counts=dataset_selection_counts,
            total_images=args.total_images,
        )

    @property
    def required_columns(self) -> Tuple[str, ...]:
        columns: List[str] = [
            SOURCE_IMAGE_COLUMN,
            KADID_REFERENCE_COLUMN,
            KADID_SIMILARITY_COLUMN,
        ]
        for dataset in self.dataset_selection_counts:
            columns.extend([dataset, f"sim_value_{dataset}"])
        return tuple(columns)


class SimilaritySelectionProcessor:
    """Select KADIS images from a multi-dataset similarity table."""

    def __init__(self, config: SelectionConfig):
        self.config = config
        self._validate_config()
        self.data = require_columns(config.input_csv, config.required_columns)

    def _validate_config(self) -> None:
        negative_counts = {
            dataset: count
            for dataset, count in self.config.dataset_selection_counts.items()
            if count < 0
        }
        if negative_counts:
            raise ValueError(f"Selection counts must be non-negative: {negative_counts}")
        if self.config.total_images <= 0:
            raise ValueError("total_images must be greater than 0.")

    @staticmethod
    def _top_indices(similarity_scores: np.ndarray, count: int) -> np.ndarray:
        if count == 0:
            return np.array([], dtype=np.int64)
        sorted_indices = np.argsort(similarity_scores)
        return sorted_indices[-count:][::-1]

    @staticmethod
    def _append_unique(
        target: List[str],
        seen: set[str],
        images: Iterable[str],
        max_items: int,
    ) -> None:
        for image in images:
            if len(target) >= max_items:
                break
            if image not in seen:
                target.append(image)
                seen.add(image)

    def select_images(self) -> Tuple[List[str], List[str], List[float]]:
        source_images = self.data[SOURCE_IMAGE_COLUMN].to_numpy()
        selected_images: List[str] = []
        selected_set: set[str] = set()

        for dataset, count in self.config.dataset_selection_counts.items():
            similarity_scores = self.data[f"sim_value_{dataset}"].to_numpy()
            top_indices = self._top_indices(similarity_scores, count)
            self._append_unique(
                selected_images,
                selected_set,
                source_images[top_indices],
                self.config.total_images,
            )

        kadid_similarity_scores = self.data[KADID_SIMILARITY_COLUMN].to_numpy()
        fallback_indices = self._top_indices(kadid_similarity_scores, len(source_images))
        for idx in fallback_indices:
            if len(selected_images) >= self.config.total_images:
                break
            self._append_unique(
                selected_images,
                selected_set,
                [source_images[idx]],
                self.config.total_images,
            )

        return self._collect_kadid_matches(selected_images)

    def _collect_kadid_matches(self, selected_images: Sequence[str]) -> Tuple[List[str], List[str], List[float]]:
        indexed_data = self.data.drop_duplicates(subset=SOURCE_IMAGE_COLUMN, keep="first")
        indexed_data = indexed_data.set_index(SOURCE_IMAGE_COLUMN)

        kadid_references: List[str] = []
        kadid_similarities: List[float] = []
        for image in tqdm(selected_images, desc="Collecting KADID references"):
            row = indexed_data.loc[image]
            kadid_references.append(row[KADID_REFERENCE_COLUMN])
            kadid_similarities.append(float(row[KADID_SIMILARITY_COLUMN]))

        return list(selected_images), kadid_references, kadid_similarities

    def save_results(
        self,
        selected_images: Sequence[str],
        reference_images: Sequence[str],
        similarity_scores: Sequence[float],
    ) -> None:
        results = pd.DataFrame(
            {
                "kadis": selected_images,
                "kadid-10k": reference_images,
                "sim_value": similarity_scores,
            }
        )
        write_csv(results, self.config.output_csv, columns=OUTPUT_COLUMNS)
        print(f"Results saved to {self.config.output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select KADIS images from a multi-dataset similarity CSV."
    )
    parser.add_argument(
        "--input-csv",
        "--input_file",
        dest="input_csv",
        type=Path,
        default=Path("./info_save/kadis_similarity_img_in_iqadataset.csv"),
        help="Path to the input multi-dataset similarity CSV.",
    )
    parser.add_argument(
        "--output-csv",
        "--output_file",
        dest="output_csv",
        type=Path,
        default=Path("./info_save/similarity_img_in_iqadataset.csv"),
        help="Path to the selected KADIS output CSV.",
    )
    parser.add_argument(
        "--tid-count",
        "--tid_count",
        dest="tid_count",
        type=int,
        default=10000,
        help="Number of images to select from TID2013 similarity ranking.",
    )
    parser.add_argument(
        "--csiq-count",
        "--csiq_count",
        dest="csiq_count",
        type=int,
        default=10000,
        help="Number of images to select from CSIQ similarity ranking.",
    )
    parser.add_argument(
        "--pipal-count",
        "--pipal_count",
        dest="pipal_count",
        type=int,
        default=10000,
        help="Number of images to select from PIPAL similarity ranking.",
    )
    parser.add_argument(
        "--total-images",
        "--total_images",
        dest="total_images",
        type=int,
        default=50000,
        help="Total number of unique KADIS images to select.",
    )
    return parser.parse_args()


def main() -> None:
    config = SelectionConfig.from_args(parse_args())
    processor = SimilaritySelectionProcessor(config)
    selected_images, reference_images, similarity_scores = processor.select_images()
    processor.save_results(selected_images, reference_images, similarity_scores)


if __name__ == "__main__":
    main()
