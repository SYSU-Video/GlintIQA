from __future__ import annotations

import random
from typing import Dict, List, Tuple


DATASET_INDEX_RANGES: Dict[str, range] = {
    "live": range(0, 29),
    "csiq": range(0, 30),
    "tid2013": range(0, 25),
    "kadid-10k": range(0, 81),
    "livemd": range(0, 15),

    "saqt-iqa": range(0, 50000),

    "bid": range(0, 586),
    "clive": range(0, 1162),
    "livec": range(0, 1162),
    "koniq-10k": range(0, 10073),
    "spaq": range(0, 11125),
    "fblive": range(0, 39810),
}

DATASET_ALIASES = {
    "livec": "clive",
}


def canonical_dataset_name(dataset_name: str) -> str:
    return DATASET_ALIASES.get(dataset_name, dataset_name)


def dataset_indices(dataset_name: str) -> List[int]:
    if dataset_name not in DATASET_INDEX_RANGES:
        supported = ", ".join(sorted(DATASET_INDEX_RANGES))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}")
    return list(DATASET_INDEX_RANGES[dataset_name])


def train_test_split_indices(dataset_name: str, seed: int, train_ratio: float = 0.8) -> Tuple[List[int], List[int]]:
    indices = dataset_indices(dataset_name)
    random.Random(seed).shuffle(indices)
    split_point = int(round(train_ratio * len(indices)))
    return indices[:split_point], indices[split_point:]
