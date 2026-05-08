from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .backbones import create_backbone
from .config import REFERENCE_IMAGE_COLUMNS
from .csv_io import write_csv
from .image_io import list_images, load_rgb_image, relative_or_name


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        return self.transform(load_rgb_image(path)), str(path)


def extract_features(
    image_paths: List[Path],
    model_name: str = "resnet101",
    weights: str = "DEFAULT",
    batch_size: int = 32,
    num_workers: int = 0,
    device: Optional[str] = None,
) -> np.ndarray:
    if not image_paths:
        raise ValueError("No images found for feature extraction.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if num_workers < 0:
        raise ValueError("num_workers must be greater than or equal to 0.")

    backbone = create_backbone(model_name=model_name, weights=weights, device=device)
    dataset = ImagePathDataset(image_paths, backbone.transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    features = np.zeros((len(image_paths), backbone.feature_dim), dtype=np.float32)
    offset = 0
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Extracting semantic features"):
            images = images.to(backbone.device, non_blocking=True)
            batch_features = backbone.model(images)
            batch_features = batch_features.flatten(1)
            batch_features = F.normalize(batch_features, p=2, dim=1)
            batch_np = batch_features.cpu().numpy().astype(np.float32)
            features[offset : offset + len(batch_np)] = batch_np
            offset += len(batch_np)

    return features


def extract_features_to_disk(
    image_root: Path,
    output_dir: Path,
    model_name: str = "resnet101",
    weights: str = "DEFAULT",
    batch_size: int = 32,
    num_workers: int = 0,
    recursive: bool = False,
    image_list: Optional[Path] = None,
    device: Optional[str] = None,
) -> None:
    image_root = Path(image_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(image_root, recursive=recursive, image_list=image_list)
    features = extract_features(
        image_paths=image_paths,
        model_name=model_name,
        weights=weights,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    np.save(output_dir / "reference_features.npy", features)
    image_table = pd.DataFrame(
        {
            "image": [relative_or_name(path, image_root) for path in image_paths],
            "path": [str(path) for path in image_paths],
        }
    )
    write_csv(image_table, output_dir / "reference_images.csv", columns=REFERENCE_IMAGE_COLUMNS)
