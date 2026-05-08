from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from PIL import Image

from .config import IMAGE_EXTENSIONS

IMAGE_LIST_COLUMNS = ("kadis", "source_image", "image", "filename")


def _validate_image_root(root: Path) -> None:
    if not root.exists():
        raise FileNotFoundError(f"Image root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Image root is not a directory: {root}")


def _read_image_list(image_list: Path) -> List[str]:
    image_list = Path(image_list)
    if not image_list.exists():
        raise FileNotFoundError(
            f"Image list not found: {image_list}. Current working directory: {Path.cwd()}. "
            "Use the correct selected image list, for example "
            "'data/selected_kadis_imgs50k.txt' or "
            "'outputs/retrieval/selected_similarity_img_in_iqadataset.csv'."
        )
    if image_list.suffix.lower() == ".csv":
        return _read_image_list_csv(image_list)
    names = []
    for line in image_list.read_text(encoding="utf-8-sig").splitlines():
        name = line.strip()
        if name and not name.startswith("#"):
            names.append(name)
    return names


def _read_image_list_csv(image_list: Path) -> List[str]:
    df = pd.read_csv(image_list, encoding="utf-8-sig")
    image_column = next((column for column in IMAGE_LIST_COLUMNS if column in df.columns), None)
    if image_column is None:
        raise ValueError(
            f"{image_list} must contain one of these image columns: {list(IMAGE_LIST_COLUMNS)}"
        )
    return (
        df[image_column]
        .dropna()
        .astype(str)
        .map(lambda name: name.strip())
        .loc[lambda series: series != ""]
        .tolist()
    )


def list_images(root: Path, recursive: bool = False, image_list: Optional[Path] = None) -> List[Path]:
    root = Path(root)
    _validate_image_root(root)
    if image_list is not None:
        image_names = _read_image_list(image_list)
        if not image_names:
            raise ValueError(f"No image names found in: {image_list}")
        image_paths = [root / name for name in image_names]
        missing = [path for path in image_paths if not path.exists()]
        if missing:
            preview = ", ".join(str(path) for path in missing[:5])
            raise FileNotFoundError(f"{len(missing)} listed images were not found. First missing: {preview}")
        invalid = [path for path in image_paths if path.suffix.lower() not in IMAGE_EXTENSIONS]
        if invalid:
            preview = ", ".join(str(path) for path in invalid[:5])
            raise ValueError(f"{len(invalid)} listed files are not supported images. First invalid: {preview}")
        return image_paths

    iterator: Iterable[Path]
    iterator = root.rglob("*") if recursive else root.iterdir()
    image_paths = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No supported images found under: {root}")
    return image_paths


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def relative_or_name(path: Path, root: Optional[Path] = None) -> str:
    if root is None:
        return path.name
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name
