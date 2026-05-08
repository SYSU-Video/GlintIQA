from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from .config import CSV_ENCODING


def read_csv(csv_path: Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path, encoding=CSV_ENCODING)


def require_columns(csv_path: Path, required_columns: Iterable[str]) -> pd.DataFrame:
    df = read_csv(csv_path)
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    return df


def write_csv(df: pd.DataFrame, output_path: Path, columns: Optional[Sequence[str]] = None) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        df = df.loc[:, list(columns)]
    df.to_csv(output_path, index=False, encoding=CSV_ENCODING)
