from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error


@dataclass(frozen=True)
class IQAMetrics:
    srocc: float
    plcc: float
    krcc: float
    rmse: float


def compute_metric(preds, labels, protocol: str = "srocc") -> float:
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    if protocol == "srocc":
        return float(stats.spearmanr(preds, labels)[0])
    if protocol == "plcc":
        return float(stats.pearsonr(preds, labels)[0])
    if protocol == "krcc":
        return float(stats.kendalltau(preds, labels)[0])
    if protocol == "rmse":
        return float(np.sqrt(mean_squared_error(preds, labels)))
    raise ValueError(f"Unsupported metric protocol: {protocol}")


def compute_iqa_metrics(preds, labels) -> IQAMetrics:
    return IQAMetrics(
        srocc=compute_metric(preds, labels, "srocc"),
        plcc=compute_metric(preds, labels, "plcc"),
        krcc=compute_metric(preds, labels, "krcc"),
        rmse=compute_metric(preds, labels, "rmse"),
    )

