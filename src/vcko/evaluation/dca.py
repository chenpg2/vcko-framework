"""Decision Curve Analysis — net benefit at each threshold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DCAResult:
    thresholds: list[float]
    net_benefit_model: list[float]
    net_benefit_all: list[float]


def decision_curve_analysis(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> DCAResult:
    """Net benefit = TP/N - FP/N * (pt / (1-pt))."""
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)

    n = len(y_true)
    nb_model: list[float] = []
    nb_all: list[float] = []

    for pt in thresholds:
        preds = (y_pred_proba >= pt).astype(int)
        tp = float(((preds == 1) & (y_true == 1)).sum())
        fp = float(((preds == 1) & (y_true == 0)).sum())
        nb = (tp / n) - (fp / n) * (pt / (1.0 - pt))
        nb_model.append(nb)
        nb_all.append(float(y_true.mean()) - (1.0 - float(y_true.mean())) * (pt / (1.0 - pt)))

    return DCAResult(
        thresholds=thresholds.tolist(),
        net_benefit_model=nb_model,
        net_benefit_all=nb_all,
    )
