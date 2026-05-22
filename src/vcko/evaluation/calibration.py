"""Calibration analysis — ECE and calibration curve data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    ece: float
    bin_edges: list[float]
    bin_accuracies: list[float]
    bin_confidences: list[float]
    bin_counts: list[int]


def compute_ece(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
) -> CalibrationResult:
    """Expected Calibration Error.

    ECE = sum_{b=1}^{B} (n_b / N) * |acc_b - conf_b|
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    accuracies: list[float] = []
    confidences: list[float] = []
    counts: list[int] = []
    ece = 0.0
    n = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_pred_proba > lo) & (y_pred_proba <= hi)
        n_bin = int(mask.sum())
        counts.append(n_bin)
        if n_bin == 0:
            accuracies.append(0.0)
            confidences.append(0.0)
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_pred_proba[mask].mean())
        accuracies.append(acc)
        confidences.append(conf)
        ece += (n_bin / n) * abs(acc - conf)

    return CalibrationResult(
        ece=ece,
        bin_edges=bin_edges.tolist(),
        bin_accuracies=accuracies,
        bin_confidences=confidences,
        bin_counts=counts,
    )
