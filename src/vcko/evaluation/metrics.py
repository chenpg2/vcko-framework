"""Evaluation metrics."""

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def calculate_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_pred_proba))


def calculate_brier_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_pred_proba))
