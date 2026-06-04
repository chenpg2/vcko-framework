"""Prediction-sharing medium for privacy-preserving multi-centre AI collaboration (v3).

Instead of exchanging model parameters, centres share their **soft predictions**
(class-score logits at temperature T) on a shared reference dataset. This is an
architecture-agnostic knowledge object that lets heterogeneous models teach each
other and boosts every party's training toward the pooled-data ceiling.

The module provides three composable primitives:

1. ``soften`` — temperature-scaled softmax producing soft targets (P-0052, Hinton).
2. ``robust_aggregate_predictions`` — coordinate-wise robust aggregation of
   per-source prediction vectors in the LOW-DIMENSIONAL output space (P-0059,
   Cronus). Because d_out << d_param, the Ω(√d) batch requirement of the
   DP×robust antagonism (P-0049) is trivially met.
3. ``ensemble_distill`` — server-side ensemble distillation: train a student to
   match the averaged teacher logits on unlabeled data (P-0061, FedDF). Turns
   diversity from a liability into a regularizer.

The honest caveat (P-0055, Stanton): good generalization can occur with LOW
fidelity to the teacher. The medium's utility guarantee must be stated in
**task-accuracy** terms, not teacher-fidelity terms.

Provenance (kw-engine principles): P-0052, P-0053, P-0055, P-0059, P-0061.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "soften",
    "robust_aggregate_predictions",
    "bucketed_robust_aggregate",
    "ensemble_distill",
]

def soften(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Temperature-scaled softmax producing soft targets (P-0052, Hinton).

    ``q_i = exp(z_i / T) / sum_j exp(z_j / T)``

    Higher temperature exposes inter-class similarity structure ("dark knowledge")
    by amplifying relative probabilities of non-top classes.

    Args:
        logits: ``(..., C)`` raw logits from a model.
        temperature: T > 0; T=1 is standard softmax; T→∞ → uniform.

    Returns:
        ``(..., C)`` probability vectors summing to 1 along the last axis.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.asarray(logits, dtype=float)
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits must be finite (no NaN/Inf)")
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=-1, keepdims=True)  # numerical stability
    exp_s = np.exp(shifted)
    return np.asarray(exp_s / exp_s.sum(axis=-1, keepdims=True), dtype=float)


def robust_aggregate_predictions(
    predictions: np.ndarray,
    method: str = "trimmed_mean",
    trim: float = 0.2,
) -> np.ndarray:
    """Robustly aggregate per-source predictions in output space (P-0059).

    Applies coordinate-wise robust aggregation (trimmed-mean or median) to the
    per-sample, per-class prediction vectors across sources. Because the
    aggregation dimension is the number of CLASSES (small), the robust rate
    (P-0027) is cheap and the DP×robust antagonism (P-0049) does not bite.

    Args:
        predictions: ``(K, N, C)`` — K sources, N samples, C classes.
        method: ``"trimmed_mean"`` or ``"median"``.
        trim: fraction to trim from each end (for trimmed_mean).

    Returns:
        ``(N, C)`` robustly aggregated predictions.
    """
    predictions = np.asarray(predictions, dtype=float)
    if not np.all(np.isfinite(predictions)):
        raise ValueError("predictions must be finite (no NaN/Inf)")
    if predictions.ndim != 3:
        raise ValueError(f"predictions must be 3-D (K, N, C), got {predictions.ndim}-D")
    k = predictions.shape[0]

    if method == "median":
        return np.asarray(np.median(predictions, axis=0), dtype=float)

    if method == "trimmed_mean":
        if not (0.0 <= trim < 0.5):
            raise ValueError("trim must be in [0, 0.5)")
        n_trim = int(np.floor(trim * k))
        sorted_preds = np.sort(predictions, axis=0)
        if n_trim > 0:
            kept = sorted_preds[n_trim : k - n_trim]
        else:
            kept = sorted_preds
        return np.asarray(kept.mean(axis=0), dtype=float)

    raise ValueError(f"unknown method '{method}'; choose 'trimmed_mean' or 'median'")


def bucketed_robust_aggregate(
    predictions: np.ndarray,
    n_buckets: int = 2,
    method: str = "trimmed_mean",
    trim: float = 0.2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Bucketed robust aggregation for heterogeneous sources (P-0053, Karimireddy).

    Standard robust aggregators (median/trimmed-mean) fail under data heterogeneity
    because heterogeneous honest contributions look like attacks. Bucketing randomly
    groups and averages contributions before robust aggregation, reducing the
    heterogeneity-induced variance while preserving Byzantine tolerance.

    Args:
        predictions: ``(K, N, C)`` — K sources, N samples, C classes.
        n_buckets: number of random groups (must divide K or the last group is smaller).
        method: robust aggregation method within each bucket.
        trim: trim fraction for trimmed_mean.
        rng: random generator for the bucket assignment.
    """
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 3:
        raise ValueError(f"predictions must be 3-D (K, N, C), got {predictions.ndim}-D")
    rng = rng or np.random.default_rng()
    k = predictions.shape[0]
    n_buckets = min(n_buckets, k)

    indices = np.arange(k)
    rng.shuffle(indices)
    splits = np.array_split(indices, n_buckets)
    bucket_means = []
    for split in splits:
        if len(split) == 0:
            continue  # skip empty buckets
        bucket_means.append(predictions[split].mean(axis=0))
    if not bucket_means:
        raise ValueError("all buckets are empty (should not happen)")
    stacked = np.stack(bucket_means, axis=0)
    return robust_aggregate_predictions(stacked, method=method, trim=trim)


def ensemble_distill(
    teacher_logits: list[np.ndarray],
    n_classes: int,
    lr: float = 0.01,
    steps: int = 100,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Server-side ensemble distillation on unlabeled data (P-0061, FedDF).

    Trains a linear student model to match the AVERAGED TEACHER LOGITS by
    minimizing KL divergence between the student's softmax output and the
    ensemble's softened output. This fuses heterogeneous teacher models in
    logit space, turning local-model diversity into a regularizer.

    This is a simplified (linear) implementation capturing the distillation
    principle. A real deployment would use a full neural student.

    Args:
        teacher_logits: list of K arrays, each ``(N, C)`` logits.
        n_classes: number of classes C.
        lr: learning rate.
        steps: number of gradient-descent steps.
        temperature: distillation temperature.
        rng: random generator for initialization.

    Returns:
        ``(N, C)`` student logits after distillation.
    """
    if not teacher_logits:
        raise ValueError("need at least one teacher's logits")
    for i, t in enumerate(teacher_logits):
        if t.ndim != 2:
            raise ValueError(f"teacher_logits[{i}] must be 2-D (N, C), got {t.ndim}-D")
        if t.shape[1] != n_classes:
            raise ValueError(
                f"teacher_logits[{i}] has {t.shape[1]} classes, expected {n_classes}"
            )
    rng = rng or np.random.default_rng()

    ensemble_logits = np.mean(teacher_logits, axis=0)  # (N, C)
    n_samples = ensemble_logits.shape[0]
    target = soften(ensemble_logits, temperature=temperature)

    # Simple linear student: W @ x + b, where x = ensemble_logits (as features).
    W = rng.normal(0, 0.01, (n_classes, n_classes))
    b = np.zeros(n_classes)

    for _step in range(steps):
        student_logits = ensemble_logits @ W.T + b  # (N, C)
        student_probs = soften(student_logits, temperature=temperature)

        # Gradient of KL(target || student) w.r.t. student logits.
        grad_logits = (student_probs - target) / (n_samples * temperature)

        # Backprop through the linear layer.
        grad_W = grad_logits.T @ ensemble_logits
        grad_b = grad_logits.sum(axis=0)

        W -= lr * grad_W
        b -= lr * grad_b

    return np.asarray(ensemble_logits @ W.T + b, dtype=float)
