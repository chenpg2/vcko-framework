"""Principled coefficient-space aggregation for VCKO (v2).

Replaces the heuristic ``w_i proportional to n_i * AUC_i`` ensemble with the
statistically grounded one-shot distributed-MLE approximation: per-coordinate
inverse-variance (Fisher-weighted) pooling of the local coefficient estimates,
with an optional DerSimonian-Laird random-effects extension and a federated
Cochran-Q heterogeneity test.

Each centre ships only its coefficient vector ``beta_i`` and the standard errors
``se_i = sqrt(diag(H_i^{-1}))`` (H_i = observed Fisher information at beta_i), so
the knowledge object stays O(d).

Provenance (kw-engine principles):
- P-0002  one-shot averaging of local M-estimators is rate-optimal in a budget.
- P-0005  optimal pooling weights each estimate by inverse total variance.
- P-0010  plug-in tau^2 is anti-conservative with few sources (small-K caveat).
- P-0015  splice local curvature to recover joint-data inference in one shot.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "fisher_weighted_pool",
    "cochran_q",
    "dersimonian_laird_tau2",
    "random_effects_pool",
    "trimmed_mean_pool",
    "coordinatewise_median",
    "AGGREGATION_METHODS",
]

# sqrt(pi/2): asymptotic SE inflation of the median relative to the mean.
_MEDIAN_SE_FACTOR = 1.2533

_EPS = 1e-12


def _validate(betas: np.ndarray, ses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to 2-D ``(K, d)`` arrays and check shapes / positivity of SEs."""
    betas = np.atleast_2d(np.asarray(betas, dtype=float))
    ses = np.atleast_2d(np.asarray(ses, dtype=float))
    if betas.shape != ses.shape:
        raise ValueError(f"betas {betas.shape} and ses {ses.shape} must match")
    if betas.shape[0] < 1:
        raise ValueError("need at least one centre")
    if not np.all(ses > 0):
        raise ValueError("standard errors must be strictly positive")
    if not (np.all(np.isfinite(betas)) and np.all(np.isfinite(ses))):
        raise ValueError("betas and ses must be finite")
    return betas, ses


def fisher_weighted_pool(
    betas: np.ndarray, ses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-effect inverse-variance pooling, per coordinate.

    ``beta_j = sum_i w_ij beta_ij / sum_i w_ij`` with ``w_ij = 1 / se_ij^2``;
    pooled SE ``= sqrt(1 / sum_i w_ij)``. This is the one-step distributed-MLE
    approximation and tracks the pooled MLE to first order (P-0002, P-0015).

    Args:
        betas: ``(K, d)`` per-centre coefficient estimates.
        ses:   ``(K, d)`` per-centre standard errors (sqrt of inverse Fisher diag).

    Returns:
        ``(pooled_beta (d,), pooled_se (d,))``.
    """
    betas, ses = _validate(betas, ses)
    w = 1.0 / (ses**2)
    sw = w.sum(axis=0)
    pooled = (w * betas).sum(axis=0) / sw
    pooled_se = np.sqrt(1.0 / sw)
    return pooled, pooled_se


def cochran_q(
    betas: np.ndarray, ses: np.ndarray
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """Federated Cochran-Q heterogeneity test, per coordinate (G4).

    ``Q_j = sum_i w_ij (beta_ij - beta_j)^2`` with fixed-effect weights and the
    fixed-effect pooled estimate ``beta_j``. Under homogeneity ``Q_j ~ chi^2_{K-1}``.
    ``I^2 = max(0, (Q - df) / Q)`` (percentage of variance due to heterogeneity).

    Returns:
        ``(Q (d,), df, p_value (d,), I2_percent (d,))``.
    """
    betas, ses = _validate(betas, ses)
    k = betas.shape[0]
    df = k - 1
    w = 1.0 / (ses**2)
    pooled = (w * betas).sum(axis=0) / w.sum(axis=0)
    q = np.asarray((w * (betas - pooled) ** 2).sum(axis=0), dtype=float)
    if df <= 0:
        # A single centre: no heterogeneity defined.
        zeros = np.zeros_like(q)
        return q, df, np.ones_like(q), zeros
    pval = np.asarray(stats.chi2.sf(q, df), dtype=float)
    i2 = np.asarray(np.maximum(0.0, (q - df) / np.maximum(q, _EPS)) * 100.0, dtype=float)
    return q, df, pval, i2


def dersimonian_laird_tau2(betas: np.ndarray, ses: np.ndarray) -> np.ndarray:
    """DerSimonian-Laird method-of-moments between-centre variance, per coordinate.

    ``tau^2 = max(0, (Q - (K-1)) / C)`` with ``C = sum w - sum w^2 / sum w`` and
    fixed-effect weights ``w = 1/se^2`` (P-0005). Returns ``(d,)``; zeros when a
    single centre is supplied.
    """
    betas, ses = _validate(betas, ses)
    k = betas.shape[0]
    if k < 2:
        return np.zeros(betas.shape[1])
    w = 1.0 / (ses**2)
    sw = w.sum(axis=0)
    pooled = (w * betas).sum(axis=0) / sw
    q = (w * (betas - pooled) ** 2).sum(axis=0)
    c = sw - (w**2).sum(axis=0) / sw
    tau2 = np.asarray(np.maximum(0.0, (q - (k - 1)) / np.maximum(c, _EPS)), dtype=float)
    return tau2


def random_effects_pool(
    betas: np.ndarray, ses: np.ndarray, tau2: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Random-effects (DerSimonian-Laird) pooling, per coordinate.

    Weights ``w*_ij = 1 / (se_ij^2 + tau^2_j)``; reduces to the fixed-effect pool
    as ``tau^2 -> 0`` (P-0005, P-0010). If ``tau2`` is None it is estimated.
    """
    betas, ses = _validate(betas, ses)
    if tau2 is None:
        tau2 = dersimonian_laird_tau2(betas, ses)
    tau2 = np.asarray(tau2, dtype=float)
    w = 1.0 / (ses**2 + tau2)
    sw = w.sum(axis=0)
    pooled = (w * betas).sum(axis=0) / sw
    pooled_se = np.sqrt(1.0 / sw)
    return pooled, pooled_se


def trimmed_mean_pool(
    betas: np.ndarray, ses: np.ndarray, trim: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Byzantine-robust coordinate-wise trimmed mean (P-0027, Yin et al.).

    For each coordinate, drop the ``floor(trim*K)`` largest and smallest estimates,
    then average the rest. Tolerates up to a ``trim`` fraction of arbitrarily
    corrupted centres — unlike any linear combine, which has zero breakdown point
    (P-0029, Blanchard et al.). ``trim=0`` recovers the plain mean.

    The pooled SE is that of a mean of the kept independent estimates,
    ``sqrt(mean(se_kept^2)/n_kept)``.
    """
    betas, ses = _validate(betas, ses)
    if not (0.0 <= trim < 0.5):
        raise ValueError("trim must be in [0, 0.5)")
    k, d = betas.shape
    n_trim = int(np.floor(trim * k))
    if k - 2 * n_trim < 1:
        raise ValueError("trim too large: no estimates would remain")
    order = np.argsort(betas, axis=0)
    pooled = np.empty(d)
    pooled_se = np.empty(d)
    for j in range(d):
        keep = order[n_trim : k - n_trim, j] if n_trim > 0 else order[:, j]
        bj = betas[keep, j]
        sj = ses[keep, j]
        pooled[j] = float(bj.mean())
        pooled_se[j] = float(np.sqrt(np.mean(sj**2) / len(bj)))
    return pooled, pooled_se


def coordinatewise_median(
    betas: np.ndarray, ses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Byzantine-robust coordinate-wise median (P-0027).

    Tolerates up to ~50% corrupted centres per coordinate. The SE is the
    asymptotic median-of-means approximation ``sqrt(pi/2)*sqrt(mean(se^2)/K)``.
    """
    betas, ses = _validate(betas, ses)
    k = betas.shape[0]
    pooled = np.median(betas, axis=0)
    pooled_se = _MEDIAN_SE_FACTOR * np.sqrt(np.mean(ses**2, axis=0) / k)
    return np.asarray(pooled, dtype=float), np.asarray(pooled_se, dtype=float)


# Registry so the aggregator (and configs) can select a method by name.
AGGREGATION_METHODS = {
    "fixed_effect": fisher_weighted_pool,
    "random_effect": random_effects_pool,
    "robust": trimmed_mean_pool,
    "median": coordinatewise_median,
}
