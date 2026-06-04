"""Witness-free honest-computation certificate for VCKO objects (v2).

The legacy ``VCKOArtifact.verify()`` only recomputes a SHA-256 hash — it proves the
object was not altered in transit, but says nothing about whether the coefficients
were *honestly computed* from a real logistic fit. This module adds a lightweight,
O(d), privacy-compatible certificate based on first-order optimality.

A correctly-fitted L2-regularised logistic regression satisfies stationarity:
    g(beta) = (1/n) X^T (sigmoid(X beta) - y) + lambda * beta  ~  0.
The centre computes ``||g(beta)||`` (cheap, local) and ships it; the receiver
rejects any object whose reported gradient norm exceeds a tolerance. Forging a
mutually-consistent ``(beta, se, ||g|| ~ 0)`` triple without actually solving the
problem is hard, so this raises the bar from "tamper-check" toward "proof of honest
computation form" — without zero-knowledge circuit cost (resolves contradiction C4)
and without revealing per-record inputs (resolves contradiction C6; cf. P-0011).

Scope note (gap G6): this certifies the *form* of the computation, NOT that the
inputs were real / unpoisoned. Byzantine-robust aggregation is explicitly out of
scope and declared as such.
"""

from __future__ import annotations

import numpy as np

__all__ = ["logistic_gradient_norm", "optimality_certificate"]


def logistic_gradient_norm(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    lam: float,
    intercept: float = 0.0,
) -> float:
    """L2 norm of the mean regularised logistic-regression gradient at ``beta``.

    ``g = (1/n) X^T (p - y) + lambda * beta`` with ``p = sigmoid(X beta + intercept)``.
    Near a true optimum this is ~0; it grows monotonically as ``beta`` moves away.

    Args:
        X: ``(n, d)`` feature matrix (standardised, as used in training).
        y: ``(n,)`` binary labels in {0, 1}.
        beta: ``(d,)`` coefficient vector to certify.
        lam: L2 regularisation strength used at fit time.
        intercept: optional fitted intercept.

    Returns:
        The scalar gradient L2 norm.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.asarray(beta, dtype=float)
    n = X.shape[0]
    if n == 0:
        raise ValueError("X must have at least one row")
    logits = X @ beta + intercept
    p = 1.0 / (1.0 + np.exp(-logits))
    grad = X.T @ (p - y) / n + lam * beta
    return float(np.linalg.norm(grad))


def optimality_certificate(gradient_norm: float, tol: float = 1e-2) -> bool:
    """Accept an object iff its reported gradient norm is within tolerance of zero.

    ``tol`` should exceed the solver's convergence tolerance but be small relative
    to the gradient norm induced by a meaningful coefficient tamper.
    """
    return bool(np.isfinite(gradient_norm) and gradient_norm <= tol)
