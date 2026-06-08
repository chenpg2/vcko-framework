"""Formal differential privacy for the VCKO object (v2).

The legacy VCKO shipped a SHA-256 commitment, which provides *integrity only* and
*zero* privacy. This module adds a real ``(epsilon, delta)`` guarantee on the
released coefficients via the Gaussian mechanism with Renyi-DP accounting.

Why these choices (kw-engine synthesis):
- Output perturbation of an L2-regularised fit has bounded L2 sensitivity
  ``2/(n*lambda)`` (P-0003, Chaudhuri-Monteleoni-Sarwate).
- Perturbing the released summary is **immune to inexact optimisation** (resolves
  contradiction C7) unlike argmin-bijection objective perturbation.
- **Gaussian** noise (not Laplace) is used so privacy composes by the clean Renyi-DP
  law (resolves contradiction C5; P-0021/P-0022, Mironov 2017).

RDP -> (eps, delta) conversion (Mironov 2017, Prop. 3):
    eps(delta) = min_{alpha>1} [ eps_RDP(alpha) + log(1/delta)/(alpha-1) ].
For Gaussian mechanisms ``eps_RDP(alpha) = alpha * a`` with ``a = Delta^2/(2 sigma^2)``,
which composes additively, and the minimisation has the closed form
    eps = a_total + 2 * sqrt(a_total * log(1/delta)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "lr_l2_sensitivity",
    "RDPAccountant",
    "calibrate_sigma",
    "gaussian_output_perturbation",
    "DPResult",
]


def lr_l2_sensitivity(n: int, lam: float) -> float:
    """L2 sensitivity of the L2-regularised logistic-regression solution.

    For a 1-Lipschitz loss (logistic) with ``||x|| <= 1`` and L2 regulariser of
    strength ``lambda``, swapping one record moves the minimiser by at most
    ``2/(n*lambda)`` (Chaudhuri-Monteleoni-Sarwate, P-0003). Features must be
    scaled/clipped so the norm bound holds.
    """
    if n <= 0 or lam <= 0:
        raise ValueError("n and lam must be positive")
    return 2.0 / (n * lam)


class RDPAccountant:
    """Renyi-DP accountant for composed Gaussian mechanisms (P-0021/P-0022).

    Tracks ``a_total = sum_i Delta_i^2 / (2 sigma_i^2)`` (the per-order slope of the
    Gaussian RDP curve), which composes by addition. Converts to ``(eps, delta)``
    with the closed-form minimisation over the Renyi order.
    """

    def __init__(self) -> None:
        self._a_total: float = 0.0

    def add_gaussian(self, *, sensitivity: float, sigma: float) -> None:
        if sigma <= 0 or sensitivity < 0:
            raise ValueError("sigma must be > 0 and sensitivity >= 0")
        self._a_total += (sensitivity**2) / (2.0 * sigma**2)

    @property
    def rdp_slope(self) -> float:
        return self._a_total

    def to_dp(self, delta: float) -> float:
        """Convert accumulated RDP to an ``(eps, delta)``-DP guarantee."""
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0, 1)")
        a = self._a_total
        if a <= 0.0:
            return 0.0
        b = math.log(1.0 / delta)
        return a + 2.0 * math.sqrt(a * b)


def calibrate_sigma(*, epsilon: float, delta: float, sensitivity: float) -> float:
    """Smallest Gaussian noise sigma achieving ``(epsilon, delta)``-DP for one release.

    Inverts ``eps = a + 2 sqrt(a*ln(1/delta))`` with ``a = Delta^2/(2 sigma^2)``.
    Equivalently, with ``u = Delta/sigma`` and ``c = sqrt(2 ln(1/delta))``:
    ``eps = u^2/2 + u*c`` -> ``u = -c + sqrt(c^2 + 2 eps)`` -> ``sigma = Delta/u``.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if sensitivity < 0:
        raise ValueError("sensitivity must be non-negative")
    if sensitivity == 0:
        return 0.0
    c = math.sqrt(2.0 * math.log(1.0 / delta))
    u = -c + math.sqrt(c**2 + 2.0 * epsilon)
    return sensitivity / u


@dataclass(frozen=True)
class DPResult:
    """A privatised coefficient vector plus its formal guarantee."""

    noised_beta: np.ndarray
    sigma: float
    epsilon: float
    delta: float
    sensitivity: float
    mechanism: str = "gaussian_output_perturbation"


def gaussian_output_perturbation(
    beta: np.ndarray,
    *,
    sensitivity: float,
    epsilon: float,
    delta: float,
    rng: np.random.Generator | None = None,
) -> DPResult:
    """Add calibrated Gaussian noise to ``beta`` for ``(epsilon, delta)``-DP.

    Returns the noised vector together with the noise scale and the guarantee, so
    the VCKO can carry an honest, verifiable privacy claim instead of a derived one.
    """
    beta = np.asarray(beta, dtype=float)
    rng = rng or np.random.default_rng()
    sigma = calibrate_sigma(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
    noised = beta + rng.normal(0.0, sigma, size=beta.shape)
    return DPResult(
        noised_beta=noised,
        sigma=sigma,
        epsilon=epsilon,
        delta=delta,
        sensitivity=sensitivity,
    )
