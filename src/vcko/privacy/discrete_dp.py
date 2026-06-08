"""Integer-domain differential privacy for secure-aggregation-compatible VCKO (v3).

Standard continuous Gaussian DP noise is real-valued and incompatible with the
modular integer arithmetic of secure aggregation protocols. This module provides
discrete noise mechanisms that live in the SAME FINITE GROUP the secure sum
operates on (P-0047, P-0045).

Two mechanisms are provided:

1. **Discrete Gaussian** (P-0047, Kairouz-Liu-Steinke): sample integer noise from
   the discrete Gaussian distribution on Z. Independent sums are near-closed under
   convolution, and the Renyi-DP can be bounded despite modular wraparound.
2. **Binomial mechanism** (P-0045, cpSGD, Agarwal et al.): add Binomial(n, p) noise
   centered at its mean. Integer-valued, sums to Binomial(Kn, p), approximates
   Gaussian for large n. Compatible with integer secure aggregation and
   communication-efficient encoding.

Both mechanisms produce integer-valued noise that stays on the lattice / in a
finite group, so they compose natively with masked-sum secure aggregation.

Provenance (kw-engine principles): P-0045, P-0047, P-0050.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "discrete_gaussian_mechanism",
    "discrete_gaussian_rdp",
    "binomial_mechanism",
]


def _sample_discrete_gaussian_scalar(sigma: float, rng: np.random.Generator) -> int:
    """Sample one integer from N_Z(0, sigma^2) via rejection sampling.

    Target pmf: p(k) proportional to exp(-k^2 / (2 sigma^2)).
    Proposal: sample z ~ N(0, sigma^2), round to nearest integer k = round(z).
    The proposal mass for integer k is Phi((k+0.5)/sigma) - Phi((k-0.5)/sigma).
    Accept with probability p_target(k) / (C * q(k)), which simplifies to:
        accept_prob = exp(-k^2/(2 sigma^2)) / exp(-z^2/(2 sigma^2))
                    = exp(-(k^2 - z^2) / (2 sigma^2))
    where z is the continuous draw that rounded to k. This is correct because
    the continuous Gaussian evaluated at z is an upper bound on the discrete
    Gaussian evaluated at k times the rounding-bin width, with C=1.

    Reference: Canonne-Kamath-Steinke 2020, Algorithm 1 (with sigma >= 1).
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    while True:
        z = rng.normal(0, sigma)
        k = int(np.round(z))
        # Acceptance: ratio of discrete target at k to continuous proposal at z,
        # both under the same N(0,sigma^2) kernel. Because z is in [k-0.5, k+0.5]
        # and the Gaussian is log-concave, k^2 >= z^2 implies log_accept <= 0.
        log_accept = -(k * k - z * z) / (2.0 * sigma * sigma)
        if log_accept >= 0 or rng.random() < math.exp(log_accept):
            return k


def discrete_gaussian_mechanism(
    value: np.ndarray,
    sigma: float,
    modulus: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add discrete Gaussian noise to an integer vector (P-0047).

    Each coordinate independently receives N_Z(0, sigma^2) noise. If ``modulus``
    is given, the result is reduced mod ``modulus`` to stay in the finite group
    Z_modulus^d (compatible with modular secure aggregation).

    Args:
        value: integer array to privatize.
        sigma: noise standard deviation (controls privacy-utility tradeoff).
        modulus: if set, result is taken mod ``modulus`` (in [0, modulus)).
        rng: random generator.

    Returns:
        Integer array of the same shape, privatized.
    """
    value = np.asarray(value, dtype=np.int64)
    rng = rng or np.random.default_rng()

    noise = np.array(
        [_sample_discrete_gaussian_scalar(sigma, rng) for _ in range(value.size)],
        dtype=np.int64,
    ).reshape(value.shape)
    # Use Python-int addition to avoid int64 overflow before modular reduction.
    result = np.asarray(
        [int(v) + int(n) for v, n in zip(value.ravel(), noise.ravel())],
        dtype=object,
    ).reshape(value.shape)

    if modulus is not None:
        if modulus <= 0:
            raise ValueError("modulus must be positive")
        result = np.asarray([int(x) % modulus for x in result.ravel()], dtype=np.int64).reshape(
            value.shape
        )
        return result

    return np.asarray([int(x) for x in result.ravel()], dtype=np.int64).reshape(value.shape)


def discrete_gaussian_rdp(alpha: float, sensitivity: int, sigma: float) -> float:
    """Upper bound on the Renyi divergence of order alpha for the discrete Gaussian.

    For large sigma, this converges to the continuous Gaussian RDP:
    ``eps(alpha) = alpha * Delta^2 / (2 sigma^2)``

    We use the continuous Gaussian formula as a **conservative upper bound** on the
    discrete Gaussian RDP. This is valid because the discrete Gaussian's Renyi
    divergence is bounded above by the continuous Gaussian's at all integer orders
    alpha >= 2 when sigma >= 1 (Canonne-Kamath-Steinke 2020, Proposition 2;
    Kairouz-Liu-Steinke 2021, Theorem 10 gives the exact discrete bound which is
    tighter but requires summing over lattice points). For sigma < 1 or non-integer
    alpha, this bound is still valid but loose.

    **This is an upper bound, not the exact discrete RDP.** For production use with
    tight budget accounting, implement the exact lattice-sum formula from
    Kairouz-Liu-Steinke 2021 Theorem 10.

    Args:
        alpha: Renyi order (> 1).
        sensitivity: L2 sensitivity (integer, typically 1 after clipping).
        sigma: noise standard deviation.

    Returns:
        eps_RDP(alpha).
    """
    if alpha <= 1:
        raise ValueError("alpha must be > 1")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    # Continuous Gaussian RDP: eps = alpha * Delta^2 / (2 sigma^2)
    return alpha * (sensitivity**2) / (2.0 * sigma**2)


def binomial_mechanism(
    value: np.ndarray,
    n_trials: int,
    p: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add centered Binomial noise to an integer vector (P-0045, cpSGD).

    Adds ``Binomial(n_trials, p) - n_trials * p`` to each coordinate (centered
    so the mechanism is unbiased). The noise is integer-valued when ``n_trials * p``
    is an integer (use ``p=0.5``), sums to ``Binomial(K * n_trials, p)`` across K
    sources, and approximates ``N(0, n_trials * p * (1-p))`` for large ``n_trials``.

    This is compatible with integer secure aggregation and uses only
    ``ceil(log2(n_trials + 1))`` bits per coordinate.

    Args:
        value: integer array to privatize.
        n_trials: number of Bernoulli trials (controls noise scale).
        p: Bernoulli parameter (0.5 gives maximum variance per bit).
        rng: random generator.

    Returns:
        Integer array of the same shape.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    center_exact = n_trials * p
    if abs(center_exact - round(center_exact)) > 1e-9:
        import warnings

        warnings.warn(
            f"n_trials*p={center_exact} is non-integer; centering will introduce "
            f"a bias of {abs(center_exact - round(center_exact)):.2e}",
            stacklevel=2,
        )
    value = np.asarray(value, dtype=np.int64)
    rng = rng or np.random.default_rng()
    noise_raw = rng.binomial(n_trials, p, size=value.shape)
    center = int(round(center_exact))
    return np.asarray(value + noise_raw - center, dtype=np.int64)
