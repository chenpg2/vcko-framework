"""Secure-robust-DP protocol stack for VCKO (v3).

Composes the three primitives the kw-engine synthesis identified as the resolution
of gap G11 (one end-to-end secure + robust + DP protocol on the knowledge objects):

1. ``geometric_median`` (P-0065, Pillutla-Kakade-Harchaoui) — full-vector robust
   aggregation via the smoothed Weiszfeld algorithm. Tolerates up to (just under)
   half the adversarial *weight mass*; computed as a sequence of reweighted averages
   (each runnable as one secure-aggregation round), so it composes with secure
   aggregation.

2. norm-bound verification (P-0064, RoFL) — ``is_within_norm`` / ``clip_to_norm``
   implement the validity predicate that, in production, a zero-knowledge range proof
   (Bulletproofs over ElGamal commitments) certifies WITHOUT unmasking the object.
   Here we implement the predicate/clipping logic that gates poisoned objects and
   bounds the L2 sensitivity for the DP step.

3. ``turbo_aggregate_sum`` (P-0067, So-Güler-Avestimehr) — secure sum via
   antisymmetric pairwise masks arranged in an O(group_size)-degree group structure.
   The sum is computed in a FINITE RING via fixed-point integer encoding, so the
   masks cancel EXACTLY (modular arithmetic), matching the real protocol's substrate;
   floating-point masks would only cancel up to rounding. This is a faithful model of
   the *masking-and-cancellation* mechanism, not the full dropout/collusion-resilient
   protocol (which adds Lagrange-coded recovery).

``SecureRobustDPProtocol`` orchestrates: per-centre coefficient objects -> norm-gate
(reject poisoned) -> optional clip + calibrated DP noise -> geometric-median robust
combine. The object is the O(d) coefficient vector in a common (raw-feature) space,
consistent with ``vcko.aggregator``.

Honest scope (STORY_LOCK): the cryptographic ZK/commitment layer is the production
substrate; this module implements the *protocol logic* (validity predicate, robust
combine, exact finite-ring secure-sum masking) that the crypto layer secures. For a
FORMAL (eps, delta) guarantee, calibrate the noise with
``vcko.privacy.dp.calibrate_sigma`` against the gated L2 sensitivity (= 2 * norm_bound)
— ``dp_sigma`` here is the resulting noise scale, applied AFTER clipping so sensitivity
is bounded.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "geometric_median",
    "is_within_norm",
    "clip_to_norm",
    "turbo_aggregate_sum",
    "SecureRobustDPProtocol",
]

# Fixed-point parameters for the exact finite-ring secure sum.
_FP_PRECISION = 1 << 40  # ~1.1e12: quantisation error ~5e-13 per coordinate
_FP_MODULUS = 1 << 62  # ample headroom for sums of O(1) coefficients


def geometric_median(
    points: np.ndarray,
    weights: np.ndarray | None = None,
    eps: float = 1e-8,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Weighted (smoothed) geometric median via the Weiszfeld algorithm (P-0065).

    Returns the point ``y`` (approximately) minimizing ``sum_i w_i ||x_i - y||_2``.
    Robust to adversarial mass below half the total weight (breakdown 1/2 in the
    uniform-weight case), unlike the linear mean (breakdown 0).

    Note: this is the *smoothed* Weiszfeld (``max(dist, eps)``), which is exact in the
    interior but slightly biased when the true median coincides with a data point; for
    aggregation of distinct centre coefficients this regime does not arise.

    Args:
        points: ``(K, d)`` array of K vectors.
        weights: optional ``(K,)`` non-negative weights with positive sum.
        eps: smoothing constant (avoids division by zero near a data point).
        max_iter: maximum Weiszfeld iterations.
        tol: convergence tolerance on the step size.

    Returns:
        ``(d,)`` geometric median (warns if it did not converge within max_iter).
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError(f"points must be 2-D (K, d), got {points.ndim}-D")
    k = points.shape[0]
    if k == 0:
        raise ValueError("need at least one point")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")
    if weights is None:
        weights = np.ones(k, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (k,):
            raise ValueError("weights must have shape (K,)")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("weights must be finite and non-negative")
        if weights.sum() <= 0:
            raise ValueError("weights must have positive sum")

    y = np.average(points, axis=0, weights=weights)
    converged = False
    for _ in range(max_iter):
        dist = np.maximum(np.linalg.norm(points - y, axis=1), eps)  # smoothing
        w = weights / dist
        y_new = (w[:, None] * points).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < tol:
            y = y_new
            converged = True
            break
        y = y_new
    if not converged:
        warnings.warn(
            f"geometric_median did not converge within {max_iter} iterations",
            stacklevel=2,
        )
    return np.asarray(y, dtype=float)


def _check_bound(bound: float) -> None:
    if not np.isfinite(bound) or bound < 0:
        raise ValueError("bound must be finite and non-negative")


def is_within_norm(vector: np.ndarray, bound: float) -> bool:
    """Validity predicate (P-0064): True iff ``||vector||_2 <= bound``.

    In production a zero-knowledge range proof certifies this on a commitment to the
    vector, so the aggregator can reject out-of-bound (poisoned) objects without
    unmasking them. Non-finite vectors are rejected.
    """
    _check_bound(bound)
    v = np.asarray(vector, dtype=float)
    if not np.all(np.isfinite(v)):
        return False
    return bool(np.linalg.norm(v) <= bound + 1e-12)


def clip_to_norm(vector: np.ndarray, bound: float) -> np.ndarray:
    """Project ``vector`` onto the L2 ball of radius ``bound`` (direction-preserving)."""
    _check_bound(bound)
    vector = np.asarray(vector, dtype=float)
    if not np.all(np.isfinite(vector)):
        raise ValueError("vector must be finite")
    norm = float(np.linalg.norm(vector))
    if norm <= bound or norm == 0.0:
        return vector
    return np.asarray(vector * (bound / norm), dtype=float)


def turbo_aggregate_sum(
    values: np.ndarray,
    group_size: int = 4,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """EXACT secure sum via grouped antisymmetric masks in a finite ring (P-0067).

    Each user adds masks that are antisymmetric across pairs (``r_{i,j} = -r_{j,i}``),
    so the masks cancel and the server recovers exactly ``sum_i x_i`` while never
    seeing an individual ``x_i``. To make the cancellation EXACT (not just up to
    floating-point rounding), values are encoded to fixed-point integers and the sum
    is taken modulo a large ring — the same finite-ring substrate real secure
    aggregation uses. Masking is confined to within-group pairs plus a circular link
    between consecutive groups, giving an O(K * group_size) mask count.

    Args:
        values: ``(K, d)`` per-user contribution vectors.
        group_size: users per group (the O(log N) degree in the real protocol).
        rng: random generator for the masks.

    Returns:
        ``(aggregate (d,), info)`` where ``info`` records mask-count statistics. The
        aggregate matches ``values.sum(0)`` up to the fixed-point quantisation
        (~1e-11 for O(1) inputs), with masks cancelling exactly in the ring.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"values must be 2-D (K, d), got {values.ndim}-D")
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")
    k, d = values.shape
    rng = rng or np.random.default_rng()

    # Encode to fixed-point integers (Python ints -> exact, no overflow).
    int_vals = np.round(values * _FP_PRECISION).astype(np.int64).astype(object)
    masks = np.zeros((k, d), dtype=object)

    groups = [list(range(g, min(g + group_size, k))) for g in range(0, k, group_size)]
    mask_pairs = 0
    masks_per_user = np.zeros(k, dtype=int)

    def add_pair(i: int, j: int) -> None:
        nonlocal mask_pairs
        r = rng.integers(0, _FP_MODULUS, size=d, dtype=np.int64).astype(object)
        masks[i] = masks[i] + r
        masks[j] = masks[j] - r
        masks_per_user[i] += 1
        masks_per_user[j] += 1
        mask_pairs += 1

    for grp in groups:
        for a in range(len(grp)):
            for b in range(a + 1, len(grp)):
                add_pair(grp[a], grp[b])
    if len(groups) > 1:
        for gi in range(len(groups)):
            rep_a, rep_b = groups[gi][0], groups[(gi + 1) % len(groups)][0]
            if rep_a != rep_b:
                add_pair(rep_a, rep_b)

    # Masked values mod ring; the masks cancel exactly under modular summation.
    masked = (int_vals + masks) % _FP_MODULUS
    agg_int = masked.sum(axis=0) % _FP_MODULUS
    # Map back to the signed representative and decode the fixed-point scale.
    agg_int = np.where(agg_int >= _FP_MODULUS // 2, agg_int - _FP_MODULUS, agg_int)
    aggregate = np.array([float(x) for x in agg_int]) / _FP_PRECISION

    info = {
        "n_users": k,
        "n_groups": len(groups),
        "total_mask_pairs": mask_pairs,
        "max_masks_per_user": int(masks_per_user.max()) if k > 0 else 0,
    }
    return np.asarray(aggregate, dtype=float), info


class SecureRobustDPProtocol:
    """End-to-end secure-robust-DP combine over per-centre coefficient objects.

    Pipeline (``secure_robust_combine``): norm-bound gate (P-0064) -> optional clip +
    DP noise -> robust combine (geometric median, P-0065). ``combine`` is the pure
    robust-combine step (call ``gate`` first, or use ``secure_robust_combine``).
    The secure transport (``turbo_aggregate_sum``) is available for the linear
    sub-steps of the Weiszfeld reweighted averages.
    """

    def __init__(
        self,
        norm_bound: float = 10.0,
        robust: str = "geometric_median",
        dp_sigma: float = 0.0,
    ) -> None:
        if robust not in ("geometric_median", "mean"):
            raise ValueError("robust must be 'geometric_median' or 'mean'")
        _check_bound(norm_bound)
        if norm_bound == 0:
            raise ValueError("norm_bound must be positive")
        if dp_sigma < 0 or not np.isfinite(dp_sigma):
            raise ValueError("dp_sigma must be finite and non-negative")
        self.norm_bound = norm_bound
        self.robust = robust
        self.dp_sigma = dp_sigma

    def gate(self, objects: np.ndarray) -> np.ndarray:
        """Reject objects whose L2 norm exceeds the bound (P-0064 validity gate)."""
        objects = np.asarray(objects, dtype=float)
        if objects.ndim != 2:
            raise ValueError("objects must be 2-D (K, d)")
        keep = [obj for obj in objects if is_within_norm(obj, self.norm_bound)]
        if not keep:
            raise ValueError("all objects rejected by the norm-bound gate")
        return np.asarray(keep, dtype=float)

    def _privatize(self, objects: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Clip to the norm bound (bounding L2 sensitivity), then add Gaussian noise.

        Clipping BEFORE noise is what makes the noise a sound DP mechanism: the per-
        object L2 sensitivity is then 2 * norm_bound. Calibrate dp_sigma for a target
        (eps, delta) via ``vcko.privacy.dp.calibrate_sigma(... sensitivity=2*norm_bound)``.
        """
        clipped = np.array([clip_to_norm(o, self.norm_bound) for o in objects])
        return clipped + rng.normal(0.0, self.dp_sigma, size=clipped.shape)

    def combine(
        self,
        objects: np.ndarray,
        weights: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Robust-combine accepted objects (no gating). Applies DP noise if configured."""
        objects = np.asarray(objects, dtype=float)
        if objects.ndim != 2:
            raise ValueError("objects must be 2-D (K, d)")
        if self.dp_sigma > 0:
            rng = rng or np.random.default_rng()
            objects = self._privatize(objects, rng)
        if self.robust == "geometric_median":
            return geometric_median(objects, weights=weights)
        return np.asarray(np.average(objects, axis=0, weights=weights), dtype=float)

    def secure_robust_combine(
        self,
        objects: np.ndarray,
        weights: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Full pipeline: norm-gate -> (clip + DP) -> robust combine.

        Note: gating drops rejected objects, so ``weights`` (aligned to the input) is
        not forwarded after a non-trivial rejection; pass weights only when no object
        is expected to be rejected, or weight upstream.
        """
        gated = self.gate(objects)
        w = weights if (weights is not None and len(weights) == len(gated)) else None
        return self.combine(gated, weights=w, rng=rng)
