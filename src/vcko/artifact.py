"""VCKO Artifact - Verifiable Clinical Knowledge Object data structure.

v2 extends the object with optional, backward-compatible fields that carry the
statistical and privacy metadata required by the principled aggregator:

- ``coef_stderr``    per-coefficient standard errors sqrt(diag(H^{-1})) for
                     inverse-variance / random-effects pooling (O(d), keeps the
                     object compact).
- ``gradient_norm``  first-order optimality certificate ||grad L(beta)|| (honest
                     -computation check; see ``vcko.verification``).
- ``regularization`` the L2 strength lambda used at fit time (bounds DP sensitivity).
- ``dp``             differential-privacy guarantee actually applied to the
                     coefficients: {mechanism, epsilon, delta, sigma, sensitivity}.

All new fields default to ``None`` and are included in the commitment hash only
when present, so legacy v1 objects hash and verify exactly as before.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _hash_payload(
    *,
    coefficients: list[float],
    intercept: float,
    feature_means: list[float],
    feature_stds: list[float],
    n_samples: int,
    outcome_rate: float,
    coef_stderr: list[float] | None = None,
    gradient_norm: float | None = None,
    regularization: float | None = None,
    dp: dict[str, Any] | None = None,
) -> str:
    """Build the canonical hash payload and return its SHA-256 hex digest.

    Legacy (v1) fields are always present; v2 fields are added only when non-None,
    so v1 objects produce byte-identical hashes to the original implementation.
    """
    data: dict[str, Any] = {
        "coefficients": coefficients,
        "intercept": float(intercept),
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "n_samples": int(n_samples),
        "outcome_rate": float(outcome_rate),
    }
    if coef_stderr is not None:
        data["coef_stderr"] = coef_stderr
    if gradient_norm is not None:
        data["gradient_norm"] = float(gradient_norm)
    if regularization is not None:
        data["regularization"] = float(regularization)
    if dp is not None:
        data["dp"] = dp
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


@dataclass(frozen=True)
class VCKOArtifact:
    """Verifiable Clinical Knowledge Object.

    A privacy-preserving knowledge object containing model coefficients and
    statistics that can be safely shared between medical centres.
    """

    centre_id: str
    feature_names: list[str]
    coefficients: list[float]
    intercept: float
    feature_means: list[float]
    feature_stds: list[float]
    n_samples: int
    outcome_rate: float
    commitment_hash: str
    metadata: dict[str, Any]
    # --- v2 optional fields (backward compatible) ---
    coef_stderr: list[float] | None = None
    gradient_norm: float | None = None
    regularization: float | None = None
    dp: dict[str, Any] | None = field(default=None)

    def save(self, path: str | Path) -> None:
        """Save VCKO to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(self)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> VCKOArtifact:
        """Load VCKO from JSON file (tolerates v1 objects without v2 fields)."""
        with open(path) as f:
            data = json.load(f)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def verify(self) -> bool:
        """Verify commitment hash matches current data."""
        return self._compute_hash() == self.commitment_hash

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of coefficients, statistics, and v2 metadata."""
        return _hash_payload(
            coefficients=self.coefficients,
            intercept=self.intercept,
            feature_means=self.feature_means,
            feature_stds=self.feature_stds,
            n_samples=self.n_samples,
            outcome_rate=self.outcome_rate,
            coef_stderr=self.coef_stderr,
            gradient_norm=self.gradient_norm,
            regularization=self.regularization,
            dp=self.dp,
        )


def create_commitment_hash(
    coefficients: np.ndarray,
    intercept: float,
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    n_samples: int,
    outcome_rate: float,
    *,
    coef_stderr: np.ndarray | list[float] | None = None,
    gradient_norm: float | None = None,
    regularization: float | None = None,
    dp: dict[str, Any] | None = None,
) -> str:
    """Create cryptographic commitment hash for VCKO data.

    Legacy positional signature is preserved; v2 metadata is keyword-only and
    optional, so existing callers are unaffected.
    """
    return _hash_payload(
        coefficients=np.asarray(coefficients).tolist(),
        intercept=float(intercept),
        feature_means=np.asarray(feature_means).tolist(),
        feature_stds=np.asarray(feature_stds).tolist(),
        n_samples=int(n_samples),
        outcome_rate=float(outcome_rate),
        coef_stderr=(None if coef_stderr is None else np.asarray(coef_stderr).tolist()),
        gradient_norm=gradient_norm,
        regularization=regularization,
        dp=dp,
    )
