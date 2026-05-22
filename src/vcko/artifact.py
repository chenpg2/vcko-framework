"""VCKO Artifact - Verifiable Clinical Knowledge Object data structure."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VCKOArtifact:
    """Verifiable Clinical Knowledge Object.

    A privacy-preserving knowledge object containing model coefficients
    and statistics that can be safely shared between medical centres.
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

    def save(self, path: str | Path) -> None:
        """Save VCKO to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(self)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> VCKOArtifact:
        """Load VCKO from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def verify(self) -> bool:
        """Verify commitment hash matches current data."""
        computed = self._compute_hash()
        return computed == self.commitment_hash

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of coefficients and statistics."""
        data = {
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "n_samples": self.n_samples,
            "outcome_rate": self.outcome_rate,
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()


def create_commitment_hash(
    coefficients: np.ndarray,
    intercept: float,
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    n_samples: int,
    outcome_rate: float,
) -> str:
    """Create cryptographic commitment hash for VCKO data."""
    data = {
        "coefficients": coefficients.tolist(),
        "intercept": float(intercept),
        "feature_means": feature_means.tolist(),
        "feature_stds": feature_stds.tolist(),
        "n_samples": int(n_samples),
        "outcome_rate": float(outcome_rate),
    }
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()
