"""VCKO Aggregator - weighted ensemble prediction from multiple VCKOs.

Aggregation formula:
  w_i = n_i * AUC_i / sum(n_j * AUC_j)
  z_i = (x - mu_i) / (sigma_i + eps)
  P(y=1|x) = sum( w_i * sigmoid(beta_i^T z_i + beta_0i) )
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .artifact import VCKOArtifact

_EPS = 1e-8


class VCKOAggregator:
    def __init__(self):
        self._vckos: list[VCKOArtifact] = []

    @property
    def vckos(self) -> list[VCKOArtifact]:
        return list(self._vckos)

    def add(self, vcko: VCKOArtifact) -> None:
        if not vcko.verify():
            raise ValueError(f"VCKO verification failed for centre: {vcko.centre_id}")
        self._vckos.append(vcko)

    def _compute_weights(self) -> np.ndarray:
        raw = np.array(
            [vcko.n_samples * float(vcko.metadata.get("local_auc", 1.0)) for vcko in self._vckos],
            dtype=float,
        )
        total = raw.sum()
        if total <= 0:
            return np.ones(len(self._vckos)) / len(self._vckos)
        return np.asarray(raw / total, dtype=float)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self._vckos:
            raise ValueError("No VCKOs added")

        feature_names = self._vckos[0].feature_names
        X = df[feature_names].to_numpy(dtype=float)
        weights = self._compute_weights()

        weighted_probs = np.zeros(len(X), dtype=float)
        for vcko, w in zip(self._vckos, weights):
            means = np.array(vcko.feature_means, dtype=float)
            stds = np.array(vcko.feature_stds, dtype=float)
            coefficients = np.array(vcko.coefficients, dtype=float)
            z = (X - means) / (stds + _EPS)
            logits = z @ coefficients + vcko.intercept
            probs = 1.0 / (1.0 + np.exp(-logits))
            weighted_probs += w * probs

        return np.asarray(weighted_probs, dtype=float)

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df) >= threshold).astype(int)

    def get_aggregated_coefficients(self) -> dict[str, float]:
        if not self._vckos:
            return {}

        feature_names = self._vckos[0].feature_names
        weights = self._compute_weights()
        all_coefs = np.array([vcko.coefficients for vcko in self._vckos])
        weighted_coefs = (all_coefs * weights[:, None]).sum(axis=0)
        return {name: float(coef) for name, coef in zip(feature_names, weighted_coefs)}
