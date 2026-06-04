"""VCKO Builder - Build VCKOs from local centre data.

v2 additionally extracts, per centre and entirely locally:
- ``coef_stderr``    sqrt(diag(H^{-1})) from the observed Fisher information, for
                     inverse-variance / random-effects aggregation;
- ``gradient_norm``  first-order optimality certificate (honest-computation check);
- optional **differential privacy** on the released coefficients (Gaussian output
  perturbation with a calibrated, formally-accounted (eps, delta)), with the noise
  variance folded into ``coef_stderr`` so the aggregator down-weights private centres.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .artifact import VCKOArtifact, create_commitment_hash
from .privacy.dp import gaussian_output_perturbation, lr_l2_sensitivity
from .verification import logistic_gradient_norm

_RIDGE = 1e-8


@dataclass(frozen=True)
class PrivacyConfig:
    """Differential-privacy request for a VCKO release."""

    epsilon: float
    delta: float = 1e-5


class VCKOBuilder:
    def __init__(
        self,
        feature_cols: list[str],
        outcome_col: str,
        model_type: str = "logistic",
        random_state: int = 42,
        regularization: float | None = None,
        privacy: PrivacyConfig | None = None,
    ):
        self.feature_cols = feature_cols
        self.outcome_col = outcome_col
        self.model_type = model_type
        self.random_state = random_state
        self.regularization = regularization
        self.privacy = privacy

        if model_type != "logistic":
            raise ValueError(f"Only 'logistic' model supported, got: {model_type}")
        if privacy is not None and regularization is None:
            raise ValueError(
                "differential privacy requires an explicit 'regularization' (lambda) "
                "to bound the L2 sensitivity"
            )

    def _fisher_stderr(self, X_scaled: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Standard errors from the observed Fisher information at the fitted beta."""
        w = np.clip(p * (1.0 - p), 1e-9, None)
        fisher = X_scaled.T @ (X_scaled * w[:, None])
        fisher += _RIDGE * np.eye(fisher.shape[0])
        cov = np.linalg.inv(fisher)
        return np.asarray(np.sqrt(np.clip(np.diag(cov), 1e-18, None)), dtype=float)

    def fit(
        self,
        df: pd.DataFrame,
        centre_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VCKOArtifact:
        centre_id = centre_id or "centre_unknown"
        metadata = dict(metadata or {})

        X = df[self.feature_cols].to_numpy()
        y = df[self.outcome_col].to_numpy()
        n_samples = len(df)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # L2 strength: sklearn objective is 0.5||w||^2 + C * sum loss, so the
        # per-sample regulariser is lambda = 1/(n*C). Honour an explicit lambda.
        if self.regularization is not None:
            C = 1.0 / (n_samples * self.regularization)
            lam = self.regularization
        else:
            C = 1.0
            lam = 1.0 / n_samples

        model = LogisticRegression(C=C, random_state=self.random_state, max_iter=1000)
        model.fit(X_scaled, y)

        coefficients = model.coef_[0].astype(float)
        intercept = float(model.intercept_[0])
        feature_means = scaler.mean_
        feature_stds = scaler.scale_
        outcome_rate = float(np.mean(y))

        p = model.predict_proba(X_scaled)[:, 1]
        stderr = self._fisher_stderr(X_scaled, p)
        # Optimality certificate computed on the honest (pre-privacy) fit.
        grad_norm = logistic_gradient_norm(X_scaled, y, coefficients, lam, intercept)

        dp_meta: dict[str, Any] | None = None
        if self.privacy is not None:
            rng = np.random.default_rng(self.random_state)
            sens = lr_l2_sensitivity(n_samples, lam)
            dp_out = gaussian_output_perturbation(
                coefficients,
                sensitivity=sens,
                epsilon=self.privacy.epsilon,
                delta=self.privacy.delta,
                rng=rng,
            )
            coefficients = dp_out.noised_beta.astype(float)
            # Fold privacy noise into the reported precision so the aggregator
            # optimally down-weights noisier (more private) centres.
            stderr = np.sqrt(stderr**2 + dp_out.sigma**2)
            dp_meta = {
                "mechanism": dp_out.mechanism,
                "epsilon": dp_out.epsilon,
                "delta": dp_out.delta,
                "sigma": dp_out.sigma,
                "sensitivity": dp_out.sensitivity,
            }

        coef_list = coefficients.tolist()
        stderr_list = stderr.tolist()
        commitment = create_commitment_hash(
            coefficients=coefficients,
            intercept=intercept,
            feature_means=feature_means,
            feature_stds=feature_stds,
            n_samples=n_samples,
            outcome_rate=outcome_rate,
            coef_stderr=stderr_list,
            gradient_norm=grad_norm,
            regularization=lam,
            dp=dp_meta,
        )

        metadata_full = {
            **metadata,
            "model_type": self.model_type,
            "created_at": datetime.now().isoformat(),
        }

        return VCKOArtifact(
            centre_id=centre_id,
            feature_names=self.feature_cols,
            coefficients=coef_list,
            intercept=intercept,
            feature_means=feature_means.tolist(),
            feature_stds=feature_stds.tolist(),
            n_samples=n_samples,
            outcome_rate=outcome_rate,
            commitment_hash=commitment,
            metadata=metadata_full,
            coef_stderr=stderr_list,
            gradient_norm=grad_norm,
            regularization=lam,
            dp=dp_meta,
        )
