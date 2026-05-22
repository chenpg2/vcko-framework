"""VCKO Builder - Build VCKOs from local centre data."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .artifact import VCKOArtifact, create_commitment_hash


class VCKOBuilder:
    def __init__(
        self,
        feature_cols: list[str],
        outcome_col: str,
        model_type: str = "logistic",
        random_state: int = 42,
    ):
        self.feature_cols = feature_cols
        self.outcome_col = outcome_col
        self.model_type = model_type
        self.random_state = random_state

        if model_type != "logistic":
            raise ValueError(f"Only 'logistic' model supported, got: {model_type}")

    def fit(
        self,
        df: pd.DataFrame,
        centre_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VCKOArtifact:

        if centre_id is None:
            centre_id = "centre_unknown"

        if metadata is None:
            metadata = {}

        X = df[self.feature_cols].to_numpy()
        y = df[self.outcome_col].to_numpy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        model.fit(X_scaled, y)

        coefficients = model.coef_[0]
        intercept = float(model.intercept_[0])
        feature_means = scaler.mean_
        feature_stds = scaler.scale_
        n_samples = len(df)
        outcome_rate = float(np.mean(y))

        commitment = create_commitment_hash(
            coefficients=coefficients,
            intercept=intercept,
            feature_means=feature_means,
            feature_stds=feature_stds,
            n_samples=n_samples,
            outcome_rate=outcome_rate,
        )

        metadata_full = {
            **metadata,
            "model_type": self.model_type,
            "created_at": datetime.now().isoformat(),
        }

        return VCKOArtifact(
            centre_id=centre_id,
            feature_names=self.feature_cols,
            coefficients=coefficients.tolist(),
            intercept=intercept,
            feature_means=feature_means.tolist(),
            feature_stds=feature_stds.tolist(),
            n_samples=n_samples,
            outcome_rate=outcome_rate,
            commitment_hash=commitment,
            metadata=metadata_full,
        )
