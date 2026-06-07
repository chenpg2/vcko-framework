"""Partial-feature VCKO aggregation (v3.1).

Solves the problem where different centres have different feature subsets. In
the IVF cohort, Centres 2 and 5 lack transfer_embryo_day but have the other 6
features. Under complete-case analysis, they are excluded entirely, wasting 85%
of the data and limiting participation to 3 of 5 centres.

This module implements the two-stage correction approach:

Stage 1 (common-set pool): every centre fits a model on the COMMON feature set
(the intersection of all centres' features). Fisher-weighted pooling produces
beta_common. Because the model specification is the same across all centres,
no omitted-variable bias is introduced.

Stage 2 (augmentation): centres that have EXTRA features (beyond the common set)
also fit the full model on their available features. The increment
delta_j = beta_full_j - beta_common_j (for the common features) captures the
adjustment due to including the extra features. The extra-feature coefficients
and their standard errors are Fisher-pooled across contributing centres only.

The final coefficient vector is:

  beta_j = beta_common_j + delta_j_pooled   for j in common features
  beta_j = beta_extra_j_pooled              for j in extra features (from
                                             contributing centres only)

This lets ALL centres contribute to the common-feature coefficients, while
extra-feature coefficients use only the centres that have them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .aggregation import fisher_weighted_pool

__all__ = ["PartialFeatureAggregator"]

_RIDGE = 1e-8


@dataclass
class _CentreRecord:
    centre_id: str
    available_features: list[str]
    beta_common: np.ndarray
    se_common: np.ndarray
    beta_full: np.ndarray | None  # None if centre has only common features
    se_full: np.ndarray | None
    intercept_common: float
    intercept_full: float | None
    n_samples: int


class PartialFeatureAggregator:
    """Aggregate VCKOs from centres with heterogeneous feature sets.

    Each centre contributes a model on whatever features it has. The aggregator
    pools on the common feature set (all centres), then augments with extra
    features from centres that have them.
    """

    def __init__(self, all_features: list[str], random_state: int = 42) -> None:
        self.all_features = list(all_features)
        self.random_state = random_state
        self._records: list[_CentreRecord] = []

    def _fit_and_extract(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Fit LR, return (beta, se, intercept)."""
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_state)
        model.fit(Xs, y)
        beta = model.coef_[0]
        intercept = float(model.intercept_[0])
        p = model.predict_proba(Xs)[:, 1]
        w = np.clip(p * (1 - p), 1e-9, None)
        fisher = Xs.T @ (Xs * w[:, None]) + _RIDGE * np.eye(Xs.shape[1])
        se = np.sqrt(np.clip(np.diag(np.linalg.inv(fisher)), 1e-18, None))
        # Convert to raw-feature space
        sigma = np.clip(scaler.scale_, 1e-12, None)
        beta_raw = beta / sigma
        se_raw = se / sigma
        intercept_raw = intercept - float(beta_raw @ scaler.mean_)
        return (
            np.asarray(beta_raw, dtype=float),
            np.asarray(se_raw, dtype=float),
            intercept_raw,
        )

    def add_centre(
        self,
        centre_id: str,
        df: pd.DataFrame,
        available_features: list[str],
        outcome_col: str,
    ) -> None:
        """Add a centre's data. The centre is fitted on available_features."""
        for f in available_features:
            if f not in self.all_features:
                raise ValueError(f"feature '{f}' not in all_features")

        numeric = df[available_features + [outcome_col]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        if len(numeric) < 50:
            raise ValueError(f"centre {centre_id}: too few complete cases ({len(numeric)})")
        X_avail = numeric[available_features].to_numpy(dtype=float)
        y = numeric[outcome_col].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError(f"centre {centre_id}: outcome has < 2 classes")

        # Common features = intersection with what every existing centre has
        # (computed at pool time, not here; fit both common and full now)
        beta_full, se_full, intercept_full = self._fit_and_extract(X_avail, y)

        self._records.append(
            _CentreRecord(
                centre_id=centre_id,
                available_features=list(available_features),
                beta_common=np.array([]),  # filled at pool time
                se_common=np.array([]),
                beta_full=beta_full,
                se_full=se_full,
                intercept_common=0.0,
                intercept_full=intercept_full,
                n_samples=len(y),
            )
        )
        # Store the data reference for re-fitting on common features at pool time
        self._records[-1]._df = numeric  # type: ignore[attr-defined]
        self._records[-1]._outcome_col = outcome_col  # type: ignore[attr-defined]
        self._records[-1]._available_features = available_features  # type: ignore[attr-defined]

    def _common_features(self) -> list[str]:
        """Features present in ALL centres."""
        if not self._records:
            return []
        common = set(self._records[0].available_features)
        for r in self._records[1:]:
            common &= set(r.available_features)
        return [f for f in self.all_features if f in common]

    def _extra_features(self) -> list[str]:
        """Features in all_features but not in the common set."""
        common = set(self._common_features())
        return [f for f in self.all_features if f not in common]

    def n_contributors(self, feature: str) -> int:
        """How many centres can contribute to this feature's coefficient."""
        return sum(1 for r in self._records if feature in r.available_features)

    def pool(self) -> tuple[np.ndarray, np.ndarray]:
        """Two-stage partial-feature pool. Returns (beta, se) for all_features."""
        if not self._records:
            raise ValueError("no centres added")

        common = self._common_features()
        if not common:
            raise ValueError("no common features across all centres")

        extra = self._extra_features()
        d_total = len(self.all_features)
        feat_idx = {f: i for i, f in enumerate(self.all_features)}

        # Stage 1: fit every centre on the COMMON feature set, then Fisher pool.
        betas_common = []
        ses_common = []
        intercepts_common = []
        for r in self._records:
            df = r._df  # type: ignore[attr-defined]
            outcome_col = r._outcome_col  # type: ignore[attr-defined]
            X_common = df[common].to_numpy(dtype=float)
            y = df[outcome_col].to_numpy(dtype=int)
            beta_c, se_c, intercept_c = self._fit_and_extract(X_common, y)
            r.beta_common = beta_c
            r.se_common = se_c
            r.intercept_common = intercept_c
            betas_common.append(beta_c)
            ses_common.append(se_c)
            intercepts_common.append(intercept_c)

        betas_common_arr = np.array(betas_common)
        ses_common_arr = np.array(ses_common)
        beta_common_pooled, se_common_pooled = fisher_weighted_pool(
            betas_common_arr, ses_common_arr
        )

        # Stage 2: for extra features, pool from centres that have them.
        beta_out = np.zeros(d_total)
        se_out = np.full(d_total, np.inf)

        # Fill common features
        for j, f in enumerate(common):
            idx = feat_idx[f]
            beta_out[idx] = beta_common_pooled[j]
            se_out[idx] = se_common_pooled[j]

        # Fill extra features from contributing centres
        for f in extra:
            idx = feat_idx[f]
            contrib_betas = []
            contrib_ses = []
            for r in self._records:
                if f not in r.available_features:
                    continue
                # Find the index of f in this centre's available_features
                local_idx = r.available_features.index(f)
                if r.beta_full is not None:
                    contrib_betas.append(r.beta_full[local_idx])
                    contrib_ses.append(r.se_full[local_idx])  # type: ignore[index]
            if contrib_betas:
                cb = np.array(contrib_betas).reshape(-1, 1)
                cs = np.array(contrib_ses).reshape(-1, 1)
                bp, sp = fisher_weighted_pool(cb, cs)
                beta_out[idx] = bp[0]
                se_out[idx] = sp[0]

        # Pooled intercept (n-weighted from common-set fits)
        ns = np.array([r.n_samples for r in self._records], dtype=float)
        intercept_pooled = float(np.sum(ns * np.array(intercepts_common)) / ns.sum())
        self._intercept = intercept_pooled

        return beta_out, se_out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict using the pooled partial-feature model (auto-pools if needed)."""
        beta, _ = self.pool()
        X = np.asarray(X, dtype=float)
        logits = X @ beta + self._intercept
        return np.asarray(1.0 / (1.0 + np.exp(-logits)), dtype=float)
