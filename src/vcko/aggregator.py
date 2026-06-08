"""VCKO Aggregator - combine multiple VCKOs into a predictive model.

Two families of aggregation are supported:

1. ``heuristic`` (legacy, default): weighted ensemble of per-centre standardised
   sigmoids, ``w_i = n_i * AUC_i / sum_j n_j * AUC_j``. Kept for backward
   compatibility and as a baseline.

2. ``fixed_effect`` / ``random_effect`` (v2): statistically principled pooling of
   the coefficient vectors in a common (raw-feature) space using inverse-variance
   / DerSimonian-Laird weights. This is the one-shot distributed-MLE approximation
   (P-0002/P-0005/P-0015): it tracks the pooled MLE, optimally down-weights noisy
   or privatised centres, and supports a federated heterogeneity test and a
   stratified (prevalence-recalibrated) intercept (P-0010/P-0012).

Aggregation math lives in ``vcko.aggregation``; this class adapts VCKO objects to
it (scaled->raw coefficient transform, intercept handling, prediction).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import aggregation as agg
from .artifact import VCKOArtifact

_EPS = 1e-8


class VCKOAggregator:
    def __init__(self) -> None:
        self._vckos: list[VCKOArtifact] = []

    @property
    def vckos(self) -> list[VCKOArtifact]:
        return list(self._vckos)

    def add(self, vcko: VCKOArtifact) -> None:
        if not vcko.verify():
            raise ValueError(f"VCKO verification failed for centre: {vcko.centre_id}")
        self._vckos.append(vcko)

    # ------------------------------------------------------------------ legacy
    def _compute_weights(self) -> np.ndarray:
        raw = np.array(
            [v.n_samples * float(v.metadata.get("local_auc", 1.0)) for v in self._vckos],
            dtype=float,
        )
        total = raw.sum()
        if total <= 0:
            return np.ones(len(self._vckos)) / len(self._vckos)
        return np.asarray(raw / total, dtype=float)

    def _predict_heuristic(self, df: pd.DataFrame) -> np.ndarray:
        feature_names = self._vckos[0].feature_names
        X = df[feature_names].to_numpy(dtype=float)
        weights = self._compute_weights()
        out = np.zeros(len(X), dtype=float)
        for vcko, w in zip(self._vckos, weights):
            means = np.array(vcko.feature_means, dtype=float)
            stds = np.array(vcko.feature_stds, dtype=float)
            coefs = np.array(vcko.coefficients, dtype=float)
            z = (X - means) / (stds + _EPS)
            out += w * (1.0 / (1.0 + np.exp(-(z @ coefs + vcko.intercept))))
        return np.asarray(out, dtype=float)

    # ------------------------------------------------------------------ v2 core
    def _raw_space(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (betas_raw, ses_raw, intercepts_raw, n) across centres.

        Converts each centre's standardised coefficients/SEs into the shared raw
        feature space so they can be pooled coherently.
        """
        betas, ses, intercepts, ns = [], [], [], []
        for v in self._vckos:
            if v.coef_stderr is None:
                raise ValueError(
                    f"centre {v.centre_id} lacks coef_stderr; principled aggregation "
                    "requires v2 VCKOs (rebuild with the current VCKOBuilder)"
                )
            sigma = np.clip(np.array(v.feature_stds, dtype=float), 1e-12, None)
            mu = np.array(v.feature_means, dtype=float)
            beta_s = np.array(v.coefficients, dtype=float)
            se_s = np.array(v.coef_stderr, dtype=float)
            beta_raw = beta_s / sigma
            ses.append(se_s / sigma)
            betas.append(beta_raw)
            intercepts.append(float(v.intercept) - float(beta_raw @ mu))
            ns.append(v.n_samples)
        return (np.array(betas), np.array(ses), np.array(intercepts), np.array(ns, dtype=float))

    def pool(self, method: str = "fixed_effect") -> tuple[np.ndarray, float, np.ndarray]:
        """Pool coefficients in raw space; return (beta_raw, intercept, pooled_se)."""
        if method not in agg.AGGREGATION_METHODS:
            raise ValueError(f"unknown method '{method}'; choose {list(agg.AGGREGATION_METHODS)}")
        betas, ses, intercepts, ns = self._raw_space()
        beta_pooled, pooled_se = agg.AGGREGATION_METHODS[method](betas, ses)
        intercept_pooled = float(np.sum(ns * intercepts) / np.sum(ns))  # n-weighted
        return beta_pooled, intercept_pooled, pooled_se

    def _calibrate_intercept(
        self, X: np.ndarray, beta: np.ndarray, target_prevalence: float
    ) -> float:
        """Solve the logit offset so the mean predicted risk matches a prevalence."""
        base = X @ beta
        target_prevalence = float(np.clip(target_prevalence, 1e-6, 1 - 1e-6))

        def gap(b0: float) -> float:
            return float(np.mean(1.0 / (1.0 + np.exp(-(base + b0)))) - target_prevalence)

        return float(brentq(gap, -50.0, 50.0))

    def predict_proba(
        self,
        df: pd.DataFrame,
        method: str = "heuristic",
        target_prevalence: float | None = None,
    ) -> np.ndarray:
        """Predict P(y=1|x).

        Args:
            method: ``heuristic`` (legacy ensemble), ``fixed_effect`` or
                ``random_effect`` (principled raw-space pooled linear model).
            target_prevalence: if given (with a principled method), the intercept is
                recalibrated so the mean predicted risk matches it (stratified
                intercept, P-0012) — the clinically correct way to transport a model
                to a centre with a different baseline rate.
        """
        if not self._vckos:
            raise ValueError("No VCKOs added")
        if method == "heuristic":
            return self._predict_heuristic(df)

        beta, intercept, _ = self.pool(method)
        X = df[self._vckos[0].feature_names].to_numpy(dtype=float)
        if target_prevalence is not None:
            intercept = self._calibrate_intercept(X, beta, target_prevalence)
        return np.asarray(1.0 / (1.0 + np.exp(-(X @ beta + intercept))), dtype=float)

    def predict(
        self, df: pd.DataFrame, threshold: float = 0.5, method: str = "heuristic"
    ) -> np.ndarray:
        return (self.predict_proba(df, method=method) >= threshold).astype(int)

    # ------------------------------------------------------------ introspection
    def heterogeneity_report(self) -> dict[str, np.ndarray]:
        """Federated Cochran-Q / I^2 / tau^2 per coefficient (raw space).

        Decides, without any raw data, whether the common-effect (fixed) model is
        tenable or a random-effects model is warranted (resolves C3 via G4).
        """
        betas, ses, _, _ = self._raw_space()
        q, dof, pval, i2 = agg.cochran_q(betas, ses)
        tau2 = agg.dersimonian_laird_tau2(betas, ses)
        return {"Q": q, "df": np.array(dof), "p_value": pval, "I2": i2, "tau2": tau2}

    def get_aggregated_coefficients(self, method: str = "fixed_effect") -> dict[str, float]:
        if not self._vckos:
            return {}
        beta, _, _ = self.pool(method)
        return {name: float(c) for name, c in zip(self._vckos[0].feature_names, beta)}
