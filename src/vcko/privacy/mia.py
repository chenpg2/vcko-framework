"""Membership Inference Attack evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from ..artifact import VCKOArtifact


@dataclass(frozen=True)
class MIAResult:
    auc: float
    accuracy: float
    true_positive_rate: float
    false_positive_rate: float


class MIAEvaluator:
    def __init__(self, n_shadow_models: int = 5, random_state: int = 42):
        self.n_shadow_models = n_shadow_models
        self.random_state = random_state

    def evaluate(
        self,
        vcko: VCKOArtifact,
        member_df: pd.DataFrame,
        nonmember_df: pd.DataFrame,
    ) -> MIAResult:

        features_member = self._extract_features(vcko, member_df)
        features_nonmember = self._extract_features(vcko, nonmember_df)

        X = np.vstack([features_member, features_nonmember])
        y = np.hstack([np.ones(len(features_member)), np.zeros(len(features_nonmember))])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )

        attack_model = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state, max_depth=5
        )
        attack_model.fit(X_train, y_train)

        y_pred = attack_model.predict(X_test)
        y_pred_proba = attack_model.predict_proba(X_test)[:, 1]

        auc = float(roc_auc_score(y_test, y_pred_proba))
        accuracy = float((y_pred == y_test).mean())

        tp = float(((y_pred == 1) & (y_test == 1)).sum() / (y_test == 1).sum())
        fp = float(((y_pred == 1) & (y_test == 0)).sum() / (y_test == 0).sum())

        return MIAResult(auc=auc, accuracy=accuracy, true_positive_rate=tp, false_positive_rate=fp)

    def _extract_features(self, vcko: VCKOArtifact, df: pd.DataFrame) -> np.ndarray:

        X = df[vcko.feature_names].values
        X_scaled = (X - np.array(vcko.feature_means)) / np.array(vcko.feature_stds)
        logits = X_scaled @ np.array(vcko.coefficients) + vcko.intercept
        probs = 1 / (1 + np.exp(-logits))

        features = np.column_stack(
            [
                probs,
                np.abs(probs - 0.5),
                logits,
                np.abs(logits),
            ]
        )

        return features
