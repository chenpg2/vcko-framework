"""Quickstart example - Build and aggregate VCKOs."""

import numpy as np
import pandas as pd

from vcko import VCKOAggregator, VCKOBuilder

np.random.seed(42)

n_samples = 1000
data_a = pd.DataFrame(
    {
        "age": np.random.normal(32, 5, n_samples),
        "AFC": np.random.normal(12, 4, n_samples),
        "FSH": np.random.normal(7, 2, n_samples),
        "LH": np.random.normal(5, 1.5, n_samples),
        "live_birth": np.random.binomial(1, 0.35, n_samples),
    }
)

data_b = pd.DataFrame(
    {
        "age": np.random.normal(33, 5, n_samples),
        "AFC": np.random.normal(11, 4, n_samples),
        "FSH": np.random.normal(7.5, 2, n_samples),
        "LH": np.random.normal(5.2, 1.5, n_samples),
        "live_birth": np.random.binomial(1, 0.32, n_samples),
    }
)

builder = VCKOBuilder(feature_cols=["age", "AFC", "FSH", "LH"], outcome_col="live_birth")

vcko_a = builder.fit(data_a, centre_id="centre_a")
vcko_b = builder.fit(data_b, centre_id="centre_b")

print(f"VCKO A verified: {vcko_a.verify()}")
print(f"VCKO B verified: {vcko_b.verify()}")

aggregator = VCKOAggregator()
aggregator.add(vcko_a)
aggregator.add(vcko_b)

test_data = pd.DataFrame(
    {
        "age": [30, 35, 28],
        "AFC": [15, 10, 18],
        "FSH": [6, 8, 5],
        "LH": [5, 6, 4],
    }
)

predictions = aggregator.predict_proba(test_data)
print(f"\nPredictions: {predictions}")

coefs = aggregator.get_aggregated_coefficients()
print("\nAggregated coefficients:")
for feat, coef in coefs.items():
    print(f"  {feat}: {coef:.4f}")
