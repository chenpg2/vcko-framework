# VCKO Framework

**Verifiable Clinical Knowledge Object** — A collaborative framework for privacy-preserving, Byzantine-robust multi-centre clinical prediction without shared infrastructure.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-83%20passed-brightgreen.svg)]()

<p align="center">
  <img src="cover_image_github.jpg" alt="VCKO framework overview" width="100%">
</p>

## Patent Notice

This project is covered by a filed patent application. Application number:
**202610687276.4**.

## The Problem

Multi-centre clinical prediction requires exchanging predictive knowledge between institutions without sharing patient data. Existing approaches each fall short:

- **Meta-analysis** lacks formal privacy guarantees and cannot detect computation fraud
- **Federated learning** requires multi-round infrastructure and leaks information through gradients
- **Neither** defends against malicious participants who submit poisoned model updates

## The VCKO Approach

VCKO reframes the question: instead of asking *"how to train collaboratively"*, it asks *"what knowledge object should institutions exchange?"*

Each centre fits a regularised logistic regression locally and packages the coefficients, per-coefficient standard errors (from the observed Fisher information), and a first-order optimality certificate into a JSON artifact of approximately 800 bytes. The receiving centre aggregates objects via inverse-variance (Fisher) weighting, which is provably equivalent to the one-shot distributed MLE.

```
Source Centres (private data stays local)
  Centre 1       Centre 2       Centre 3       Centre 4       Centre 5
  (47K cycles)   (57K cycles)   (83K cycles)   (74K cycles)   (73K cycles)
       |              |              |              |              |
       v              v              v              v              v
  [Local LR +    [Local LR +    [Local LR +    [Local LR +    [Local LR +
   Fisher info]   Fisher info]   Fisher info]   Fisher info]   Fisher info]
       |              |              |              |              |
       v              v              v              v              v
  VCKO_1          VCKO_2          VCKO_3          VCKO_4          VCKO_5
  {beta, se,      {beta, se,      {beta, se,      {beta, se,      {beta, se,
   mu, sigma,      mu, sigma,      mu, sigma,      mu, sigma,      mu, sigma,
   n, g, dp, H}   n, g, dp, H}   n, g, dp, H}   n, g, dp, H}   n, g, dp, H}
       |              |              |              |              |
       |   +--- Gaussian DP noise (calibrated per centre) ---+   |
       |   +--- Pairwise-mask secure sum (finite ring) ------+   |
       v              v              v              v              v
       +---------+---------+---------+---------+
                 |
                 v
  Target Centre (aggregation)
  1. Verify:  optimality certificate ||grad|| < tau
  2. Detect:  Cochran Q / I^2 heterogeneity test
  3. Pool:    Fisher-weighted (inverse-variance) aggregation
  4. Defend:  geometric-median robust combine (breakdown 1/2)
  5. Predict: P(y=1|x) from pooled coefficients
```

### Six improvements over the prior heuristic approach

| # | Prior limitation | VCKO solution |
|---|-----------------|---------------|
| 1 | Ad hoc n x AUC weighting | Statistically grounded distributed MLE via Fisher pooling |
| 2 | Post-hoc privacy bound only | Enforced Gaussian DP with Renyi accounting |
| 3 | No Byzantine defence | Geometric-median robust aggregation (breakdown 1/2) |
| 4 | SHA-256 tamper check only | Gradient-norm optimality certificate |
| 5 | No heterogeneity detection | Federated Cochran Q / I^2 test |
| 6 | Centres with missing features excluded | Partial-feature two-stage aggregation (3/5 to 5/5) |

## Mathematical Formulation

### Knowledge Object

```
VCKO_i = { beta_i, beta_0i, mu_i, sigma_i, se_i, n_i, g_i, lambda_i, dp_i, H_i }
```

where `se_i = sqrt(diag(H_i^{-1}))` are per-coefficient standard errors from the observed Fisher information, `g_i = ||grad L(beta_i) + lambda_i * beta_i||` is the optimality certificate, `dp_i = {epsilon, delta, sigma, Delta}` records the applied privacy mechanism, and `H_i = SHA-256(serialize(VCKO_i \ H_i))`.

### Fisher-Weighted Aggregation

Per-coordinate inverse-variance pooling (one-shot distributed MLE):

```
beta_hat_j = sum_i(w_ij * beta_ij) / sum_i(w_ij),   w_ij = 1 / se_ij^2
```

### Random-Effects Extension

DerSimonian-Laird between-centre variance tau^2 estimated per coordinate from Cochran Q:

```
w_ij = 1 / (se_ij^2 + tau^2_j)
```

### Differential Privacy

Calibrated Gaussian output perturbation with L2 sensitivity Delta_2 = 2/(n*lambda):

```
beta_noisy = beta + N(0, sigma^2 * I_d),   sigma = Delta_2 * sqrt(2 * log(1.25/delta)) / epsilon
```

### Byzantine Robustness

Geometric median minimises sum of Euclidean distances to all inputs (breakdown point 1/2):

```
beta_robust = argmin_b sum_i ||b - beta_i||_2
```

## Installation

```bash
git clone https://github.com/chenpg2/vcko-framework.git
cd vcko-framework
uv sync --extra dev
```

Standard pip also works:

```bash
pip install -e ".[dev]"
```

## Reviewer Quickstart

The repository contains no patient-level data. Reviewers can verify the install, core functionality, and static checks:

```bash
# Run all 83 tests
uv run pytest -q

# Static analysis
uv run ruff check .
uv run mypy src

# Synthetic quickstart
uv run python examples/01_quickstart.py
```

Paper-reproduction scripts require the approved private dataset:

```bash
uv run python scripts/run_loco.py --data-dir data/processed --bootstrap 1000
uv run python scripts/run_mia.py --data-dir data/processed --shadow-models 5
uv run python scripts/vcko_v2_benchmark.py
```

## Usage

### Step 1 -- Build VCKOs Locally

```python
from vcko import VCKOBuilder

builder = VCKOBuilder(
    feature_cols=["age", "AFC", "FSH", "LH", "oocytes", "embryos"],
    outcome_col="live_birth",
)

# Each centre fits locally; standard errors extracted from Fisher information
vcko_a = builder.fit(centre_a_df, centre_id="centre_a")
vcko_b = builder.fit(centre_b_df, centre_id="centre_b")

# Save for transmission (JSON, ~800 bytes)
vcko_a.save("vcko_centre_a.json")
```

### Step 2 -- Apply Differential Privacy

```python
from vcko.privacy.dp import gaussian_output_perturbation, calibrate_sigma, lr_l2_sensitivity

# Calibrate noise for (epsilon=5, delta=1e-5)-DP
sensitivity = lr_l2_sensitivity(n=50000, lam=0.01)
sigma = calibrate_sigma(epsilon=5.0, delta=1e-5, sensitivity=sensitivity)

# Apply Gaussian DP before transmission
dp_result = gaussian_output_perturbation(
    vcko_a.coefficients, sigma=sigma, sensitivity=sensitivity,
    epsilon=5.0, delta=1e-5,
)
```

### Step 3 -- Fisher-Weighted Aggregation

```python
from vcko.aggregation import fisher_weighted_pool, cochran_q

import numpy as np

# Collect coefficient matrices and SE matrices (shape: K x d)
betas = np.stack([v.coefficients for v in vcko_list])
ses = np.stack([v.standard_errors for v in vcko_list])

# Inverse-variance pooling (distributed MLE)
pooled_beta, pooled_se = fisher_weighted_pool(betas, ses)

# Heterogeneity detection per feature
Q, I2, p = cochran_q(betas, ses)
for j, feat in enumerate(feature_names):
    print(f"{feat}: Q={Q[j]:.1f}, I2={I2[j]:.1f}%, p={p[j]:.4f}")
```

### Step 4 -- Byzantine-Robust Aggregation

```python
from vcko.protocol import geometric_median

# Robust combine: tolerates < 50% adversarial inputs
beta_robust = geometric_median(
    [v.coefficients for v in vcko_list],
    max_iter=100,
    tol=1e-7,
)
```

### Step 5 -- Verify Computation Integrity

```python
from vcko.verification import logistic_gradient_norm, optimality_certificate

# Compute gradient norm (requires local data access at the source centre)
grad_norm = logistic_gradient_norm(X, y, beta, lam=0.01)

# Receiver checks the reported certificate
is_honest = optimality_certificate(grad_norm, tol=1e-2)
print(f"grad_norm={grad_norm:.6f}, honest={is_honest}")
```

### Step 6 -- Partial-Feature Aggregation

```python
from vcko.partial import PartialFeatureAggregator

agg = PartialFeatureAggregator()

# Register centres with their available feature sets
for vcko in vcko_list:
    agg.add(vcko)

# Stage 1: pool common features across ALL centres
# Stage 2: augment with extra features from contributing centres
result = agg.aggregate()
```

### Step 7 -- Evaluate and Validate

```python
from vcko.evaluation import calculate_auc, calculate_brier_score, compute_ece
from vcko.clinical import validate_coefficient_direction

auc = calculate_auc(y_true, probs)
brier = calculate_brier_score(y_true, probs)
cal = compute_ece(y_true, probs, n_bins=10)
print(f"AUC={auc:.4f}  Brier={brier:.4f}  ECE={cal.ece:.4f}")

# Coefficient direction vs ESHRE/Bologna guidelines
coefs = aggregator.get_aggregated_coefficients()
for feature, coef in coefs.items():
    v = validate_coefficient_direction(feature, coef)
    print(f"{v.feature}: coef={v.coefficient:.3f}  expected={v.expected_sign}  match={v.match}")
```

## Package Structure

```
src/vcko/
  __init__.py              # Public API: VCKOArtifact, VCKOBuilder, VCKOAggregator
  artifact.py              # VCKOArtifact dataclass + SHA-256 commitment
  builder.py               # Local logistic regression -> VCKO extraction with Fisher info
  aggregator.py            # Weighted ensemble aggregation
  aggregation.py           # Fisher (inverse-variance) pooling, Cochran Q, I^2, DerSimonian-Laird
  partial.py               # Two-stage partial-feature aggregation
  protocol.py              # Secure-robust-DP protocol stack (geometric median, secure sum, norm verification)
  verification.py          # Gradient-norm optimality certificate
  medium.py                # Prediction medium (standardisation + inference)
  data_utils.py            # Data preprocessing utilities
  privacy/
    dp.py                  # Gaussian DP mechanism + Renyi accounting
    discrete_dp.py         # Discrete DP mechanism (finite-ring compatible)
    mia.py                 # Membership Inference Attack evaluation
  evaluation/
    metrics.py             # AUC, Brier Score
    calibration.py         # Expected Calibration Error (ECE)
    dca.py                 # Decision Curve Analysis
  clinical/
    subgroups.py           # Bologna 2011 POR/NOR/HYR classification
    validation.py          # Coefficient direction vs ESHRE guidelines
```

## Key Results

Validated on 333,962 IVF cycles from 5 centres, 152,105 patients (leave-one-centre-out protocol):

### Aggregation quality

| Method | LOCO AUC (Core 6) | vs Pooled MLE |
|--------|-------------------|---------------|
| Fisher-weighted (VCKO) | 0.650 | Matches upper bound |
| Heuristic (n x AUC) | 0.648 | -0.002 |
| FedAvg (10 rounds) | 0.649 | -0.001 |

### Differential privacy utility

| Privacy budget (epsilon) | AUC | % retained |
|--------------------------|-----|------------|
| infinity (no DP) | 0.651 | 100.0% |
| 10 | 0.650 | 99.8% |
| 5 | 0.649 | 99.7% |
| 2 | 0.607 | 93.2% |
| 1 | 0.561 | 86.1% |

### Byzantine robustness

| Attack magnitude | Linear AUC | Geometric median AUC | Recovery |
|------------------|-----------|----------------------|----------|
| None (clean) | 0.650 | 0.651 | -- |
| 10x | 0.421 | 0.651 | +0.230 |
| 50x | 0.349 | 0.652 | +0.303 |
| 100x | 0.332 | 0.652 | +0.320 |

### Optimality certificate

| Object type | Gradient norm | Verdict |
|-------------|---------------|---------|
| Honestly fitted | < 6e-5 | Accept |
| Tampered coefficients | > 0.18 | Reject |

### Heterogeneity detection

| Feature | Cochran Q | I^2 (%) | Interpretation |
|---------|-----------|---------|----------------|
| Embryos transferred | 135.9 | 97.1 | Very high |
| Maternal age | 78.3 | 94.9 | High |
| Oocytes retrieved | 20.9 | 80.9 | High |

All statistics computed from transmitted knowledge objects alone, without patient-level data access.

## VCKO vs Alternatives

| Dimension | VCKO | Federated Learning | Meta-analysis |
|-----------|------|--------------------|---------------|
| What is exchanged | Coefficients + Fisher info (~800 B) | Model gradients per round | Summary statistics |
| Communication | One-shot | Tens to hundreds of rounds | One-shot |
| Privacy guarantee | Enforced Gaussian DP (Renyi) | Gradient leakage risk | None |
| Byzantine defence | Geometric median (breakdown 1/2) | None standard | None |
| Computation integrity | Gradient-norm certificate | None | None |
| Heterogeneity detection | Federated Cochran Q / I^2 | Not applicable | Cochran Q (requires raw data) |
| Partial participation | Two-stage aggregation | Requires shared architecture | Feature alignment |
| Infrastructure | Offline file transfer suffices | Real-time synchronisation | Offline |
| Interpretability | Coefficients directly inspectable | Black-box model weights | Coefficients inspectable |

## Citation

```bibtex
@article{chen2026vcko,
  title   = {VCKO: a collaborative framework for privacy-preserving,
             Byzantine-robust multi-centre clinical prediction
             without shared infrastructure},
  author  = {Chen, Peigen and Jin, Lei and Mao, Yundong and
             Zhang, Cuilian and Shi, Juanzi and Fang, Cong and Li, Tingting},
  journal = {npj Digital Medicine (submit)},
  year    = {2026}
}
```

## License

MIT -- see [LICENSE](LICENSE).
