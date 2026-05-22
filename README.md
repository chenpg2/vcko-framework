# VCKO Framework

**Verifiable Clinical Knowledge Object** — Privacy-preserving multi-centre clinical prediction through knowledge object exchange.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

<p align="center">
  <img src="cover_image_github.jpg" alt="VCKO framework overview" width="100%">
</p>

## Patent Notice

This project is covered by a filed patent application. Application number:
**202610687276.4**.

## The Problem

Small medical centres lack sufficient data to train reliable prediction models. Large centres have data, but privacy regulations (GDPR, China PIPL) prohibit sharing patient-level records. Traditional federated learning mitigates this by sharing gradients instead of data — but gradients still leak individual information and require complex multi-round communication infrastructure.

## The VCKO Approach

VCKO reframes the question: instead of asking *"how to train collaboratively"*, it asks *"what knowledge object should institutions exchange?"*

Each source centre trains a local logistic regression, then extracts a compact knowledge object — coefficients, feature statistics, and a cryptographic commitment — totalling O(d) dimensions (d = number of features). No patient-level data, no gradients. The target centre collects these objects and produces weighted ensemble predictions.

```
Source Centres (private data stays local)
  Centre 1 (47K)    Centre 2 (57K)    Centre 3 (83K)    Centre 4 (74K)
       |                 |                 |                 |
       v                 v                 v                 v
  [Local LR]        [Local LR]        [Local LR]        [Local LR]
       |                 |                 |                 |
       v                 v                 v                 v
  VCKO_1             VCKO_2             VCKO_3             VCKO_4
  {beta, mu, sigma,  {beta, mu, sigma,  {beta, mu, sigma,  {beta, mu, sigma,
   n, AUC, H}        n, AUC, H}        n, AUC, H}        n, AUC, H}
       |                 |                 |                 |
       +--------+--------+--------+-------+
                |  TLS + hash verification
                v
  Target Centre (e.g. a small hospital with 500 cases)
  1. Verify: H' = SHA-256(VCKO_i) == H_i
  2. Weight: w_i = n_i * AUC_i / sum(n_j * AUC_j)
  3. Predict: P(y=1|x) = sum( w_i * sigmoid(beta_i^T z_i + beta_0i) )
  4. Privacy audit: MIA evaluation
  5. Clinical validation: coefficient direction + subgroup analysis
```

## Mathematical Formulation

**Local Training (Centre i):**
Each centre fits a logistic regression on standardised features and packages the result:

```
VCKO_i = { beta_i, beta_0i, mu_i, sigma_i, n_i, AUC_i, H_i }
where H_i = SHA-256(serialize(VCKO_i))
```

**Weighted Aggregation (Target Centre):**
Given K received VCKOs, the target centre computes:

```
Weight:    w_i = (n_i * AUC_i) / sum_{j=1}^{K} (n_j * AUC_j)
Normalise: z_i = (x - mu_i) / (sigma_i + eps)
Predict:   P(y=1|x) = sum_{i=1}^{K} w_i * sigmoid(beta_i^T z_i + beta_0i)
```

Centres with more data (higher n) and better local performance (higher AUC) receive proportionally greater weight.

**Verification:**
The receiver recomputes `H' = SHA-256(serialize(VCKO))` and accepts only if `H' == H`.

## Installation

For editor/reviewer inspection, install directly from this repository:

```bash
git clone https://github.com/chenpg2/vcko-framework.git
cd vcko-framework
uv sync --extra dev
```

If you are not using `uv`, a standard editable install also works:

```bash
pip install -e ".[dev]"
```

The package name is reserved for release after publication; source installation is the
supported path during peer review.

## Reviewer Quickstart

The public repository contains no patient-level data. Editors and reviewers can still
verify the install, core package behavior, static checks, and synthetic quickstart:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src
uv run python examples/01_quickstart.py
```

Optional paper-reproduction entry points are provided in `scripts/`. They require the
approved private centre-level dataset, kept outside git:

```bash
uv run python scripts/run_loco.py --data-dir data/processed --bootstrap 1000
uv run python scripts/run_mia.py --data-dir data/processed --shadow-models 5
```

See `data/README.md` and `docs/getting_started.md` for the expected local data layout.

## Usage

### Step 1 — Build VCKOs Locally

```python
from vcko import VCKOBuilder

builder = VCKOBuilder(
    feature_cols=["age", "AFC", "FSH", "LH"],
    outcome_col="live_birth",
)

# Each centre fits on its own data
vcko_a = builder.fit(centre_a_df, centre_id="centre_a")
vcko_b = builder.fit(centre_b_df, centre_id="centre_b")

# Save for transmission (JSON, ~1-10 KB)
vcko_a.save("vcko_centre_a.json")
```

### Step 2 — Verify and Aggregate

```python
from vcko import VCKOAggregator, VCKOArtifact

aggregator = VCKOAggregator()
for path in ["vcko_a.json", "vcko_b.json", "vcko_c.json"]:
    vcko = VCKOArtifact.load(path)
    aggregator.add(vcko)  # automatically verifies commitment hash

probs = aggregator.predict_proba(target_df)
preds = aggregator.predict(target_df, threshold=0.5)
```

### Step 3 — Evaluate Performance

```python
from vcko.evaluation import calculate_auc, calculate_brier_score, compute_ece

auc = calculate_auc(y_true, probs)
brier = calculate_brier_score(y_true, probs)
cal = compute_ece(y_true, probs, n_bins=10)
print(f"AUC={auc:.4f}  Brier={brier:.4f}  ECE={cal.ece:.4f}")
```

Decision curve analysis:

```python
from vcko.evaluation import decision_curve_analysis

dca = decision_curve_analysis(y_true, probs)
# dca.thresholds, dca.net_benefit_model, dca.net_benefit_all
```

### Step 4 — Privacy Audit

```python
from vcko.privacy import MIAEvaluator

mia = MIAEvaluator(n_shadow_models=5)
result = mia.evaluate(vcko_a, member_df, nonmember_df)
print(f"MIA AUC: {result.auc:.4f}")  # ~0.50 means privacy-safe
```

### Step 5 — Clinical Validation

Verify that model coefficients align with established clinical guidelines:

```python
from vcko.clinical import validate_coefficient_direction

coefs = aggregator.get_aggregated_coefficients()
for feature, coef in coefs.items():
    v = validate_coefficient_direction(feature, coef)
    print(f"{v.feature}: coef={v.coefficient:.3f}  expected={v.expected_sign}  match={v.match}")
```

Bologna 2011 subgroup analysis:

```python
from vcko.clinical import assign_bologna_subgroup

# Classify patients into POR / NOR / HYR
subgroup = assign_bologna_subgroup(age=38, afc=4, fsh=11)
print(subgroup)  # BolognaSubgroup.POR
```

## Package Structure

```
vcko/
  __init__.py          # Public API: VCKOArtifact, VCKOBuilder, VCKOAggregator
  artifact.py          # VCKOArtifact dataclass + SHA-256 commitment
  builder.py           # Local LogisticRegression training -> VCKO extraction
  aggregator.py        # Weighted ensemble: w_i = n_i*AUC_i / sum(n_j*AUC_j)
  privacy/
    mia.py             # Membership Inference Attack evaluation
  evaluation/
    metrics.py         # AUC, Brier Score
    calibration.py     # Expected Calibration Error (ECE)
    dca.py             # Decision Curve Analysis
  clinical/
    subgroups.py       # Bologna 2011 POR/NOR/HYR classification
    validation.py      # Coefficient direction vs ESHRE guidelines
```

## Key Results

Validated on 333,962 IVF cycles across 5 centres (leave-one-centre-out protocol):

| Property | Experiment | Result |
|----------|-----------|--------|
| Constructible | E1 | 5/5 centres built in <0.02s |
| Verifiable | E2 | 20/20 cross-centre pairs verified, SHA-256 100% match |
| Privacy-safe | E3 | MIA AUC = 0.50 (random guess level), equivalent to DP-SGD at epsilon <= 1 |
| Effective | E4 | LOCO AUC = 0.778 (95% CI: 0.770-0.784) |
| Clinically interpretable | E5 | 83% coefficient directions match ESHRE/Bologna consensus |
| Universally applicable | E6 | Positive lift across all Bologna subgroups (POR/NOR/HYR) |
| Decision-ready | E7 | Cohen's Kappa = 0.951 vs pooled model (98.1% agreement) |
| Robust | E8 | Seed CV = 0.00%, Bootstrap CI excludes zero |
| Data-efficient | E9 | 100 local samples + VCKO matches 20,000 samples standalone (200x efficiency) |

## VCKO vs Federated Learning

| Dimension | VCKO | Federated Learning |
|-----------|------|--------------------|
| What is exchanged | Aggregated statistics (coefficients + moments) | Model gradients per training round |
| Communication | One-shot | Tens to hundreds of rounds |
| Information dimensionality | O(d), where d = number of features | O(p), where p = full model parameters |
| Privacy risk | Minimal — only aggregated statistics | Gradients can reconstruct individual records |
| Infrastructure | Offline file transfer suffices | Real-time synchronisation required |
| Interpretability | Coefficients directly inspectable | Black-box model weights |

## Citation

Citation details will be added after publication.

```bibtex
@article{vcko2026,
  title   = {Verifiable Clinical Knowledge Objects for Privacy-Preserving
             Multi-Centre Prediction},
  author  = {VCKO Team},
  journal = {Forthcoming},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
