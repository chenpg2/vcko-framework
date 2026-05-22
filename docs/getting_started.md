# Getting Started with VCKO

## Installation

For peer review, install from the repository:

```bash
git clone https://github.com/chenpg2/vcko-framework.git
cd vcko-framework
uv sync --extra dev
```

Without `uv`, use an editable install:

```bash
pip install -e ".[dev]"
```

For paper reproduction scripts, keep private patient-level data outside git.
The default expected path is `data/combine_cache.feather`; override it with
`VCKO_DATA_PATH=/path/to/combine_cache.feather`.

Reviewers can run the public smoke checks without private data:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src
uv run python examples/01_quickstart.py
```

## Quick Example

```python
from vcko import VCKOBuilder, VCKOAggregator
import pandas as pd

# Step 1: Each centre builds a VCKO locally
builder = VCKOBuilder(
    feature_cols=["age", "AFC", "FSH", "LH"],
    outcome_col="live_birth"
)

vcko_a = builder.fit(centre_a_data, centre_id="centre_a")
vcko_a.save("vcko_centre_a.json")

# Step 2: Target centre aggregates VCKOs
aggregator = VCKOAggregator()
aggregator.add(vcko_a)
aggregator.add(vcko_b)
aggregator.add(vcko_c)

# Step 3: Predict
predictions = aggregator.predict_proba(target_data)
```

## Core Concepts

### VCKOArtifact

A verifiable knowledge object containing:
- Model coefficients
- Feature statistics (means, stds)
- Cryptographic commitment hash
- Centre metadata

### VCKOBuilder

Trains a logistic regression model locally and extracts a VCKO.

### VCKOAggregator

Aggregates multiple VCKOs for prediction on target centre data.

## Next Steps

- See `examples/` for complete workflows
- Use `scripts/run_loco.py` and `scripts/run_mia.py` with approved private parquet files
- Check `data/README.md` for the expected local data layout
