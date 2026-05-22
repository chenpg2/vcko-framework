# Reviewer Notes

This repository is prepared for editor and reviewer inspection during peer review.

## Public Contents

- `src/vcko/`: reusable VCKO package code.
- `examples/`: synthetic quickstart that does not require patient data.
- `tests/`: smoke tests for verifiable artifact behavior.
- `conf/`: default configuration values for reviewer reference.
- `scripts/run_loco.py`: leave-one-centre-out evaluation entry point for approved local parquet data.
- `scripts/run_mia.py`: membership inference audit entry point for approved local parquet data.
- `data/README.md`: expected private data layout.

## Data Boundary

No patient-level records, centre exports, generated experiment outputs, submission PDFs,
cover letters, or patent drafts are included in the public git history.

The clinical dataset used for the manuscript remains under governance restrictions.
Reviewers with approved access can place anonymised per-centre parquet files at
`data/processed/centre_*.parquet` and run:

```bash
uv run python scripts/run_loco.py --data-dir data/processed --bootstrap 1000
uv run python scripts/run_mia.py --data-dir data/processed --shadow-models 5
```

## Public Verification

These checks require no private data:

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run mypy src
uv run python examples/01_quickstart.py
```
