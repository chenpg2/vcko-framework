# Data Availability

Patient data is NOT stored in this repository.

The public repository is intended for code inspection, installation checks, and
synthetic examples. Paper reproduction scripts require an approved private dataset
held outside git.

Some internal analysis utilities use a private feather file named
`combine_cache.feather`. By default they look for it at `data/combine_cache.feather`;
set `VCKO_DATA_PATH=/path/to/combine_cache.feather` to use a different local
location.

The public LOCO and MIA scripts expect anonymised per-centre parquet files under
`data/processed/` when a reviewer has been granted data access.

## Structure

```
data/
├── raw/           # Raw data files (gitignored)
├── combine_cache.feather  # Private analytic dataset, gitignored
└── processed/     # Preprocessed parquet files (gitignored)
    ├── centre_1.parquet
    ├── centre_2.parquet
    ├── centre_3.parquet
    ├── centre_4.parquet
    └── centre_5.parquet
```

## Data Format

Each centre parquet file must contain columns:
- `age`: Patient age (float)
- `AFC`: Antral Follicle Count (float)
- `FSH`: Follicle-Stimulating Hormone (float)
- `LH`: Luteinizing Hormone (float)
- `live_birth`: Outcome (0 or 1)

## Preprocessing

All preprocessing and data access must be scripted. Do not commit raw, processed, or
derived patient-level files to git.
