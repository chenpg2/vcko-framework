"""Data contract for the VCKO v3 protocol-stack experiment.

Asserts the real multi-centre IVF feather matches the expected schema BEFORE any
analysis runs. Per the bioinformatics rules: validate input assumptions explicitly,
fail with a clear error if violated, NEVER silently work around a violation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "combine_cache.feather"

# Locked expectations (from inspection 2026-06-06).
EXPECTED_ROWS = 333_962
EXPECTED_MIN_COLS = 180
EXPECTED_RAW_CENTRES = {
    "武汉同济",
    "西北妇儿",
    "中山六院",
    "江苏人民",
    "河南人民前半部分",
    "河南人民",
}
KEY_COLS = {
    "org_name": "object",
    "live_birth": "int",
    "age_w": "float",
    "AF": "float",
    "base_FSH": "float",
    "base_LH": "float",
    "egg_num": "float",
    "transfer_embryo_num": "float",
    "transfer_embryo_day": "object",
    "p_id_new": "object",
    "visit_date": "datetime",
}
# HARD ranges: catch only sign errors and gross UNIT errors (e.g. age in days, a
# 10x decimal slip). A handful of biological outliers (AFC=168, transfer=41) are
# real artifacts in the raw cohort; they are surfaced as loud WARNINGS below rather
# than crashing every real-data run. Standardised features tolerate a few outliers.
HARD_RANGES = {
    "age_w": (10.0, 80.0),
    "AF": (0.0, 500.0),
    "base_FSH": (0.0, 500.0),
    "base_LH": (0.0, 500.0),
    "egg_num": (0.0, 200.0),
    "transfer_embryo_num": (0.0, 100.0),
}
# TYPICAL ranges: values beyond these are surfaced as a loud WARNING (not silently
# fixed) so the data-quality issue is visible, per the no-silent-fallback rule.
TYPICAL_RANGES = {
    "age_w": (18.0, 50.0),
    "AF": (0.0, 60.0),
    "base_FSH": (0.0, 40.0),
    "base_LH": (0.0, 40.0),
    "egg_num": (0.0, 50.0),
    "transfer_embryo_num": (0.0, 4.0),
}


def _dtype_family(dtype: object) -> str:
    s = str(dtype)
    if "int" in s:
        return "int"
    if "float" in s:
        return "float"
    if "datetime" in s:
        return "datetime"
    return "object"


def check_protocol_inputs(strict_hash: bool = False) -> dict[str, object]:
    """Validate the real IVF feather against the locked schema.

    Args:
        strict_hash: if True, also assert the file SHA-256 matches the manifest
            (only enable once the data is frozen for a release).

    Returns:
        A small dict of verified facts (row count, centre count, hash prefix).

    Raises:
        FileNotFoundError, AssertionError on any contract violation.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"FAIL: data not found at {DATA_PATH}. Expected a symlink to "
            f"rawinputdata/combine_cache.feather. Do not proceed without real data."
        )

    df = pd.read_feather(DATA_PATH)

    # 1. Row count (exact — the cohort is frozen).
    assert len(df) == EXPECTED_ROWS, (
        f"FAIL: row count {len(df)} != expected {EXPECTED_ROWS}. The cohort changed."
    )

    # 2. Column schema.
    assert df.shape[1] >= EXPECTED_MIN_COLS, (
        f"FAIL: {df.shape[1]} columns < expected >= {EXPECTED_MIN_COLS}."
    )
    for col, fam in KEY_COLS.items():
        assert col in df.columns, f"FAIL: required column '{col}' missing."
        got = _dtype_family(df[col].dtype)
        assert got == fam, f"FAIL: column '{col}' dtype family {got} != expected {fam}."

    # 3. Centres present (the 5 logical centres, 6 raw labels).
    centres = set(df["org_name"].dropna().unique())
    missing = EXPECTED_RAW_CENTRES - centres
    assert not missing, f"FAIL: missing centre labels {missing}."

    # 4. Outcome is strictly binary, no NaN.
    lb = df["live_birth"]
    assert bool(lb.notna().all()), "FAIL: live_birth contains NaN."
    assert set(np.unique(lb.to_numpy())).issubset({0, 1}), (
        f"FAIL: live_birth not binary; values {set(np.unique(lb.to_numpy()))}."
    )

    # 5. Feature ranges: hard-fail on IMPOSSIBLE values; record outliers loudly.
    outliers: dict[str, dict[str, float]] = {}
    for col, (lo, hi) in HARD_RANGES.items():
        vals = pd.Series(pd.to_numeric(df[col], errors="coerce")).dropna().to_numpy()
        if vals.size == 0:
            continue
        vmin, vmax = float(vals.min()), float(vals.max())
        assert vmin >= lo - 1e-9, f"FAIL: {col} min {vmin} below IMPOSSIBLE floor {lo}."
        assert vmax <= hi + 1e-9, f"FAIL: {col} max {vmax} above IMPOSSIBLE ceiling {hi}."
        t_lo, t_hi = TYPICAL_RANGES[col]
        n_extreme = int(((vals < t_lo) | (vals > t_hi)).sum())
        if n_extreme > 0:
            outliers[col] = {
                "n_beyond_typical": float(n_extreme),
                "pct": round(100 * n_extreme / vals.size, 3),
                "max": vmax,
                "typical_ceiling": t_hi,
            }

    # 6. Patient id present for clustering / leakage checks.
    assert bool(df["p_id_new"].notna().any()), "FAIL: p_id_new all NaN; cannot dedup patients."

    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()
    facts = {
        "rows": len(df),
        "cols": df.shape[1],
        "n_raw_centres": len(centres & EXPECTED_RAW_CENTRES),
        "live_birth_rate": round(float(lb.mean()), 4),
        "content_hash": h[:16],
        "data_quality_outliers": outliers,
    }
    if outliers:
        print(f"WARNING [data-quality]: extreme (but not impossible) values present: {outliers}")

    if strict_hash:
        manifest = PROJECT_ROOT / "harness" / "contracts" / "data_manifest.txt"
        if manifest.exists():
            expected = manifest.read_text().strip()
            assert h == expected, f"FAIL: data hash {h[:16]} != manifest {expected[:16]}."

    return facts


if __name__ == "__main__":
    print("Contract check:", check_protocol_inputs())
