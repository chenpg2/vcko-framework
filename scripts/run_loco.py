"""Leave-One-Centre-Out evaluation pipeline.

Usage:
    python scripts/run_loco.py
    python scripts/run_loco.py --data-dir data/processed --bootstrap 1000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vcko import VCKOAggregator, VCKOBuilder
from vcko.evaluation import calculate_auc, calculate_brier_score, compute_ece

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_loco(
    centre_dfs: dict[str, pd.DataFrame],
    feature_cols: list[str],
    outcome_col: str,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Run Leave-One-Centre-Out protocol.

    Args:
        centre_dfs: Mapping from centre_id to DataFrame.
        feature_cols: Feature column names.
        outcome_col: Outcome column name.
        seed: Random seed.

    Returns:
        Dict mapping held-out centre_id to metrics dict.
    """
    results = {}

    for target_id in centre_dfs:
        logger.info(f"LOCO: holding out {target_id}")

        builder = VCKOBuilder(
            feature_cols=feature_cols,
            outcome_col=outcome_col,
            random_state=seed,
        )

        aggregator = VCKOAggregator()
        for source_id, source_df in centre_dfs.items():
            if source_id == target_id:
                continue
            vcko = builder.fit(source_df, centre_id=source_id)
            aggregator.add(vcko)

        target_df = centre_dfs[target_id]
        y_true = target_df[outcome_col].values
        y_pred = aggregator.predict_proba(target_df)

        auc = calculate_auc(y_true, y_pred)
        brier = calculate_brier_score(y_true, y_pred)
        cal = compute_ece(y_true, y_pred)

        results[target_id] = {
            "auc": auc,
            "brier": brier,
            "ece": cal.ece,
            "n_samples": len(target_df),
        }
        logger.info(f"  {target_id}: AUC={auc:.4f}, Brier={brier:.4f}, ECE={cal.ece:.4f}")

    return results


def bootstrap_loco(
    centre_dfs: dict[str, pd.DataFrame],
    feature_cols: list[str],
    outcome_col: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, list[float]]]:
    """Bootstrap confidence intervals for LOCO AUC."""
    rng = np.random.default_rng(seed)
    all_results: dict[str, list[float]] = {cid: [] for cid in centre_dfs}

    for i in range(n_bootstrap):
        boot_dfs = {}
        for cid, df in centre_dfs.items():
            idx = rng.choice(len(df), size=len(df), replace=True)
            boot_dfs[cid] = df.iloc[idx].reset_index(drop=True)

        results = run_loco(boot_dfs, feature_cols, outcome_col, seed=seed + i)
        for cid, metrics in results.items():
            all_results[cid].append(metrics["auc"])

    ci_results = {}
    for cid, aucs in all_results.items():
        ci_results[cid] = {
            "mean_auc": float(np.mean(aucs)),
            "ci_lower": float(np.percentile(aucs, 2.5)),
            "ci_upper": float(np.percentile(aucs, 97.5)),
        }

    return ci_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VCKO LOCO evaluation")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--output", type=str, default="outputs/loco_results.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    centre_dfs = {}
    for f in sorted(data_dir.glob("centre_*.parquet")):
        cid = f.stem
        centre_dfs[cid] = pd.read_parquet(f)
        logger.info(f"Loaded {cid}: {len(centre_dfs[cid])} samples")

    if not centre_dfs:
        logger.error(f"No centre data found in {data_dir}")
        raise SystemExit(1)

    results = run_loco(
        centre_dfs,
        feature_cols=["age", "AFC", "FSH", "LH"],
        outcome_col="live_birth",
        seed=args.seed,
    )

    if args.bootstrap > 0:
        logger.info(f"Running {args.bootstrap} bootstrap iterations...")
        ci = bootstrap_loco(
            centre_dfs,
            feature_cols=["age", "AFC", "FSH", "LH"],
            outcome_col="live_birth",
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        )
        results["bootstrap_ci"] = ci

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")
