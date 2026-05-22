"""Membership Inference Attack evaluation pipeline.

Usage:
    python scripts/run_mia.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vcko import VCKOBuilder
from vcko.privacy import MIAEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_mia_evaluation(
    centre_dfs: dict[str, pd.DataFrame],
    feature_cols: list[str],
    outcome_col: str,
    n_shadow_models: int = 5,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Evaluate MIA risk for each centre's VCKO.

    Args:
        centre_dfs: Mapping from centre_id to DataFrame.
        feature_cols: Feature column names.
        outcome_col: Outcome column name.
        n_shadow_models: Number of shadow models for MIA.
        seed: Random seed.

    Returns:
        Dict mapping centre_id to MIA metrics.
    """
    results = {}
    builder = VCKOBuilder(
        feature_cols=feature_cols,
        outcome_col=outcome_col,
        random_state=seed,
    )
    evaluator = MIAEvaluator(n_shadow_models=n_shadow_models, random_state=seed)

    for centre_id, df in centre_dfs.items():
        logger.info(f"MIA evaluation for {centre_id}")

        n = len(df)
        half = n // 2
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)

        member_df = df.iloc[idx[:half]].reset_index(drop=True)
        nonmember_df = df.iloc[idx[half:]].reset_index(drop=True)

        vcko = builder.fit(member_df, centre_id=centre_id)
        result = evaluator.evaluate(vcko, member_df, nonmember_df)

        results[centre_id] = {
            "mia_auc": result.auc,
            "mia_accuracy": result.accuracy,
            "tpr": result.true_positive_rate,
            "fpr": result.false_positive_rate,
        }
        logger.info(
            f"  {centre_id}: MIA AUC={result.auc:.4f}, "
            f"Acc={result.accuracy:.4f}, TPR={result.true_positive_rate:.4f}"
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VCKO MIA evaluation")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shadow-models", type=int, default=5)
    parser.add_argument("--output", type=str, default="outputs/mia_results.json")
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

    results = run_mia_evaluation(
        centre_dfs,
        feature_cols=["age", "AFC", "FSH", "LH"],
        outcome_col="live_birth",
        n_shadow_models=args.shadow_models,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")
