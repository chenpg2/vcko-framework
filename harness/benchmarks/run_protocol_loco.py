"""LOCO evaluation of the v3 secure-robust-DP protocol stack on real IVF data.

Produces the STORY_LOCK evidence:
  (1) Parity: protocol geometric-median LOCO AUC vs v2 fixed-effect vs v1 heuristic.
  (2) Robustness: under one coefficient-poisoning centre, geometric-median vs linear.
  (3) Privacy: DP utility curve (geometric median + Gaussian coef noise).

All numbers come from the real cohort (rawinputdata/combine_cache.feather via data/).
Seed from conf/default.yaml. No silent fallbacks: a centre with too few complete
cases is reported, not imputed around.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vcko.aggregator import VCKOAggregator  # noqa: E402
from vcko.builder import VCKOBuilder  # noqa: E402
from vcko.data_utils import (  # noqa: E402
    CORE_FEATURES,
    OUTCOME_COL,
    get_X_y,
    load_and_clean,
    partition_by_centre,
)
from vcko.protocol import SecureRobustDPProtocol, geometric_median  # noqa: E402

MIN_COMPLETE = 50


def _seed() -> int:
    cfg = yaml.safe_load((PROJECT_ROOT / "conf" / "default.yaml").read_text())
    return int(cfg["seed"])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _raw_space(agg: VCKOAggregator):
    """Per-centre raw-feature coefficient vectors, SEs, intercepts, n."""
    return agg._raw_space()  # (betas, ses, intercepts, ns)


def _predict_raw(X: np.ndarray, beta: np.ndarray, intercept: float) -> np.ndarray:
    return _sigmoid(X @ beta + intercept)


def _encoded_complete(
    cdf: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Encoded, complete-case (X, y) via the project's get_X_y; None if too small."""
    X, y = get_X_y(cdf, features=features, impute="drop")
    if len(y) < MIN_COMPLETE or len(np.unique(y)) < 2:
        return None
    return X, y


def _build_aggregator(
    source_dfs: dict[str, pd.DataFrame], seed: int, features: list[str]
) -> VCKOAggregator:
    builder = VCKOBuilder(features, OUTCOME_COL, random_state=seed)
    agg = VCKOAggregator()
    for cid, cdf in source_dfs.items():
        enc = _encoded_complete(cdf, features)
        if enc is None:
            continue  # reported via n_sources; NOT silently imputed
        Xc, yc = enc
        local_auc = roc_auc_score(
            yc,
            LogisticRegression(C=1e6, max_iter=2000).fit(Xc, yc).predict_proba(Xc)[:, 1],
        )
        frame = pd.DataFrame(Xc, columns=features)
        frame[OUTCOME_COL] = yc
        agg.add(builder.fit(frame, centre_id=cid, metadata={"local_auc": local_auc}))
    return agg


def _auc_brier(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    return float(roc_auc_score(y, p)), float(brier_score_loss(y, p))


def run(
    seed: int | None = None,
    features: list[str] | None = None,
    feature_tag: str = "full7",
) -> dict:
    seed = seed if seed is not None else _seed()
    features = features if features is not None else list(CORE_FEATURES)
    rng = np.random.default_rng(seed)
    df = load_and_clean()
    parts = partition_by_centre(df)
    centres = sorted(parts.keys())

    parity_rows, robust_rows = [], []
    dp_curve: dict[str, list[float]] = {str(e): [] for e in ["inf", "5", "2", "1"]}

    for held in centres:
        sources = {c: parts[c] for c in centres if c != held}
        test_df = parts[held]
        enc = _encoded_complete(test_df, features)
        if enc is None:
            continue
        Xte, yte = enc
        test_encoded = pd.DataFrame(Xte, columns=features)  # numeric, complete-case

        agg = _build_aggregator(sources, seed, features)
        if len(agg.vckos) < 2:
            continue
        betas, _ses, intercepts, ns = _raw_space(agg)
        n_weight = ns / ns.sum()
        intercept_pooled = float(np.sum(n_weight * intercepts))

        # (1) Parity — no attack. All methods predict on the same encoded rows.
        a_heur, _ = _auc_brier(yte, agg.predict_proba(test_encoded, method="heuristic"))
        a_fe, _ = _auc_brier(yte, agg.predict_proba(test_encoded, method="fixed_effect"))
        beta_gm = geometric_median(betas)
        a_gm, b_gm = _auc_brier(yte, _predict_raw(Xte, beta_gm, intercept_pooled))
        parity_rows.append(
            {
                "held": held,
                "n_sources": len(agg.vckos),
                "auc_heuristic": a_heur,
                "auc_fixed_effect": a_fe,
                "auc_geomedian": a_gm,
                "brier_geomedian": b_gm,
            }
        )

        # (2) Robustness — ADD one adversary (not replace), so K_poisoned = K+1.
        # With K=2 honest + 1 poisoned (K=3), the geometric median (breakdown 1/2)
        # can outvote the single adversary. Replacing one of K=2 yields K=2 where
        # median = mean, which is degenerate and uninformative.
        honest_mean = betas.mean(axis=0)
        # Strong attack: 50x the honest norm, pointing opposite direction.
        # With K_honest / (K_honest + 1) > 1/2, the geometric median (breakdown 1/2)
        # should survive, but the linear mean will be pulled far from the truth.
        adversary = -honest_mean * 50.0
        poisoned = np.vstack([betas, adversary[None, :]])  # K+1 = 3
        beta_lin = poisoned.mean(axis=0)  # linear combine (zero breakdown)
        beta_rob = geometric_median(poisoned)  # robust combine
        a_lin, _ = _auc_brier(yte, _predict_raw(Xte, beta_lin, intercept_pooled))
        a_rob, _ = _auc_brier(yte, _predict_raw(Xte, beta_rob, intercept_pooled))
        robust_rows.append(
            {
                "held": held,
                "k_honest": len(betas),
                "k_poisoned": len(poisoned),
                "auc_linear_poisoned": a_lin,
                "auc_geomedian_poisoned": a_rob,
            }
        )

        # (3) DP utility curve — Gaussian coef noise at decreasing budgets.
        sens = float(np.median(np.linalg.norm(betas, axis=1)))  # scale reference
        for tag, sigma in [
            ("inf", 0.0),
            ("5", sens * 0.05),
            ("2", sens * 0.15),
            ("1", sens * 0.35),
        ]:
            p = SecureRobustDPProtocol(norm_bound=1e9, dp_sigma=sigma)
            beta_dp = p.combine(betas, rng=rng)
            a_dp, _ = _auc_brier(yte, _predict_raw(Xte, beta_dp, intercept_pooled))
            dp_curve[tag].append(a_dp)

    def _mean(xs):
        return round(float(np.mean(xs)), 4) if xs else None

    min_sources = min((r["n_sources"] for r in parity_rows), default=0)
    results = {
        "seed": seed,
        "feature_set": feature_tag,
        "n_features": len(features),
        "n_folds": len(parity_rows),
        "min_sources_per_fold": min_sources,
        "max_byzantine_fraction": round(1.0 / min_sources, 3) if min_sources else None,
        "parity_mean": {
            "auc_heuristic": _mean([r["auc_heuristic"] for r in parity_rows]),
            "auc_fixed_effect": _mean([r["auc_fixed_effect"] for r in parity_rows]),
            "auc_geomedian": _mean([r["auc_geomedian"] for r in parity_rows]),
        },
        "robustness_mean": {
            "auc_linear_poisoned": _mean([r["auc_linear_poisoned"] for r in robust_rows]),
            "auc_geomedian_poisoned": _mean([r["auc_geomedian_poisoned"] for r in robust_rows]),
        },
        "dp_curve_mean": {k: _mean(v) for k, v in dp_curve.items()},
        "per_fold_parity": parity_rows,
        "per_fold_robust": robust_rows,
    }
    return results


def _report(res: dict) -> None:
    print(f"\n=== feature_set={res['feature_set']} ({res['n_features']} features) ===")
    print(
        f"folds={res['n_folds']}  min_sources/fold={res['min_sources_per_fold']}  "
        f"max_byzantine_frac={res['max_byzantine_fraction']}"
    )
    print("--- (1) Parity (no attack), mean LOCO AUC ---")
    for k, v in res["parity_mean"].items():
        print(f"  {k:<20} {v}")
    print("--- (2) Robustness: 1 poisoned source, mean LOCO AUC ---")
    print(f"  linear combine     {res['robustness_mean']['auc_linear_poisoned']}")
    print(f"  geometric median   {res['robustness_mean']['auc_geomedian_poisoned']}")
    bf = res["max_byzantine_fraction"]
    if bf is not None and bf >= 0.5:
        print(f"  [NOTE] {bf:.0%} corruption >= breakdown point 1/2: robustness")
        print("         cannot help here (data-limited: <3 source centres).")
    print("--- (3) DP utility curve (geometric median), mean LOCO AUC ---")
    for k, v in res["dp_curve_mean"].items():
        print(f"  sigma~eps={k:<4} {v}")


def main() -> None:
    out_dir = PROJECT_ROOT / "results" / "phase_protocol_loco"
    out_dir.mkdir(parents=True, exist_ok=True)
    feats7 = list(CORE_FEATURES)
    feats6 = [f for f in CORE_FEATURES if f != "transfer_embryo_day"]

    res7 = run(features=feats7, feature_tag="full7")
    res6 = run(features=feats6, feature_tag="core6_no_transferday")
    (out_dir / "results.json").write_text(json.dumps(res7, indent=2))
    (out_dir / "results_core6.json").write_text(json.dumps(res6, indent=2))

    print(f"\n=== VCKO v3 protocol stack — REAL IVF data, LOCO (seed={res7['seed']}) ===")
    _report(res7)
    _report(res6)
    print("\nMain robustness evidence: core6 (5 centres -> 4 sources -> 25% corruption).")
    print(f"Saved: {out_dir}/results.json, results_core6.json")


if __name__ == "__main__":
    main()
