"""Negative controls for the v3 protocol-stack experiment.

A result that survives a destroyed signal is an artifact. These controls must FAIL
(produce chance-level output) for the real result to be trustworthy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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

CORE6 = [f for f in CORE_FEATURES if f != "transfer_embryo_day"]


def check_shuffled_labels(seed: int = 42, tol: float = 0.08) -> dict[str, object]:
    """Negative control: shuffle live_birth within each source centre.

    If the LOCO geometric-median AUC is still well above 0.5 with shuffled labels,
    the pipeline has leakage or an artifact. The control PASSES when held-out AUC is
    within ``tol`` of 0.5.

    Raises AssertionError if the shuffled-label AUC exceeds 0.5 + tol.
    """
    from vcko.protocol import geometric_median

    rng = np.random.default_rng(seed)
    df = load_and_clean()
    parts = partition_by_centre(df)
    centres = sorted(parts.keys())
    held = centres[0]
    sources = {c: parts[c] for c in centres if c != held}

    builder = VCKOBuilder(CORE6, OUTCOME_COL, random_state=seed)
    agg = VCKOAggregator()
    for cid, cdf in sources.items():
        X, y = get_X_y(cdf, features=CORE6, impute="drop")
        if len(y) < 50 or len(np.unique(y)) < 2:
            continue
        y_shuffled = rng.permutation(y)  # destroy the label-feature link
        frame = pd.DataFrame(X, columns=CORE6)
        frame[OUTCOME_COL] = y_shuffled
        agg.add(builder.fit(frame, centre_id=cid, metadata={"local_auc": 0.5}))

    Xte, yte = get_X_y(parts[held], features=CORE6, impute="drop")
    betas, _, intercepts, ns = agg._raw_space()
    beta = geometric_median(betas)
    b0 = float(np.sum((ns / ns.sum()) * intercepts))
    logits = Xte @ beta + b0
    p = np.where(logits >= 0, 1 / (1 + np.exp(-logits)), np.exp(logits) / (1 + np.exp(logits)))
    auc = float(roc_auc_score(yte, p))

    assert abs(auc - 0.5) <= tol, (
        f"FAIL [negative control]: shuffled-label AUC={auc:.4f} not near 0.5 "
        f"(tol={tol}); the pipeline has leakage or an artifact."
    )
    return {"shuffled_label_auc": round(auc, 4), "tol": tol, "held_out": held}


if __name__ == "__main__":
    print("Shuffled-label negative control:", check_shuffled_labels())
