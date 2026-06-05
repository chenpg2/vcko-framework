"""Phase registry for the VCKO v3 protocol-stack experiment.

A phase bundles: its data contract, its negative control, its run, and its
benchmark metrics. `run_all` executes contract -> sanity -> run and prints the
evidence, exiting non-zero on any failure so drift is caught in CI.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY: dict[str, Phase] = {}


class Phase:
    name: str = "base"

    def contract(self) -> dict:
        raise NotImplementedError

    def sanity(self) -> dict:
        raise NotImplementedError

    def run(self) -> dict:
        raise NotImplementedError


def register_phase(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        cls.name = name  # type: ignore[attr-defined]
        _REGISTRY[name] = cls()  # type: ignore[call-arg]
        return cls

    return deco


@register_phase("phase_protocol_loco")
class PhaseProtocolLOCO(Phase):
    """v3 secure-robust-DP protocol stack on the real IVF cohort, LOCO."""

    def contract(self) -> dict:
        from harness.contracts.protocol_data import check_protocol_inputs

        return check_protocol_inputs()

    def sanity(self) -> dict:
        from harness.sanity.protocol_sanity import check_shuffled_labels

        return check_shuffled_labels(seed=42)

    def run(self) -> dict:
        from harness.benchmarks.run_protocol_loco import run

        feats6 = ["age_w", "AF", "base_FSH", "base_LH", "egg_num", "transfer_embryo_num"]
        res7 = run(feature_tag="full7")
        res6 = run(features=feats6, feature_tag="core6_no_transferday")
        out = PROJECT_ROOT / "results" / "phase_protocol_loco"
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(res7, indent=2))
        (out / "results_core6.json").write_text(json.dumps(res6, indent=2))
        return {"full7": res7, "core6": res6}


def run_all() -> int:
    rc = 0
    for name, phase in _REGISTRY.items():
        print(f"\n{'=' * 60}\nPHASE: {name}\n{'=' * 60}")
        try:
            print("[contract]", phase.contract().get("rows", "?"), "rows verified")
            print("[sanity]  ", phase.sanity())
            res = phase.run()
            core6 = res["core6"]
            print(
                "[run] core6 robustness: linear",
                core6["robustness_mean"]["auc_linear_poisoned"],
                "-> geomedian",
                core6["robustness_mean"]["auc_geomedian_poisoned"],
            )
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {name}: {e}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(run_all())
