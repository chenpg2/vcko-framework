"""Coefficient direction validation against clinical knowledge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CLINICAL_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "age": {"expected_sign": "negative", "rationale": "Older age lowers IVF success (ESHRE)"},
    "AFC": {
        "expected_sign": "positive",
        "rationale": "Higher AFC indicates better reserve (Bologna 2011)",
    },
    "FSH": {"expected_sign": "negative", "rationale": "Higher FSH indicates diminished reserve"},
    "LH": {"expected_sign": "variable", "rationale": "Effect depends on FSH/LH ratio context"},
}


@dataclass(frozen=True)
class ValidationResult:
    feature: str
    coefficient: float
    expected_sign: str
    actual_sign: str
    match: bool
    rationale: str


def validate_coefficient_direction(
    feature_name: str,
    coefficient: float,
    expectations: dict[str, dict[str, Any]] | None = None,
) -> ValidationResult:
    if expectations is None:
        expectations = CLINICAL_EXPECTATIONS

    info = expectations.get(feature_name)
    if info is None:
        return ValidationResult(
            feature=feature_name,
            coefficient=coefficient,
            expected_sign="unknown",
            actual_sign="positive" if coefficient > 0 else "negative",
            match=True,
            rationale="No clinical expectation defined",
        )

    expected = info["expected_sign"]
    actual = "positive" if coefficient > 0 else ("negative" if coefficient < 0 else "zero")

    if expected == "variable":
        match = True
    elif expected == "negative":
        match = coefficient < 0
    else:
        match = coefficient > 0

    return ValidationResult(
        feature=feature_name,
        coefficient=coefficient,
        expected_sign=expected,
        actual_sign=actual,
        match=match,
        rationale=info["rationale"],
    )
