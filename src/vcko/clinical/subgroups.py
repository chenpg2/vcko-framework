"""Bologna 2011 subgroup classification for IVF patients.

References:
  Ferraretti et al. (2011) ESHRE consensus on poor ovarian response.
"""

from __future__ import annotations

from enum import Enum
from typing import cast

import numpy as np
import pandas as pd


class BolognaSubgroup(Enum):
    POR = "POR"
    NOR = "NOR"
    HYR = "HYR"


def assign_bologna_subgroup(
    age: float | None,
    afc: float | None,
    fsh: float | None = None,
) -> BolognaSubgroup:
    """Simplified Bologna 2011 classification.

    HYR: AFC > 20
    POR: AFC < 5 OR FSH >= 10 OR age >= 40
    NOR: otherwise
    """
    if afc is not None and not (isinstance(afc, float) and np.isnan(afc)):
        if float(afc) > 20:
            return BolognaSubgroup.HYR

    is_poor = False
    if afc is not None and not (isinstance(afc, float) and np.isnan(afc)) and float(afc) < 5:
        is_poor = True
    if fsh is not None and not (isinstance(fsh, float) and np.isnan(fsh)) and float(fsh) >= 10:
        is_poor = True
    if age is not None and not (isinstance(age, float) and np.isnan(age)) and float(age) >= 40:
        is_poor = True

    return BolognaSubgroup.POR if is_poor else BolognaSubgroup.NOR


def add_subgroup_column(
    df: pd.DataFrame,
    age_col: str = "age",
    afc_col: str = "AFC",
    fsh_col: str | None = "FSH",
) -> pd.DataFrame:
    result = df.copy()
    result["subgroup"] = [
        assign_bologna_subgroup(
            age=row.get(age_col),
            afc=row.get(afc_col),
            fsh=row.get(fsh_col) if fsh_col else None,
        ).value
        for _, row in df.iterrows()
    ]
    return cast(pd.DataFrame, result)
