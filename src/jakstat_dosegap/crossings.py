"""The validated crossing rule: what single-cell state does an "X% inhibited"
population actually contain?

THE RULE (authoritative - reproduces the published values to machine precision)
    For one profile (experiment x inhibitor x readout) and one population axis
    (arithmetic-mean or log-mean inhibition):
      1. Fit a 4PL to dose -> population inhibition for that axis.
      2. Walk the MEASURED doses in ascending order and take the FIRST dose at which
         the FITTED curve is >= the threshold.  That is the crossing dose.
      3. Report the OBSERVED complete and partial fractions AT that dose - the values
         measured in that well, not values read off any fitted curve.
      If the fitted curve never reaches the threshold within the measured dose range,
      the profile contributes nothing at that threshold (this is why the profile count
      falls from 24 at threshold 0.2 to 22-23 at 0.8).

WHY IT IS DEFINED THIS WAY, AND WHY NOT TO "IMPROVE" IT
    The fit is used only to smooth the decision of WHERE the population axis crosses
    the threshold; the reported quantity is always a directly measured fraction at a
    dose that was actually tested.  Interpolating the state fraction between doses, or
    inverting the 4PL for a continuous crossing dose, gives systematically different
    numbers and does NOT reproduce the published values.  The rule is snap-to-measured
    -dose by construction, and that discreteness is part of the published definition.

CONTRACT
--------
crossing_for_profile(profile, threshold, pop_measure) -> dict | None
    dict has: dose, complete, partial, pop_fitted, pop_observed, dose_index,
              n_doses, n_cells_at_dose.  None when the curve never crosses or the
              population fit failed.

crossings_table(perdose, thresholds=..., pop_measures=...) -> pd.DataFrame
    Long format, one row per (profile x threshold x axis):
        experiment, cell, inhibitor, readout, threshold, axis ("arith"/"log"),
        pop_measure, dose, dose_index, complete, partial, pop_fitted, pop_observed
    Both state fractions are read at the same crossing dose, so `complete` and
    `partial` in one row are directly comparable.

crossing_medians(crossings) -> pd.DataFrame
    Median observed complete and partial fraction, and the profile count, per
    (threshold, axis).  These are the published summary numbers.

INVARIANTS
    * Thresholds are applied to the FITTED curve, never to the observed population value.
    * `axis` is the short key from config.AXIS_KEY: mean_inh -> "arith",
      logmean_inh -> "log".
    * A profile can be missing at a high threshold but present at a low one; the
      profile count is therefore threshold- and axis-specific.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .fitting import fit_4pl


def crossing_for_profile(profile: pd.DataFrame, threshold: float,
                         pop_measure: str) -> dict | None:
    """Apply the crossing rule to one profile / threshold / population axis."""
    g = profile.sort_values("dose")
    dose = g["dose"].values
    fit = fit_4pl(dose, g[pop_measure].values)
    if not fit.ok:
        return None

    # fit.dose is the filtered+sorted dose vector actually fitted; index into it, not
    # into the raw profile, so `fitted` and `dose` stay aligned.
    fitted = fit.predict()
    hit = np.where(fitted >= threshold)[0]
    if len(hit) == 0:
        return None
    i = int(hit[0])
    cross_dose = float(fit.dose[i])

    at = g[g["dose"] == cross_dose]
    if at.empty:
        return None
    row = at.iloc[0]
    return {
        "dose": cross_dose,
        "dose_index": i,
        "n_doses": int(len(fit.dose)),
        "complete": float(row["complete"]),
        "partial": float(row["partial"]),
        "pop_fitted": float(fitted[i]),
        "pop_observed": float(row[pop_measure]),
        "n_cells_at_dose": int(row["n"]),
    }


def crossings_table(perdose: pd.DataFrame,
                    thresholds: tuple[float, ...] = config.CROSSING_THRESHOLDS,
                    pop_measures: tuple[str, ...] = config.POP_MEASURES
                    ) -> pd.DataFrame:
    """Per-dose table -> long crossings table.  See module docstring."""
    rows = []
    for (exp_id, inhib, readout), g in perdose.groupby(
            ["experiment", "inhibitor", "readout"], sort=True):
        for pop_measure in pop_measures:
            for threshold in thresholds:
                res = crossing_for_profile(g, threshold, pop_measure)
                if res is None:
                    continue
                rows.append({
                    "experiment": exp_id,
                    "cell": g["cell"].iloc[0],
                    "inhibitor": inhib,
                    "readout": readout,
                    "threshold": float(threshold),
                    "axis": config.AXIS_KEY[pop_measure],
                    "pop_measure": pop_measure,
                    **res,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["threshold", "axis", "experiment", "inhibitor", "readout"]
        ).reset_index(drop=True)
    return out


def crossing_medians(crossings: pd.DataFrame) -> pd.DataFrame:
    """Median complete/partial fraction and profile count per (threshold, axis)."""
    return (crossings
            .groupby(["threshold", "axis"], sort=True)
            .agg(complete_median=("complete", "median"),
                 partial_median=("partial", "median"),
                 n_profiles=("complete", "size"))
            .reset_index())
