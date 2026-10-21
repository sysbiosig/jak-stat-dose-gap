"""Four-parameter logistic dose-response fit and IC50 extraction.

CONTRACT
--------
four_pl(x, bottom, top, ic50, hill) -> y
    y = bottom + (top - bottom) / (1 + (x / ic50) ** hill)
    x    : dose, in the plate's own concentration units (>0)
    y    : inhibition fraction or state fraction, nominally [0,1]
    NOTE the plateau convention, which is easy to get backwards: with hill > 0,
    x -> 0 gives `top` and x -> inf gives `bottom`.  So `top` is the LOW-dose plateau
    and `bottom` is the HIGH-dose plateau.  Inhibition INCREASES with dose, so a
    well-fitted profile has bottom > top (e.g. bottom ~ 1.0, top ~ 0.0) - the opposite
    of what the parameter names suggest.  The hill bound is [0.2, 5.0], i.e. strictly
    positive: the rise is carried by bottom > top, never by a negative exponent.
    This is the reference implementation's parameterisation; do not "fix" the naming or
    re-orient the curve, as either changes the fitted IC50.

fit_4pl(dose, y) -> FitResult
    scipy.optimize.curve_fit, least squares, bounded, maxfev = config.FIT_MAXFEV.
    p0 = [0, 1, median(dose), 1]; bounds scale the IC50 to the measured dose window
    (dose.min()/10 .. dose.max()*10) so an unconverged fit cannot report an IC50
    outside a decade of what was actually tested.
    Returns FitResult(ok=False, ...) on non-convergence instead of raising - a single
    bad profile must not abort a 24-profile sweep.

ic50(fit) -> float
    The fitted `ic50` parameter, i.e. the dose at which the curve is halfway between
    its two plateaus.  This is a HALFWAY-BETWEEN-PLATEAUS potency, not the dose at
    which the response equals 0.5 absolute; for a curve that does not span [0,1] the
    two differ.  Reported as NaN when the fit failed.

fit_profile(profile_df, measures=...) -> dict
    Fits every measure in `measures` for one (experiment, inhibitor, readout) profile
    and returns {f"ic50_{measure}": value, f"fitok_{measure}": bool, ...}.

ic50_table(perdose, measures=...) -> pd.DataFrame
    One row per profile with the four IC50s and the derived fold-ratios:
        fold_complete_over_logmean = ic50_complete / ic50_logmean_inh
        fold_complete_over_arith   = ic50_complete / ic50_mean_inh
        fold_partial_over_arith    = ic50_partial  / ic50_mean_inh
        fold_partial_over_logmean  = ic50_partial  / ic50_logmean_inh
    WHY these four: each is the factor by which the dose needed for a given
    non-stimulated range exceeds the dose an average-based readout would call
    equivalent.  That factor IS the dose gap.

INVARIANTS
    * Fits are per profile; doses are sorted ascending and duplicates are not merged.
    * A profile needs at least 4 distinct doses (4 free parameters) to be fitted.
    * No smoothing, weighting or outlier removal is applied to the dose-response points.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from . import config


def four_pl(x, bottom, top, ic50, hill):
    """4PL curve.  See module docstring for the sign convention."""
    return bottom + (top - bottom) / (1.0 + (np.asarray(x, dtype=float) / ic50) ** hill)


@dataclass
class FitResult:
    """Outcome of one 4PL fit.

    ok      : did curve_fit converge within bounds and maxfev
    params  : (bottom, top, ic50, hill); all NaN when ok is False
    dose    : the sorted dose vector fitted, in plate units
    y       : the response vector fitted
    message : reason for failure, empty when ok
    """

    ok: bool
    params: np.ndarray
    dose: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    message: str = ""

    @property
    def ic50(self) -> float:
        """Fitted IC50 in plate concentration units, NaN when the fit failed."""
        return float(self.params[2]) if self.ok else float("nan")

    def predict(self, x=None) -> np.ndarray:
        """Fitted curve at `x` (default: the fitted doses).  NaN when fit failed."""
        xs = self.dose if x is None else np.asarray(x, dtype=float)
        if not self.ok:
            return np.full(np.shape(xs), np.nan, dtype=float)
        return four_pl(xs, *self.params)


_NAN4 = np.full(4, np.nan)


def fit_4pl(dose, y) -> FitResult:
    """Bounded 4PL least-squares fit of y against dose.  Never raises."""
    x = np.asarray(dose, dtype=float)
    yy = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(yy) & (x > 0)
    x, yy = x[keep], yy[keep]
    order = np.argsort(x)
    x, yy = x[order], yy[order]

    if len(np.unique(x)) < 4:
        return FitResult(False, _NAN4.copy(), x, yy,
                         f"need >=4 distinct doses, got {len(np.unique(x))}")

    lo = list(config.FIT_BOUNDS_LOW)
    hi = list(config.FIT_BOUNDS_HIGH)
    # Scale the IC50 bounds to this profile's own dose window (see module docstring).
    lo[2] = x.min() * config.FIT_BOUNDS_LOW[2]
    hi[2] = x.max() * config.FIT_BOUNDS_HIGH[2]

    try:
        with warnings.catch_warnings():
            # An overflow inside (x/ic50)**hill during a trial step is normal for the
            # optimiser and does not invalidate the returned optimum.
            warnings.simplefilter("ignore")
            params, _ = curve_fit(
                four_pl, x, yy,
                p0=[0.0, 1.0, float(np.median(x)), 1.0],
                bounds=(lo, hi),
                maxfev=config.FIT_MAXFEV,
            )
    except Exception as exc:  # RuntimeError (no convergence), ValueError (bad input)
        return FitResult(False, _NAN4.copy(), x, yy, f"{type(exc).__name__}: {exc}")
    return FitResult(True, np.asarray(params, dtype=float), x, yy)


def ic50(fit: FitResult) -> float:
    """Fitted IC50 of a FitResult; NaN when the fit failed."""
    return fit.ic50


def fit_profile(profile: pd.DataFrame,
                measures: tuple[str, ...] = config.ALL_MEASURES) -> dict:
    """Fit every measure for one profile.  Returns ic50_*/fitok_*/hill_* per measure."""
    g = profile.sort_values("dose")
    out: dict[str, float | bool] = {}
    for measure in measures:
        fit = fit_4pl(g["dose"].values, g[measure].values)
        out[f"ic50_{measure}"] = fit.ic50
        out[f"fitok_{measure}"] = fit.ok
        out[f"hill_{measure}"] = float(fit.params[3]) if fit.ok else float("nan")
    return out


# Numerator/denominator pairs behind the four published fold-ratios.
FOLD_DEFS = {
    "fold_complete_over_logmean": ("complete", "logmean_inh"),
    "fold_complete_over_arith": ("complete", "mean_inh"),
    "fold_partial_over_arith": ("partial", "mean_inh"),
    "fold_partial_over_logmean": ("partial", "logmean_inh"),
}


def ic50_table(perdose: pd.DataFrame,
               measures: tuple[str, ...] = config.ALL_MEASURES) -> pd.DataFrame:
    """Per-dose table -> one row per profile with IC50s and fold-ratios."""
    recs = []
    for (exp_id, inhib, readout), g in perdose.groupby(
            ["experiment", "inhibitor", "readout"], sort=True):
        rec = {
            "experiment": exp_id,
            "cell": g["cell"].iloc[0],
            "inhibitor": inhib,
            "readout": readout,
            "n_doses": int(g["dose"].nunique()),
            "n_cells": int(g["n"].sum()),
        }
        rec.update(fit_profile(g, measures))
        for name, (num, den) in FOLD_DEFS.items():
            a, b = rec.get(f"ic50_{num}"), rec.get(f"ic50_{den}")
            rec[name] = (a / b) if (a is not None and b is not None
                                    and np.isfinite(a) and np.isfinite(b) and b != 0) \
                else float("nan")
        recs.append(rec)
    return pd.DataFrame(recs).sort_values(
        ["experiment", "inhibitor", "readout"]).reset_index(drop=True)
