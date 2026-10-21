"""Type-II ANOVA variance decomposition and bootstrap confidence intervals.

THE QUESTION THIS ANSWERS
    Given a per-profile quantity (a state fraction at a crossing, or a dose penalty),
    how much of its spread across the 24 profiles is attributable to the cell line, to
    the inhibitor, to the phospho-target read, and how much is unexplained?  A large
    unexplained share means the quantity is not a stable property of the reagent - it
    is context-specific, which is the paper's argument.

CONTRACT
--------
variance_shares(df, response, factors=("cell","inhibitor","readout")) -> dict
    Fits the additive OLS model  response ~ C(cell) + C(inhibitor) + C(readout)
    and returns the type-II sum-of-squares share of each factor as a PERCENT of the
    total sum of squares, plus "unexplained" (the residual share).
    Keys use the published factor labels (config.ANOVA_FACTOR_LABEL): the `readout`
    factor is reported as "target".
    Shares sum to 100 by construction: total SS = sum(factor SS) + residual SS.
    WHY type II (not type I): the design is unbalanced (profiles drop out at high
    thresholds), so type-I shares would depend on the order the terms are listed.
    Type II tests each factor against the model containing all the others, making the
    reported share order-independent.
    WHY additive, no interactions: with 24 profiles across 2 cells x 6 inhibitors x 2
    readouts there is one observation per cell, so no interaction term is estimable.
    Any interaction variance therefore lands in "unexplained" - which is the accurate
    place for it given the design.
    Returns all-NaN shares when fewer than len(factors)+2 observations are available.

decomposition_table(crossings) -> pd.DataFrame
    variance_shares for every (threshold, axis, state) combination, long format:
        quantity ("complete_fraction"/"partial_fraction"), threshold, axis,
        cell, inhibitor, target, unexplained, n, total_ss

dose_penalty_table(ic50) -> pd.DataFrame
    Per-profile dose penalties (the four fold-ratios from fitting.FOLD_DEFS) in both
    linear and log10 form, one row per profile x penalty.

dose_penalty_decomposition(ic50, transform="log10") -> pd.DataFrame
    variance_shares of each dose penalty.  transform="log10" decomposes log10(ratio)
    rather than the ratio itself.  WHY log10 by default: a dose penalty is a
    multiplicative quantity spanning about two decades, so its raw scale is strongly
    right-skewed and a linear ANOVA on it would be dominated by the largest few
    profiles.  Additivity of effects is only plausible on the log scale.

bootstrap_median_ci(values, n=10000, seed=7, alpha=0.05) -> (median, lo, hi)
    Percentile bootstrap of the MEDIAN over profiles, resampling profiles with
    replacement.  Fixed seed (config.BOOTSTRAP_SEED) so the published interval is
    exactly reproducible.  NaNs are dropped before resampling.
    WHY the median, not the mean: fold-ratios are right-skewed and a handful of
    profiles with a near-zero denominator IC50 would dominate a mean.

fold_ratio_summary(ic50, ...) -> pd.DataFrame
    Median and bootstrap CI for each of the four fold-ratios; the published
    dose-penalty numbers.

INVARIANTS
    * One observation per profile; nothing is pooled or averaged before the ANOVA.
    * Random state is drawn from a single np.random.default_rng(seed) per call, so
      results depend on the seed only, never on call order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from . import config
from .fitting import FOLD_DEFS


def variance_shares(df: pd.DataFrame, response: str,
                    factors: tuple[str, ...] = config.ANOVA_FACTORS) -> dict:
    """Type-II SS shares (percent of total SS) for `response`.  See module docstring."""
    cols = [response, *factors]
    d = df[cols].dropna()
    labels = [config.ANOVA_FACTOR_LABEL.get(f, f) for f in factors]
    nan_out = {**{lab: np.nan for lab in labels},
               "unexplained": np.nan, "n": int(len(d)), "total_ss": np.nan}
    if len(d) < len(factors) + 2:
        return nan_out
    # Drop factors that are constant in this subset: they carry 0 df and statsmodels
    # would emit a singular design rather than a zero share.
    usable = [f for f in factors if d[f].nunique() > 1]
    if not usable:
        return nan_out

    formula = f"Q('{response}') ~ " + " + ".join(f"C({f})" for f in usable)
    try:
        model = smf.ols(formula, data=d).fit()
        table = anova_lm(model, typ=2)
    except Exception:
        return nan_out

    total_ss = float(table["sum_sq"].sum())
    if not np.isfinite(total_ss) or total_ss <= 0:
        return nan_out

    out: dict[str, float] = {}
    for f, lab in zip(factors, labels):
        key = f"C({f})"
        ss = float(table.loc[key, "sum_sq"]) if key in table.index else 0.0
        out[lab] = 100.0 * ss / total_ss
    out["unexplained"] = 100.0 * float(table.loc["Residual", "sum_sq"]) / total_ss
    out["n"] = int(len(d))
    out["total_ss"] = total_ss
    return out


def decomposition_table(crossings: pd.DataFrame,
                        states: tuple[str, ...] = config.STATE_MEASURES
                        ) -> pd.DataFrame:
    """Variance shares of each state fraction, per threshold and population axis."""
    rows = []
    for (threshold, axis), g in crossings.groupby(["threshold", "axis"], sort=True):
        for state in states:
            rows.append({
                "quantity": f"{state}_fraction",
                "threshold": float(threshold),
                "axis": axis,
                **variance_shares(g, state),
            })
    return pd.DataFrame(rows)


def dose_penalty_table(ic50: pd.DataFrame) -> pd.DataFrame:
    """Per-profile dose penalties, long format, with a log10 column."""
    keep = ["experiment", "cell", "inhibitor", "readout"]
    rows = []
    for name in FOLD_DEFS:
        if name not in ic50.columns:
            continue
        sub = ic50[keep + [name]].rename(columns={name: "penalty"}).copy()
        sub["quantity"] = name
        # numerator state and denominator axis, split out for easy grouping
        num, den = FOLD_DEFS[name]
        sub["state"] = num
        sub["axis"] = config.AXIS_KEY[den]
        sub["log10_penalty"] = np.log10(sub["penalty"].where(sub["penalty"] > 0))
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        out = out[keep + ["quantity", "state", "axis", "penalty", "log10_penalty"]]
    return out.reset_index(drop=True)


def dose_penalty_decomposition(ic50: pd.DataFrame,
                               transform: str = "log10") -> pd.DataFrame:
    """Variance shares of each dose penalty.  transform in {"log10", "linear"}."""
    pen = dose_penalty_table(ic50)
    response = "log10_penalty" if transform == "log10" else "penalty"
    rows = []
    for (quantity, state, axis), g in pen.groupby(
            ["quantity", "state", "axis"], sort=True):
        rows.append({
            "quantity": quantity,
            "state": state,
            "axis": axis,
            "transform": transform,
            **variance_shares(g, response),
        })
    return pd.DataFrame(rows)


def bootstrap_median_ci(values, n: int = config.BOOTSTRAP_N,
                        seed: int = config.BOOTSTRAP_SEED,
                        alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the median.  Returns (median, lo, hi)."""
    v = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(n, len(v)), replace=True)
    meds = np.median(draws, axis=1)
    lo, hi = np.percentile(meds, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(np.median(v)), float(lo), float(hi))


def fold_ratio_summary(ic50: pd.DataFrame, n: int = config.BOOTSTRAP_N,
                       seed: int = config.BOOTSTRAP_SEED) -> pd.DataFrame:
    """Median + bootstrap CI for each of the four dose-penalty fold-ratios."""
    rows = []
    for name in FOLD_DEFS:
        if name not in ic50.columns:
            continue
        med, lo, hi = bootstrap_median_ci(ic50[name], n=n, seed=seed)
        rows.append({"quantity": name, "median": med, "ci_lo": lo, "ci_hi": hi,
                     "n_profiles": int(np.isfinite(ic50[name]).sum()),
                     "bootstrap_n": n, "bootstrap_seed": seed})
    return pd.DataFrame(rows)


__all__ = ["variance_shares", "decomposition_table", "dose_penalty_table",
           "dose_penalty_decomposition", "bootstrap_median_ci", "fold_ratio_summary",
           "sm"]
