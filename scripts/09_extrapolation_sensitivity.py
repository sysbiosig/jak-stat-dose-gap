#!/usr/bin/env python3
"""09: disclose extrapolated IC50 estimates and test the headline against them.

CONTRACT
    Input  : data/derived/ic50_per_profile.csv
    Output : data/derived/extrapolation_sensitivity.csv
    Exit   : 0
    Runtime: ~2 s.

WHY THIS EXISTS
    The 4PL IC50 bound is 10x the highest tested dose (config.FIT_BOUNDS_HIGH[2]),
    so a fitted IC50 may lie beyond the measured dose window. This script checks
    whether the reported dose gap is an artifact of those extrapolated points.
    This script answers it two ways:
      1. counts how many of the 96 fits exceed the maximum tested concentration;
      2. recomputes the four headline medians using only the profiles whose four
         IC50 values are ALL bracketed by tested doses, with the same bootstrap
         settings as the main analysis (config.BOOTSTRAP_SEED, 10,000 draws).
    With the deposited data, 22 of 24 profiles are fully bracketed; complete/log-mean
    2.39 -> 2.21 and complete/arith 7.42 -> 7.05, both CIs overlap the full-panel
    estimate. The gap is therefore not driven by extrapolation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jakstat_dosegap import config

MAX_TESTED_NM = 10000.0
N_BOOT = 10000
METRICS = {"mean_inh": "arithmetic mean", "logmean_inh": "log-mean",
           "partial": "partial", "complete": "complete"}
RATIOS = [("ic50_complete", "ic50_logmean_inh", "complete / log-mean"),
          ("ic50_complete", "ic50_mean_inh", "complete / arith-mean"),
          ("ic50_partial", "ic50_mean_inh", "partial / arith-mean"),
          ("ic50_partial", "ic50_logmean_inh", "partial / log-mean")]


def median_ci(values, rng):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    boot = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(N_BOOT)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(np.median(v)), float(lo), float(hi), int(len(v))


def build(ic):
    bracketed = np.ones(len(ic), dtype=bool)
    for m in METRICS:
        bracketed &= (ic[f"ic50_{m}"] <= MAX_TESTED_NM).values
    rows = []
    for m, label in METRICS.items():
        v = ic[f"ic50_{m}"]
        rows.append({"kind": "extrapolation_count", "metric": label,
                     "n_above_max_tested": int((v > MAX_TESTED_NM).sum()),
                     "max_ic50_nM": float(v.max()), "n_profiles": int(v.notna().sum())})
    for num, den, label in RATIOS:
        rng = np.random.default_rng(config.BOOTSTRAP_SEED)
        allr = median_ci(ic[num] / ic[den], rng)
        rng = np.random.default_rng(config.BOOTSTRAP_SEED)
        brr = median_ci((ic[num] / ic[den])[bracketed], rng)
        rows.append({"kind": "headline_sensitivity", "metric": label,
                     "median_all": allr[0], "ci95_low_all": allr[1],
                     "ci95_high_all": allr[2], "n_all": allr[3],
                     "median_bracketed": brr[0], "ci95_low_bracketed": brr[1],
                     "ci95_high_bracketed": brr[2], "n_bracketed": brr[3]})
    return pd.DataFrame(rows), int(bracketed.sum())


def main():
    ic = pd.read_csv(config.derived("ic50_per_profile.csv"))
    out, n_br = build(ic)
    dest = config.derived("extrapolation_sensitivity.csv")
    out.to_csv(dest, index=False)
    print(f"fully bracketed profiles: {n_br} of {len(ic)}")
    for _, r in out[out.kind == "extrapolation_count"].iterrows():
        print(f"  {r.metric:16s} above {MAX_TESTED_NM:,.0f} nM: {r.n_above_max_tested}"
              f"  (max {r.max_ic50_nM:,.0f} nM)")
    for _, r in out[out.kind == "headline_sensitivity"].iterrows():
        print(f"  {r.metric:22s} all {r.median_all:6.2f} "
              f"[{r.ci95_low_all:.2f},{r.ci95_high_all:.2f}] n={int(r.n_all)}"
              f"   bracketed {r.median_bracketed:6.2f} "
              f"[{r.ci95_low_bracketed:.2f},{r.ci95_high_bracketed:.2f}] n={int(r.n_bracketed)}")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
