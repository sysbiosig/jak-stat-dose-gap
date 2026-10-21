#!/usr/bin/env python3
"""02: data/derived/perdose.csv -> IC50 table + dose-penalty summary

CONTRACT
    Input  : data/derived/perdose.csv (from 01_build_perdose.py)
    Output : data/derived/ic50_per_profile.csv
                one row per WT profile (24 expected: 4 experiments x 3 inhibitors
                x 2 readouts), with a fitted IC50 for each of the four measures, the
                fit-success flag and Hill slope, and the four dose-penalty fold-ratios.
             data/derived/ic50_fold_summary.csv
                median + percentile-bootstrap 95% CI for each fold-ratio
                (n=10000, seed=7).  These are the published dose-penalty numbers.
    Runtime: < 5 s.

Reads doses in the plates' own concentration units; IC50s are in those same units and
are NOT comparable between inhibitors, only between measures within a profile.  That is
why the reported quantity is always a within-profile RATIO of IC50s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jakstat_dosegap import config, decomposition, fitting  # noqa: E402


def main() -> int:
    perdose_path = config.DATA_DERIVED / "perdose.csv"
    if not perdose_path.exists():
        print(f"missing {perdose_path}; run scripts/01_build_perdose.py first")
        return 1
    perdose = pd.read_csv(perdose_path)

    ic50 = fitting.ic50_table(perdose)
    out = config.derived("ic50_per_profile.csv")
    ic50.to_csv(out, index=False)

    failed = {m: int((~ic50[f"fitok_{m}"]).sum()) for m in config.ALL_MEASURES}
    print(f"wrote {out.relative_to(config.REPO_ROOT)}: {len(ic50)} profiles")
    print(f"  4PL fit failures per measure: {failed}")

    summary = decomposition.fold_ratio_summary(ic50)
    out2 = config.derived("ic50_fold_summary.csv")
    summary.to_csv(out2, index=False)
    print(f"\nwrote {out2.relative_to(config.REPO_ROOT)}")
    print(f"  dose penalties (median [95% CI], bootstrap "
          f"n={config.BOOTSTRAP_N} seed={config.BOOTSTRAP_SEED}):")
    for _, r in summary.iterrows():
        print(f"    {r['quantity']:28s} {r['median']:7.2f}x "
              f"[{r['ci_lo']:6.2f}, {r['ci_hi']:6.2f}]  n={int(r['n_profiles'])}")

    if len(ic50) != 24:
        print(f"WARNING: expected 24 profiles, got {len(ic50)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
