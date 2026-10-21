#!/usr/bin/env python3
"""03: data/derived/perdose.csv -> crossings table + medians

CONTRACT
    Input  : data/derived/perdose.csv (from 01_build_perdose.py)
    Output : data/derived/crossings.csv
                long format, one row per (profile x threshold x population axis), for
                thresholds 0.2/0.5/0.8 and both axes (arithmetic-mean, log-mean).
                Each row carries BOTH observed state fractions (complete and partial)
                read at the same crossing dose.
             data/derived/crossing_medians.csv
                median complete and partial fraction plus profile count per
                (threshold, axis) - the published summary numbers.
    Runtime: < 5 s.

THE RULE (see crossings.py for the full rationale): fit a 4PL to dose -> population
inhibition, take the FIRST MEASURED dose at which the FITTED curve reaches the
threshold, and report the OBSERVED state fractions at that dose.  Profiles whose fitted
curve never reaches the threshold within the tested dose range drop out, so the profile
count falls with increasing threshold (24 -> 22-23).  That attrition is expected and is
itself a reported quantity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jakstat_dosegap import config, crossings  # noqa: E402


def main() -> int:
    perdose_path = config.DATA_DERIVED / "perdose.csv"
    if not perdose_path.exists():
        print(f"missing {perdose_path}; run scripts/01_build_perdose.py first")
        return 1
    perdose = pd.read_csv(perdose_path)

    cross = crossings.crossings_table(perdose)
    out = config.derived("crossings.csv")
    cross.to_csv(out, index=False)
    print(f"wrote {out.relative_to(config.REPO_ROOT)}: {len(cross)} rows")

    med = crossings.crossing_medians(cross)
    out2 = config.derived("crossing_medians.csv")
    med.to_csv(out2, index=False)
    print(f"wrote {out2.relative_to(config.REPO_ROOT)}\n")
    print("  thr  axis   median complete   median partial   n profiles")
    for _, r in med.iterrows():
        print(f"  {r['threshold']:.1f}  {r['axis']:5s} "
              f"{r['complete_median']:14.3f} {r['partial_median']:16.3f} "
              f"{int(r['n_profiles']):12d}")

    n_expected = len(config.CROSSING_THRESHOLDS) * len(config.POP_MEASURES)
    if len(med) != n_expected:
        print(f"WARNING: expected {n_expected} (threshold, axis) groups, got {len(med)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
