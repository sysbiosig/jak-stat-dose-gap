#!/usr/bin/env python3
"""04: crossings + IC50 tables -> variance decompositions and dose-penalty ratios

CONTRACT
    Input  : data/derived/crossings.csv        (from 03_crossings.py)
             data/derived/ic50_per_profile.csv (from 02_fit_ic50.py)
    Output : data/derived/decompositions.csv
                Type-II ANOVA variance shares (percent of total SS, including
                "unexplained") for every decomposed quantity:
                  - complete_fraction / partial_fraction at each (threshold, axis)
                  - each of the four dose penalties, on log10 scale
                Factors: cell + inhibitor + readout (reported as "target").
             data/derived/dose_penalty_ratios.csv
                Per-profile dose penalties (linear and log10) - the per-profile input
                behind the decomposition and the bootstrap CIs.
    Runtime: < 5 s.

WHY log10 for the dose penalties: a penalty is a multiplicative quantity spanning ~2
decades, so a linear ANOVA is dominated by the few largest profiles.  This is not a
cosmetic choice - decomposing the raw ratios instead shifts the cell-line share of the
complete-state penalty by more than 20 percentage points.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jakstat_dosegap import config, decomposition  # noqa: E402

FACTORS = ("cell", "inhibitor", "target", "unexplained")


def main() -> int:
    cross_path = config.DATA_DERIVED / "crossings.csv"
    ic50_path = config.DATA_DERIVED / "ic50_per_profile.csv"
    for p, script in ((cross_path, "03_crossings.py"), (ic50_path, "02_fit_ic50.py")):
        if not p.exists():
            print(f"missing {p}; run scripts/{script} first")
            return 1
    cross = pd.read_csv(cross_path)
    ic50 = pd.read_csv(ic50_path)

    state_dec = decomposition.decomposition_table(cross)
    pen_dec = decomposition.dose_penalty_decomposition(ic50, transform="log10")
    # Align the two blocks on a single schema so one CSV holds every decomposition.
    pen_dec = pen_dec.rename(columns={"quantity": "quantity_raw"})
    pen_dec["quantity"] = "dose_penalty_" + pen_dec["state"]
    # float NaN rather than pd.NA: the dose-penalty block has no threshold, and an
    # all-NA object column would change the concatenated dtype of `threshold`.
    pen_dec["threshold"] = float("nan")
    state_dec = state_dec.assign(quantity_raw=pd.NA, state=pd.NA, transform=pd.NA)
    dec = pd.concat([state_dec, pen_dec], ignore_index=True)
    cols = ["quantity", "quantity_raw", "state", "threshold", "axis", "transform",
            *FACTORS, "n", "total_ss"]
    dec = dec.reindex(columns=[c for c in cols if c in dec.columns])
    out = config.derived("decompositions.csv")
    dec.to_csv(out, index=False)
    print(f"wrote {out.relative_to(config.REPO_ROOT)}: {len(dec)} decompositions\n")

    print("  quantity                     thr  axis   cell  inhib target unexpl   n")
    for _, r in dec.iterrows():
        thr = "  - " if pd.isna(r.get("threshold")) else f"{float(r['threshold']):.1f}"
        print(f"  {str(r['quantity']):28s} {thr} {r['axis']:5s} "
              + " ".join(f"{r[f]:6.1f}" for f in FACTORS)
              + f" {int(r['n']):3d}")

    pen = decomposition.dose_penalty_table(ic50)
    out2 = config.derived("dose_penalty_ratios.csv")
    pen.to_csv(out2, index=False)
    print(f"\nwrote {out2.relative_to(config.REPO_ROOT)}: {len(pen)} per-profile penalties")

    summary = decomposition.fold_ratio_summary(ic50)
    print("  median penalty [95% CI]:")
    for _, r in summary.iterrows():
        print(f"    {r['quantity']:28s} {r['median']:7.2f}x "
              f"[{r['ci_lo']:6.2f}, {r['ci_hi']:6.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
