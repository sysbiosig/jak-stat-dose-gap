"""CONTRACT
Purpose: quantify how well each plate's two reference distributions separate, and
         detect the case where they INVERT so that the partial and complete bounds
         collapse onto the same value.

WHY this exists: in Figure S1, three profiles (KZ223 / U937 / pSTAT3, all three
         inhibitors) appear to be MISSING their partial-inhibition curve.  They are
         not missing.  The partial bound is max(complete_bound, 5th pct of the
         saturated cells); on that plate the saturated 5th percentile falls BELOW
         the unstimulated 95th percentile, the clamp in metrics.reference_bounds
         fires, and partial == complete at every dose.  The partial curve is drawn
         exactly underneath the complete curve.  This script is the evidence for
         that explanation and the guard against it changing silently.

Input  : data/raw/<EXPID>/ShrinkedNuclei.csv  (read-only)
Output : data/derived/threshold_separation_diagnostic.csv - one row per
         (experiment, readout): reference-well counts, the stimulation fold range,
         the % of saturated cells below the unstimulated 95th percentile, the
         separation AUC, and the clamp_fires flag.
Runtime: ~6 s.

Usage:  python scripts/07_threshold_separation.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jakstat_dosegap import config, io


def gated(frame, well_series, wells, readout):
    """Gated intensity vector for one well set and one readout."""
    x = pd.to_numeric(frame.loc[well_series.isin(wells), readout], errors="coerce")
    x = x[(x > config.SIGNAL_MIN) & (x < config.SIGNAL_MAX)]
    return x.dropna().values


def separation_auc(sat, unstim, seed=7, cap=4000):
    """P(a random saturated cell exceeds a random unstimulated cell).

    Subsampled because the full outer product is O(n_sat * n_unstim); cap=4000
    keeps the estimate within ~0.005 of the exact value at this sample size.
    """
    rng = np.random.default_rng(seed)
    a = rng.choice(sat, size=min(len(sat), cap), replace=False)
    b = rng.choice(unstim, size=min(len(unstim), cap), replace=False)
    return float(np.mean(a[:, None] > b[None, :]))


def build():
    rows = []
    for exp_id in config.PANEL_EXPERIMENTS:
        frame = io.load_experiment(exp_id)
        unstim_wells, sat_wells = io.reference_wells(exp_id)
        well = frame["well"].astype(str).str.strip()
        cell = config.CELL[exp_id]
        for readout in config.READOUTS:
            u = gated(frame, well, unstim_wells, readout)
            s = gated(frame, well, sat_wells, readout)
            p95 = float(np.percentile(u, 95))
            p05 = float(np.percentile(s, 5))
            # a well counts as contributing only if it survives gating for THIS readout
            u_ids = sorted(w for w in unstim_wells
                           if len(gated(frame, well, {w}, readout)) > 0)
            s_ids = sorted(w for w in sat_wells
                           if len(gated(frame, well, {w}, readout)) > 0)
            rows.append({
                "experiment": exp_id,
                "cell": cell,
                "readout": readout,
                "unstim_wells": len(u_ids),
                "sat_wells": len(s_ids),
                "unstim_well_ids": " ".join(u_ids),
                "unstim_p95": p95,
                "sat_p05": p05,
                "median_fold": float(np.median(s) / np.median(u)),
                "sat_below_unstim_p95_pct": 100.0 * float(np.mean(s < p95)),
                "auc_sat_gt_unstim": separation_auc(s, u),
                # the clamp condition, computed rather than hardcoded
                "clamp_fires": bool(p05 <= p95),
            })
    return pd.DataFrame(rows).sort_values(
        "sat_below_unstim_p95_pct", ascending=False).reset_index(drop=True)


def main():
    out = build()
    dest = config.derived("threshold_separation_diagnostic.csv")
    out.round(4).to_csv(dest, index=False)
    print(out[["experiment", "cell", "readout", "unstim_wells", "sat_wells",
               "median_fold", "sat_below_unstim_p95_pct", "clamp_fires"]]
          .to_string(index=False))
    clamped = out[out.clamp_fires]
    print(f"\nprofiles whose partial bound collapses onto complete: {len(clamped)} "
          f"of {len(out)} (experiment, readout) cells")
    for _, r in clamped.iterrows():
        print(f"  {r.experiment} / {r.cell} / {r.readout}: "
              f"sat p05 {r.sat_p05:.2f} <= unstim p95 {r.unstim_p95:.2f}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
