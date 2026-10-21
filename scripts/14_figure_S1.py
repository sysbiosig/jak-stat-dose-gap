#!/usr/bin/env python3
"""Figure S1 - the full 24-profile dose-response atlas, all four measures.

CONTRACT
--------
Input (read-only, produced by scripts/05_build_figure_tables.py from data/raw/):
    data/derived/figure_tables/Figure_S1_observed_points.csv   1064 rows
        experiment_id, cell_line, inhibitor_label, readout, dose (nM), type, inhibition
    data/derived/figure_tables/Figure_S1_fitted_curves.csv     19200 rows
        same keys + metric, fit_status, 200 log-spaced dose samples per profile
Output:
    figures/Figure_S1.png    4 rows x 6 cols, figsize (14.96, 10.0767) in, 300 dpi
                             = 4488 x 3023 px, matching the reference raster dimensions

GRID
    rows    (cell line x readout), in ROW_ORDER: THP1|pSTAT1, THP1|pSTAT3,
            U937|pSTAT1, U937|pSTAT3
    cols    the six inhibitors in figure_tables.ATLAS_INHIBITORS order - the three original
            compounds first, then the three clinical JAK inhibitors
    24 panels = the 24 profiles the paper's IC50 statistics are computed over.

COLOUR SEMANTICS
    complete #2171b5 solid   partial #cb181d solid
    mean     #2ca25f dashed  log-mean #238b45 dotted
    Same assignment as scripts/13_figure_2.py, and deliberately not figstyle.PALETTE's
    #1f6cb4 / #c0392b. See docs/FIGURE_TABLES.md.

KNOWN RESIDUALS vs the published figure
    * `mean` curves: up to 0.235 in inhibition units (published-figure saturated reference not
      recoverable - docs/FIGURE_TABLES.md "Cause 1").
    * KZ226 (THP-1, the three clinical inhibitors) `log-mean`: up to 8.5e-03, which is
      under a line width at this raster ("Cause 2").
    * `complete` and `partial` reproduce to 1e-16 on every profile.

INVARIANTS
    * Deterministic: no stochastic element.
    * fig.savefig only - never plt.savefig.
    * Log x-axis, so dose-0 rows are dropped rather than clipped.
    * Panels keep a COMMON y-range (YLIM) so a reader can compare inhibition depth
      across compounds; x-ranges are per-panel because the dose grids differ (KZ223
      baricitinib starts at 0.5 nM, everything else at 10 nM).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from jakstat_dosegap import figure_tables  # noqa: E402

TABLES_DIR = REPO / "data" / "derived" / "figure_tables"
FIGURES = REPO / "figures"

# --- published-figure presentation constants (pinned; see docstring) ------------------
STYLE = {
    "mean":     {"color": "#2ca25f", "ls": "--", "label": "mean"},
    "log-mean": {"color": "#238b45", "ls": ":",  "label": "log-mean"},
    "partial":  {"color": "#cb181d", "ls": "-",  "label": "partial"},
    "complete": {"color": "#2171b5", "ls": "-",  "label": "complete"},
}
METRIC_ORDER = ("mean", "log-mean", "partial", "complete")

ROW_ORDER = (("THP1", "pSTAT1"), ("THP1", "pSTAT3"),
             ("U937", "pSTAT1"), ("U937", "pSTAT3"))

FIGSIZE = (14.96, 10.0767)   # x 300 dpi = 4488 x 3023 px, the published raster
DPI = 300
YLIM = (-0.05, 1.08)
MARKER_SIZE = 3.0
LINE_WIDTH = 1.35
FS_COLTITLE = 11
FS_ROWLABEL = 9
FS_LABEL = 9
FS_TICK = 8
FS_LEGEND = 13


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    obs_p = TABLES_DIR / "Figure_S1_observed_points.csv"
    fit_p = TABLES_DIR / "Figure_S1_fitted_curves.csv"
    for p in (obs_p, fit_p):
        if not p.exists():
            raise SystemExit(f"missing {p.relative_to(REPO)} - "
                             "run scripts/05_build_figure_tables.py first")
    obs = pd.read_csv(obs_p)
    fit = pd.read_csv(fit_p)
    # The observed table names the measure column `type`; the curve table calls the
    # same thing `metric`.  Unify so one draw loop serves both.
    obs = obs.rename(columns={"type": "metric"})
    return obs[obs["dose"] > 0], fit[fit["dose"] > 0]


def main() -> int:
    obs, fit = load()
    FIGURES.mkdir(exist_ok=True)

    fig, axes = plt.subplots(len(ROW_ORDER), len(figure_tables.ATLAS_INHIBITORS),
                             figsize=FIGSIZE, sharey=True)
    n_curves = 0
    for r, (cell, readout) in enumerate(ROW_ORDER):
        for c, inhibitor in enumerate(figure_tables.ATLAS_INHIBITORS):
            ax = axes[r, c]
            sel_o = ((obs["cell_line"] == cell) & (obs["readout"] == readout)
                     & (obs["inhibitor_label"] == inhibitor))
            sel_f = ((fit["cell_line"] == cell) & (fit["readout"] == readout)
                     & (fit["inhibitor_label"] == inhibitor))
            for metric in METRIC_ORDER:
                st = STYLE[metric]
                o = obs[sel_o & (obs["metric"] == metric)].sort_values("dose")
                f = fit[sel_f & (fit["metric"] == metric)].sort_values("dose")
                if len(f):
                    ax.plot(f["dose"], f["inhibition"], color=st["color"],
                            ls=st["ls"], lw=LINE_WIDTH, zorder=2, label=st["label"])
                    n_curves += 1
                ax.plot(o["dose"], o["inhibition"], "o", color=st["color"],
                        ms=MARKER_SIZE, mew=0.0, zorder=3)

            ax.set_xscale("log")
            ax.set_ylim(*YLIM)
            ax.tick_params(axis="both", labelsize=FS_TICK)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(inhibitor, fontsize=FS_COLTITLE, fontweight="bold")
            if r == len(ROW_ORDER) - 1:
                ax.set_xlabel("Dose (nM)", fontsize=FS_LABEL)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if c == 0:
                # Row identity and the y quantity in one label: with 24 panels a
                # separate row-header column would cost more space than it earns.
                ax.set_ylabel(f"{cell} | {readout}\nInhibition", fontsize=FS_ROWLABEL)

    # One figure legend above the grid: 24 per-panel legends would obscure the curves.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, lab in zip(handles, labels):
        seen.setdefault(lab, h)
    fig.legend([seen[m] for m in METRIC_ORDER], list(METRIC_ORDER),
               loc="upper center", ncol=4, frameon=False, fontsize=FS_LEGEND,
               bbox_to_anchor=(0.5, 1.0), handlelength=3.0, columnspacing=3.5)

    fig.subplots_adjust(left=0.052, right=0.995, top=0.945, bottom=0.055,
                        wspace=0.14, hspace=0.20)
    out = FIGURES / "Figure_S1.png"
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)

    print(f"wrote {out.relative_to(REPO)}")
    print(f"  {len(obs)} observed points, {n_curves} fitted curves over "
          f"{obs.groupby(['cell_line', 'readout', 'inhibitor_label']).ngroups} profiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
