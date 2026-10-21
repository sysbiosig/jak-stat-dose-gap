#!/usr/bin/env python3
"""Figure 2 - the four inhibition measures diverge along one dose-response.

CONTRACT
--------
Input (read-only, produced by scripts/05_build_figure_tables.py from data/raw/):
    data/derived/figure_tables/Figure_2_observed_points.csv   176 rows
        panel A-D, inhibitor, readout, metric, dose (nM), inhibition, data_type
    data/derived/figure_tables/Figure_2_fitted_curves.csv     3200 rows
        same keys + 200 log-spaced dose samples per (panel, metric) with the published-figure 4PL
Output:
    figures/Figure_2.png    2 x 2, figsize (10, 8) in, 300 dpi

PANEL MAP  (all KZ214 / THP-1; see figure_tables.FIG2_PANELS)
    A  solcitinib   x  pY-STAT1        C  cerdulatinib x  pY-STAT1
    B  solcitinib   x  pY-STAT3        D  cerdulatinib x  pY-STAT3

WHY FOUR CURVES PER PANEL
    Two POPULATION axes (arithmetic mean, log-mean of intensity) and two SINGLE-CELL
    state fractions (partial, complete) of the same cells at the same doses.  The gap
    between the population curves and the `complete` curve at a fixed dose IS the dose
    gap the paper is about; this figure is where the reader first sees it.

COLOUR SEMANTICS
    complete  #2171b5 solid    partial  #cb181d solid
    mean      #2ca25f dashed   log-mean #238b45 dotted
    Figure 1 states blue = complete / red = partial in on-figure text and this figure
    carries the same assignment; Figures 3 and 4 inherit the SEMANTICS from here while
    using slightly different hexes (#1f6cb4 / #c0392b) from figstyle.PALETTE.
    See docs/FIGURE_TABLES.md.

KNOWN RESIDUAL
    The `mean` curve differs from the published one by up to 0.083 in inhibition units
    at the low-dose end, because the published-figure arithmetic-mean saturated reference could
    not be recovered from raw.  The other three curves reproduce to 1e-16 / 4e-09.
    See docs/FIGURE_TABLES.md, "Cause 1".

INVARIANTS
    * Deterministic: no stochastic element.
    * fig.savefig only - never plt.savefig.
    * Log x-axis, so the dose-0 rows in the input tables have no position and are
      dropped here rather than clipped to the axis edge.
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
# Legend labels are the published figure's own wording - the bare metric name.
STYLE = {
    "mean":     {"color": "#2ca25f", "ls": "--", "label": "mean"},
    "log-mean": {"color": "#238b45", "ls": ":",  "label": "log-mean"},
    "partial":  {"color": "#cb181d", "ls": "-",  "label": "partial"},
    "complete": {"color": "#2171b5", "ls": "-",  "label": "complete"},
}
# Draw order: population axes first, single-cell states on top.  The `complete` curve
# is the figure's point, so nothing overdraws it.
METRIC_ORDER = ("mean", "log-mean", "partial", "complete")

FIGSIZE = (10.0, 8.0)
DPI = 300
YLIM = (-0.04, 1.06)
MARKER = "o"
MARKER_SIZE = 6.0
LINE_WIDTH = 2.2
FS_PANEL = 17
FS_LABEL = 14
FS_TICK = 12
FS_TITLE = 15
FS_LEGEND = 12


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Observed points and fitted curves, dose > 0 only (the x-axis is logarithmic)."""
    obs_p = TABLES_DIR / "Figure_2_observed_points.csv"
    fit_p = TABLES_DIR / "Figure_2_fitted_curves.csv"
    for p in (obs_p, fit_p):
        if not p.exists():
            raise SystemExit(f"missing {p.relative_to(REPO)} - "
                             "run scripts/05_build_figure_tables.py first")
    obs = pd.read_csv(obs_p)
    fit = pd.read_csv(fit_p)
    return obs[obs["dose"] > 0], fit[fit["dose"] > 0]


def draw_panel(ax, panel: str, obs: pd.DataFrame, fit: pd.DataFrame,
               inhibitor: str, readout: str) -> None:
    """One (inhibitor x readout) dose-response with all four measures."""
    for metric in METRIC_ORDER:
        st = STYLE[metric]
        o = obs[(obs["panel"] == panel) & (obs["metric"] == metric)].sort_values("dose")
        f = fit[(fit["panel"] == panel) & (fit["metric"] == metric)].sort_values("dose")
        ax.plot(f["dose"], f["inhibition"], color=st["color"], ls=st["ls"],
                lw=LINE_WIDTH, zorder=2, label=st["label"])
        ax.plot(o["dose"], o["inhibition"], MARKER, color=st["color"],
                ms=MARKER_SIZE, mew=0.0, zorder=3)

    ax.set_xscale("log")
    ax.set_ylim(*YLIM)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"{inhibitor} / {figure_tables.READOUT_LABEL[readout]}",
                 fontsize=FS_TITLE, fontweight="bold")
    ax.text(-0.15, 1.10, panel, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight="bold", va="top", ha="left")
    # PER-PANEL legend, as published.  Four panels x four keys is more ink than one
    # shared legend, but the published figure reads each panel standalone and the
    # upper-left corner is empty in all four.
    ax.legend(loc="upper left", frameon=False, fontsize=FS_LEGEND,
              handlelength=2.4, borderpad=0.2, labelspacing=0.35)


def main() -> int:
    obs, fit = load()
    FIGURES.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    for ax, (panel, _exp, col, readout) in zip(axes.ravel(), figure_tables.FIG2_PANELS):
        inhibitor = figure_tables.COLUMN_COMPOUND["KZ214"][col]
        draw_panel(ax, panel, obs, fit, inhibitor, readout)

    # Every panel carries both axis labels in the published figure - the panels are
    # read standalone, so a shared outer label would leave two of them unlabelled.
    for ax in axes.ravel():
        ax.set_xlabel("Dose (nM)", fontsize=FS_LABEL, fontweight="bold")
        ax.set_ylabel("Inhibition", fontsize=FS_LABEL, fontweight="bold")

    fig.subplots_adjust(left=0.085, right=0.985, top=0.945, bottom=0.075,
                        wspace=0.24, hspace=0.42)
    out = FIGURES / "Figure_2.png"
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)

    print(f"wrote {out.relative_to(REPO)}")
    print(f"  {len(obs)} observed points, {len(fit)} curve samples, "
          f"{obs.groupby(['panel', 'metric']).ngroups} curves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
