#!/usr/bin/env python3
"""Figure S2 (selectivity atlas) - pSTAT1-minus-pSTAT3 selectivity across the panel.

RELATION TO PUBLISHED Figure_S2.png
-----------------------------------
`docs/published_figures/Figure_S2.png` is a three-panel IC50 replicate-concordance
scatter: "IC50 shown (nM)" on x against "IC50 replicate (nM)" on y, one panel per
measure (arithmetic mean r=0.93, log-mean r=0.95, complete inhibition r=0.87), with
an identity line.  This script emits the 2 x 6 SELECTIVITY ATLAS from
`the published-figure S2 convention`, saved under its own name
(`Figure_S2_selectivity.png`) so the two are never confused.

The concordance figure is not regenerated here. Its inputs are deposited --
`data/reference/replicate_ic50_shown_vs_replicate.csv` holds the 48 IC50 rows it
plots, and the four replicate experiments are in `data/raw/` (KZ191, KZ203, KZ222,
KZ227). See docs/BIOLOGICAL_REPLICATE.md for the experiment pairing, exclusions
and concordance values.

CONTRACT
--------
Input (read-only, produced by scripts/05_build_figure_tables.py from data/raw/):
    data/derived/figure_tables/Figure_S2_selectivity_delta.csv   266 rows
        experiment, cell_line, inhibitor, dose (nM), metric, pSTAT1, pSTAT3,
        delta_inhibition = pSTAT1 - pSTAT3
Output:
    figures/Figure_S2_selectivity.png   2 x 6, figsize (10.9535, 3.96) in, 300 dpi
                                        = 3286 x 1188 px (the published S2 raster's
                                        dimensions, for side-by-side placement)

WHAT IT SHOWS
    delta > 0  the compound shuts pY-STAT1 down more than pY-STAT3 at that dose
    delta < 0  the reverse
    Only the log-mean (population) and complete (single-cell) measures are plotted -
    the published figure drops mean and partial here, so cause 1 in
    docs/FIGURE_TABLES.md does not enter this figure at all.

COLOUR SEMANTICS
    complete #2171b5, log-mean #238b45.  Same assignment as scripts/13_figure_2.py and
    14_figure_S1.py; deliberately not figstyle.PALETTE.  See docs/FIGURE_TABLES.md.

KNOWN RESIDUAL
    KZ226 (THP-1 / clinical inhibitors) log-mean, up to 0.0107 in delta units
    ("Cause 2").  Every other block reproduces to 1e-16.

INVARIANTS
    * Deterministic; fig.savefig only - never plt.savefig.
    * Log x-axis, so dose-0 rows are dropped rather than clipped.
    * y is symmetric about 0 (YLIM) because the sign is the readable quantity - an
      asymmetric range would make "no selectivity" look like a trend.
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

STYLE = {
    "log-mean": {"color": "#238b45", "label": "log-mean"},
    "complete": {"color": "#2171b5", "label": "complete"},
}
METRIC_ORDER = ("log-mean", "complete")

FIGSIZE = (10.9535, 3.96)   # x 300 dpi = 3286 x 1188 px
DPI = 300
YLIM = (-1.0, 1.0)
MARKER_SIZE = 3.5
LINE_WIDTH = 1.5
FS_COLTITLE = 9.5
FS_LABEL = 8.5
FS_TICK = 7.5
FS_LEGEND = 9


def load() -> pd.DataFrame:
    path = TABLES_DIR / "Figure_S2_selectivity_delta.csv"
    if not path.exists():
        raise SystemExit(f"missing {path.relative_to(REPO)} - "
                         "run scripts/05_build_figure_tables.py first")
    d = pd.read_csv(path)
    # The delta table names the compound column `inhibitor` (already carrying the
    # published capitalisation, e.g. "Baricitinib"); the atlas tables call it
    # `inhibitor_label`.  Rename so ATLAS_INHIBITORS indexes it directly.
    d = d.rename(columns={"inhibitor": "inhibitor_label"})
    return d[d["dose"] > 0]


def main() -> int:
    delta = load()
    FIGURES.mkdir(exist_ok=True)

    fig, axes = plt.subplots(len(figure_tables.ATLAS_CELL_LINES),
                             len(figure_tables.ATLAS_INHIBITORS),
                             figsize=FIGSIZE, sharey=True)
    n_series = 0
    for r, cell in enumerate(figure_tables.ATLAS_CELL_LINES):
        for c, inhibitor in enumerate(figure_tables.ATLAS_INHIBITORS):
            ax = axes[r, c]
            sel = ((delta["cell_line"] == cell)
                   & (delta["inhibitor_label"] == inhibitor))
            # Zero line first: it is the reference the sign is read against.
            ax.axhline(0.0, color="0.6", lw=0.7, zorder=1)
            for metric in METRIC_ORDER:
                st = STYLE[metric]
                g = delta[sel & (delta["metric"] == metric)].sort_values("dose")
                if not len(g):
                    continue
                ax.plot(g["dose"], g["delta_inhibition"], "-o", color=st["color"],
                        lw=LINE_WIDTH, ms=MARKER_SIZE, mew=0.0, zorder=2,
                        label=st["label"])
                n_series += 1

            ax.set_xscale("log")
            ax.set_ylim(*YLIM)
            ax.tick_params(axis="both", labelsize=FS_TICK)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(inhibitor, fontsize=FS_COLTITLE, fontweight="bold")
            if r == len(figure_tables.ATLAS_CELL_LINES) - 1:
                ax.set_xlabel("Dose (nM)", fontsize=FS_LABEL)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if c == 0:
                ax.set_ylabel(f"{cell}\n\u0394 inhibition\n(pY-STAT1 \u2212 pY-STAT3)",
                              fontsize=FS_LABEL)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, lab in zip(handles, labels):
        seen.setdefault(lab, h)
    fig.legend([seen[m] for m in METRIC_ORDER if m in seen],
               [m for m in METRIC_ORDER if m in seen],
               loc="upper center", ncol=2, frameon=False, fontsize=FS_LEGEND,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=3.0)

    fig.subplots_adjust(left=0.078, right=0.994, top=0.845, bottom=0.135,
                        wspace=0.16, hspace=0.22)
    out = FIGURES / "Figure_S2_selectivity.png"
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)

    print(f"wrote {out.relative_to(REPO)}")
    print(f"  {len(delta)} points, {n_series} series over "
          f"{delta.groupby(['cell_line', 'inhibitor_label']).ngroups} panels")
    print("  docs/published_figures/Figure_S2.png is the IC50 replicate scatter; "
          "this script emits the selectivity atlas. See docs/FIGURE_TABLES.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
