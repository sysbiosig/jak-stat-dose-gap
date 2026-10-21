#!/usr/bin/env python3
"""Figure 3 - what single-cell state a given POPULATION-LEVEL inhibition contains.

CONTRACT
--------
Inputs  (read-only, produced by scripts/01..04):
    data/derived/perdose.csv         one row per (experiment, inhibitor, readout, dose):
                                     mean_inh, logmean_inh in [0,1] (population axes),
                                     complete, partial in [0,1] (single-cell fractions)
    data/derived/crossings.csv       one row per (profile, threshold, axis): the state
                                     fractions observed at the crossing dose
    data/derived/decompositions.csv  type-II ANOVA variance shares, % of total SS
Output:
    figures/Figure_3.png             2 rows x 5 cols, figsize (16.5, 7.4) in, 300 dpi
Invariants:
    * TOP block = PARTIAL inhibition (panels A-E); BOTTOM block = COMPLETE (F-J).
      PARTIAL-FIRST IS LOAD-BEARING and editorial: partial agrees with the population
      average, complete departs from it, so the reader meets the agreement before the
      discrepancy.  Do not reorder to put "complete" first because it is the headline
      result - the argument is built in this order.
    * Deterministic: the ONLY stochastic element is strip-plot x jitter, drawn from a
      single np.random.default_rng(JITTER_SEED) consumed by the four strip panels in
      layout order.  Re-ordering the strip() calls changes the jitter (not the data),
      so keep the call order as written.
    * fig.savefig only - never plt.savefig.

PANEL MAP (per block; the two blocks are identical in structure)
    1st, 2nd : per-dose scatter of state fraction vs population inhibition, on the
               arithmetic-mean axis then the log-mean axis, with an identity line.
               Every measured dose of every profile is a point, so these show the
               state-vs-average relationship without any thresholding.
    3rd, 4th : strip plots of the state fraction AT the dose where the population axis
               first crosses 0.2 / 0.5 / 0.8, again arith-mean then log-mean, with a
               median bar per threshold.  These are the crossing rule's output.
    5th      : stacked variance decomposition of those crossing values, % of total SS
               attributed to cell line / inhibitor / target / unexplained.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from jakstat_dosegap import figstyle as fs   # noqa: E402

DERIVED = REPO / "data" / "derived"
FIGURES = REPO / "figures"
THRESHOLDS = (0.2, 0.5, 0.8)
JITTER_SEED = 3
JITTER_SD = 0.08          # in x-category units; ~1/3 of the 0.22 median-bar half-width
AXIS_LABEL = {"arith": "arith-mean", "log": "log-mean"}


# --- data ------------------------------------------------------------------
def load_perdose() -> pd.DataFrame:
    """perdose.csv as-is.  One row per measured dose; columns used: cell, readout,
    mean_inh, logmean_inh, complete, partial."""
    return pd.read_csv(DERIVED / "perdose.csv")


def load_crossings(state: str) -> dict[float, pd.DataFrame]:
    """crossings.csv -> {threshold: wide frame} for one single-cell state.

    Returns, per threshold, one row per profile (experiment x inhibitor x readout) with
    `<state>_at_mean` and `<state>_at_lm` = the observed state fraction at the dose
    where the arithmetic-mean / log-mean population axis first crossed the threshold.
    A profile whose fitted population curve never reaches the threshold on a given axis
    yields NaN in that column - it is NOT dropped, because the two axes drop out at
    different thresholds (24 profiles at 0.2 falling to 22 at 0.8 on the log axis) and
    the medians must be taken over each axis's own surviving set.

    WHY pivot_table WITHOUT dropna=False: the four key columns are not a full factorial
    (each cell line appears on its own plates), so dropna=False materialises the 96-row
    cross product of experiment x cell and pads it with NaN.  The default drops those
    unobserved combinations and gives the 24 / 24 / 23 profiles the published medians are
    taken over.  Pinned by the row-count assertion in main().
    """
    src = pd.read_csv(DERIVED / "crossings.csv")
    keys = ["experiment", "cell", "inhibitor", "readout"]
    out = {}
    for thr in THRESHOLDS:
        sub = src[src.threshold == thr]
        wide = (sub.pivot_table(index=keys, columns="axis", values=state)
                   .rename(columns={"arith": f"{state}_at_mean", "log": f"{state}_at_lm"})
                   .reset_index())
        wide.columns.name = None
        out[thr] = wide
    return out


def load_decomposition(state: str) -> pd.DataFrame:
    """decompositions.csv -> the six (threshold x axis) rows for one state's crossing
    fractions, with `axis` relabelled to the panel's own axis names.

    WHY NOTHING IS RENAMED TO "target": `target` is already the name of the Target
    variance-share column (config.ANOVA_FACTOR_LABEL maps the readout factor to
    "target").  Writing the threshold into a column of that name silently overwrites the
    share, and the Target segment then gets drawn with a height of 0.2-0.8% instead of
    15-70% - which is exactly the bug the palette pixel-audit caught.  Key the panel on
    `threshold`, never on `target`.
    """
    src = pd.read_csv(DERIVED / "decompositions.csv")
    sub = src[src.quantity == f"{state}_fraction"].copy()
    sub["threshold"] = sub["threshold"].astype(float)
    sub["axis"] = sub["axis"].map({"arith": "arith-mean", "log": "log-mean"})
    return sub


# --- panels ----------------------------------------------------------------
def scatter(ax, dfp, xcol, ycol, xlabel, ylabel, tag, leg=False):
    """Per-dose state fraction vs population inhibition, split by cell line (shape) and
    readout (colour), with the identity line.  Low alpha because all 242 dose points of
    all 24 profiles are drawn: the SHAPE of the point cloud relative to identity is the
    message, not any individual dose."""
    ax.plot([0, 1], [0, 1], ls="--", color=fs.NEUTRAL, lw=1, zorder=1)
    for cl in fs.CELLS:
        for tg in fs.READOUTS:
            d = dfp[(dfp.cell == cl) & (dfp.readout == tg)]
            ax.scatter(d[xcol], d[ycol], s=9, marker=fs.MARKER[cl],
                       facecolor=fs.TARGET[tg], alpha=0.42, edgecolor="none", zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.06)
    if ylabel:
        ax.set_ylabel(ylabel)
    tx, ha = (0.96, "right") if leg else (0.04, "left")
    ax.text(tx, 0.08 if leg else 0.97, tag, transform=ax.transAxes, va="top", ha=ha,
            fontsize=6, color=fs.TAG, fontweight="bold")
    if leg:
        fs.readout_cell_legend(ax, loc="upper left", bbox_to_anchor=(0.02, 0.99))


def strip(ax, wide_by_thr, col, tag, state, ylabel, rng):
    """State fraction at the crossing dose, one x-category per threshold.

    Median bar + printed median in the state colour; the dashed line joins the medians
    so the reader can read the trend across thresholds without re-finding each bar.
    Jitter is x-only, so no data value is ever displaced."""
    sc = fs.STATE[state]
    for ti, thr in enumerate(THRESHOLDS):
        d = wide_by_thr[thr].dropna(subset=[col])
        y = d[col].values
        xx = ti + JITTER_SD * rng.standard_normal(len(y))
        for cl in fs.CELLS:
            for tg in fs.READOUTS:
                m = ((d.cell == cl) & (d.readout == tg)).values
                ax.scatter(xx[m], y[m], marker=fs.MARKER[cl], c=fs.TARGET[tg], s=15,
                           alpha=0.75, edgecolor="white", lw=0.3, zorder=3)
        med = np.median(y)
        ax.plot([ti - 0.22, ti + 0.22], [med, med], color=sc, lw=2.4, zorder=5,
                solid_capstyle="round")
        ax.annotate(f"{med:.2f}", (ti + 0.26, med), va="center", fontsize=5.3,
                    color=sc, fontweight="bold")
    medians = [np.median(wide_by_thr[t][col].dropna()) for t in THRESHOLDS]
    ax.plot(range(len(THRESHOLDS)), medians, color=sc, lw=1.0, ls="--", alpha=0.8,
            zorder=4)
    ax.set_xticks(range(len(THRESHOLDS)))
    ax.set_xticklabels([f"{t:g}" for t in THRESHOLDS])
    ax.set_xlabel("Population inhibition\nthreshold")
    ax.set_ylim(-0.03, 1.10)
    # Right margin is wider than the left so the median annotations have room; without
    # it the 0.8 annotation collides with the spine.
    ax.set_xlim(-0.7, 2.75)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.text(0.5, 0.97, tag, transform=ax.transAxes, va="top", ha="center",
            fontsize=5.8, color=fs.TAG, fontweight="bold")


def decomp(ax, frame, ylabel):
    """Stacked % variance, six bars = 3 thresholds x 2 population axes.

    Bars are grouped in threshold pairs (arith, log) with a gap between groups, so the
    within-threshold axis comparison is adjacent and the across-threshold comparison is
    the coarser visual step.  Segments under 8% carry no number - the label would be
    taller than its segment."""
    labels, xs, x = [], [], 0.0
    for thr in THRESHOLDS:
        for axis in ("arith-mean", "log-mean"):
            labels.append((thr, axis))
            xs.append(x)
            x += 1
        x += 0.5
    xs = np.array(xs)
    bottom = np.zeros(len(labels))
    for comp in fs.FACTOR_ORDER:
        col = fs.FACTOR_COLUMN[comp]
        vals = np.array([
            frame[(frame.threshold == t) & (frame.axis == a)][col].iloc[0]
            for t, a in labels
        ])
        ax.bar(xs, vals, bottom=bottom, width=0.8, color=fs.FACTOR[comp], label=comp,
               edgecolor="white", lw=0.5)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 8:
                ax.text(xs[xi], b + v / 2, f"{v:.0f}", ha="center", va="center",
                        fontsize=4.8,
                        color="#333333" if comp == "Unexplained" else "white",
                        fontweight="bold")
        bottom += vals
    # Shares sum to 100 by construction (total SS = factor SS + residual SS).  A bar that
    # does not reach 100 means a component column was mis-read, not that variance went
    # missing - assert rather than ship a short bar.
    assert np.allclose(bottom, 100.0, atol=0.5), f"stacks sum to {bottom}"
    ax.set_xticks(xs)
    ax.set_xticklabels([
        f"{'Arithmetic mean' if axis == 'arith-mean' else 'Log-mean'}\n{thr:g}"
        for thr, axis in labels
    ], fontsize=5, rotation=55, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2, frameon=False,
              fontsize=fs.FS_LEGEND, handlelength=1.0, columnspacing=0.7)


# --- figure ----------------------------------------------------------------
def build(dfp, cross_p, cross_c, dec_p, dec_c):
    """Assemble the 10-panel figure.  Returns the Figure; writes nothing."""
    rng = np.random.default_rng(JITTER_SEED)
    fig = plt.figure(figsize=(16.5, 7.4))
    gs = fig.add_gridspec(2, 5, hspace=0.72, wspace=0.5)

    # TOP BLOCK - partial inhibition (panels A-E).
    p = [fig.add_subplot(gs[0, i]) for i in range(5)]
    scatter(p[0], dfp, "mean_inh", "partial", "Arithmetic-mean inhibition",
            "Partial-inhibition\noccupancy", "Arithmetic mean", leg=True)
    scatter(p[1], dfp, "logmean_inh", "partial", "Log-mean inhibition", "", "Log-mean")
    strip(p[2], cross_p, "partial_at_mean", "Arithmetic mean", "partial",
          "Partial-inhibition\noccupancy", rng)
    strip(p[3], cross_p, "partial_at_lm", "Log-mean", "partial", "", rng)
    decomp(p[4], dec_p, "Variance (%)")

    # BOTTOM BLOCK - complete inhibition (panels F-J).
    c = [fig.add_subplot(gs[1, i]) for i in range(5)]
    scatter(c[0], dfp, "mean_inh", "complete", "Arithmetic-mean inhibition",
            "Complete-inhibition\noccupancy", "Arithmetic mean")
    scatter(c[1], dfp, "logmean_inh", "complete", "Log-mean inhibition", "", "Log-mean")
    strip(c[2], cross_c, "complete_at_mean", "Arithmetic mean", "complete",
          "Complete-inhibition\noccupancy", rng)
    strip(c[3], cross_c, "complete_at_lm", "Log-mean", "complete", "", rng)
    decomp(c[4], dec_c, "Variance (%)")

    for ax, letter in zip(p + c, "ABCDEFGHIJ"):
        fs.panel_letter(ax, letter, case="upper")
    fs.block_header(fig, "Partial inhibition", 0.955, "partial")
    fs.block_header(fig, "Complete inhibition", 0.470, "complete")
    return fig


def main() -> int:
    fs.apply_style()
    dfp = load_perdose()
    cross_p = load_crossings("partial")
    cross_c = load_crossings("complete")
    dec_p = load_decomposition("partial")
    dec_c = load_decomposition("complete")

    assert len(dfp) == 242, f"perdose rows: {len(dfp)}"
    # Published profile counts per threshold: profiles drop out at high thresholds when
    # the fitted population curve never reaches them inside the measured dose range.
    expected_n = {0.2: (24, 24), 0.5: (24, 23), 0.8: (23, 22)}
    for state, cross in (("partial", cross_p), ("complete", cross_c)):
        for thr, (n_arith, n_log) in expected_n.items():
            w = cross[thr]
            got = (int(w[f"{state}_at_mean"].notna().sum()),
                   int(w[f"{state}_at_lm"].notna().sum()))
            assert got == (n_arith, n_log), \
                f"{state} @ {thr}: arith/log profiles {got}, published {(n_arith, n_log)}"

    fig = build(dfp, cross_p, cross_c, dec_p, dec_c)
    collisions = fs.text_overlaps(fig)
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "Figure_3.png"
    fig.savefig(out, dpi=fs.DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out.relative_to(REPO)}")
    print(f"text collisions: {len(collisions)}")
    for a, b in collisions:
        print(f"  {a!r} <-> {b!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
