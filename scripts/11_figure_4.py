#!/usr/bin/env python3
"""Figure 4 - the DOSE cost of a single-cell endpoint, per profile, as an IC50 ratio.

CONTRACT
--------
Inputs  (read-only, produced by scripts/02 and 04):
    data/derived/ic50_per_profile.csv  one row per profile (experiment x inhibitor x
                                       readout), 24 rows.  IC50s in nM for the four
                                       measures (arith-mean, log-mean, partial,
                                       complete) plus the four fold ratios.
    data/derived/ic50_fold_summary.csv median and bootstrap percentile CI of each fold
                                       ratio (BOOTSTRAP_N = 10000, seed 7)
    data/derived/decompositions.csv    type-II ANOVA shares of log10(fold ratio), %
Output:
    figures/Figure_4.png               2 rows x 4 cols, figsize (13.5, 7.4) in, 300 dpi
Invariants:
    * TOP row = PARTIAL (panels A-D); BOTTOM row = COMPLETE (E-H).  Partial-first is
      the same load-bearing editorial order as Figure 3 - see that script's docstring.
    * Published medians this figure must show: partial 1.7x (vs arith-mean) and 0.8x
      (vs log-mean); complete 7.4x and 2.4x.  main() asserts these before drawing, so a
      pipeline regression fails here rather than shipping a wrong number.
    * The ratio panels are LOG-scaled.  A dose ratio is multiplicative: 0.5x and 2x are
      the same size of effect in opposite directions, and only a log axis renders them
      symmetrically about the 1x reference line.  Never switch these to linear.
    * Deterministic: x jitter only, from np.random.default_rng(JITTER_SEED), consumed by
      the two ratio panels in layout order.
    * fig.savefig only - never plt.savefig.

WHY THE TWO ROWS USE DIFFERENT Y-TICKS AND DIFFERENT Y-LABELS
    Partial ratios straddle 1 (a partial-inhibition IC50 can be BELOW the population
    IC50), so that panel is labelled "Dose ratio" and ticked 0.2-10. Complete ratios
    generally lie above 1 and span a wider range, so that panel is
    labelled "Dose ratio (fold)" and ticked 0.5-100. The asymmetry is deliberate:
    forcing a shared scale would compress the partial panel into a line.
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
JITTER_SEED = 5
JITTER_SD = 0.075

# Published IC50 axis window, nM.  Fixed (not data-driven) so the two scatter panels in
# a row share a scale and the identity line has the same slope in both.
IC50_LO, IC50_HI = 1e-1, 1e5

# ic50_per_profile.csv column -> the short key used by the panels.
IC50_COLS = {
    "ic50_mean_inh": "mean",
    "ic50_logmean_inh": "log-mean",
    "ic50_partial": "partial",
    "ic50_complete": "complete",
}
FOLD_COLS = {
    "fold_complete_over_arith": "C_over_M",
    "fold_complete_over_logmean": "C_over_LM",
    "fold_partial_over_arith": "P_over_M",
    "fold_partial_over_logmean": "P_over_LM",
}

# Which ratio columns each row shows, in x order, and the CI keys backing them.
ROW_RATIOS = {
    "partial": (("P_over_M", "P_over_LM"),
                ("fold_partial_over_arith", "fold_partial_over_logmean")),
    "complete": (("C_over_M", "C_over_LM"),
                 ("fold_complete_over_arith", "fold_complete_over_logmean")),
}
RATIO_XLABELS = ["Relative to\narithmetic mean", "Relative to\nlog-mean"]
YTICKS = {"partial": [0.2, 0.5, 1, 2, 5, 10],
          "complete": [0.5, 1, 2, 5, 10, 20, 50, 100]}
YLABEL = {"partial": "Dose ratio", "complete": "Dose ratio (fold)"}


# --- data ------------------------------------------------------------------
def load_profiles() -> pd.DataFrame:
    """ic50_per_profile.csv, renamed to the short panel keys.  24 rows, one per profile.

    Only profiles whose four fits converged carry finite ratios; a NaN ratio is kept as
    a row (so cell/readout grouping stays aligned) and dropped per-panel at plot time.
    """
    df = pd.read_csv(DERIVED / "ic50_per_profile.csv")
    return df.rename(columns={**IC50_COLS, **FOLD_COLS})


def load_fold_cis() -> dict[str, tuple[float, float]]:
    """ic50_fold_summary.csv -> {quantity: (ci_lo, ci_hi)} in the same units as the
    ratios themselves (percentile bootstrap, so the interval is asymmetric on a linear
    scale and roughly symmetric on the log scale the panels use)."""
    s = pd.read_csv(DERIVED / "ic50_fold_summary.csv").set_index("quantity")
    return {q: (float(r.ci_lo), float(r.ci_hi)) for q, r in s.iterrows()}


def load_fold_medians() -> dict[str, float]:
    """ic50_fold_summary.csv -> {quantity: median}.  The published point estimates.

    NOTE the bracket access: the CSV's column is literally named "median", which shadows
    DataFrame.median as an attribute, so `row.median` returns the bound method.
    """
    s = pd.read_csv(DERIVED / "ic50_fold_summary.csv").set_index("quantity")
    return {q: float(v) for q, v in s["median"].items()}


def load_decomposition(state: str) -> dict[str, dict[str, float]]:
    """decompositions.csv -> {axis_label: {component_column: %}} for one state's dose
    penalty.  The decomposition is fitted on log10(ratio), not the raw ratio: variance
    shares of a multiplicative quantity are only meaningful in log space."""
    src = pd.read_csv(DERIVED / "decompositions.csv")
    sub = src[src.quantity == f"dose_penalty_{state}"]
    # Bracket access: the column is named "transform", which shadows DataFrame.transform.
    assert (sub["transform"] == "log10").all(), "expected log10-transformed penalties"
    axis_label = {"arith": "arith-mean", "log": "log-mean"}
    out = {}
    for _, r in sub.iterrows():
        out[axis_label[r.axis]] = {c: float(r[c]) for c in fs.FACTOR_COLUMN.values()}
    return out


# --- panels ----------------------------------------------------------------
def ic_scatter(ax, wt, xcol, ycol, xlabel, ylabel, tag, leg=False):
    """Single-cell-endpoint IC50 vs population IC50, one point per profile, log-log.

    Points above the identity line are profiles that need MORE dose to reach the
    single-cell endpoint than the population axis suggests.  Higher alpha and white
    edges than Figure 3's per-dose clouds because there are only 24 points here and each
    one is a profile the reader may want to identify."""
    ax.plot([IC50_LO, IC50_HI], [IC50_LO, IC50_HI], ls="--", color=fs.NEUTRAL, lw=1,
            zorder=1)
    for cl in fs.CELLS:
        for tg in fs.READOUTS:
            d = wt[(wt.cell == cl) & (wt.readout == tg)]
            ax.scatter(d[xcol], d[ycol], s=22, marker=fs.MARKER[cl],
                       facecolor=fs.TARGET[tg], alpha=0.8, edgecolor="white", lw=0.4,
                       zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(IC50_LO, IC50_HI)
    ax.set_ylim(IC50_LO, IC50_HI)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    # Tag sits bottom-right: the data cloud hugs the identity line, so the lower-right
    # triangle is the one reliably empty corner in both rows.
    ax.text(0.96, 0.06, tag, transform=ax.transAxes, ha="right", fontsize=6,
            color=fs.TAG, fontweight="bold")
    if leg:
        fs.readout_cell_legend(ax, loc="upper left")


def ratio_panel(ax, wt, cols, cis, medians, state, rng):
    """Per-profile dose ratios on a log axis, with median bar + bootstrap CI whisker.

    The 1x reference line is the whole point of the panel: everything above it is dose
    the population average did not ask for.  The printed median uses one decimal because
    that is the precision the manuscript quotes."""
    sc = fs.STATE[state]
    ax.axhline(1, ls="--", color=fs.NEUTRAL, lw=1, zorder=1)
    for i, (col, ci_key) in enumerate(zip(cols, cis)):
        for cl in fs.CELLS:
            for tg in fs.READOUTS:
                m = ((wt.cell == cl) & (wt.readout == tg) & wt[col].notna()).values
                ax.scatter(i + JITTER_SD * rng.standard_normal(m.sum()),
                           wt.loc[m, col].values, marker=fs.MARKER[cl],
                           c=fs.TARGET[tg], s=16, alpha=0.75, edgecolor="white",
                           lw=0.3, zorder=3)
        med = medians[ci_key]
        lo, hi = cis[ci_key]
        ax.plot([i - 0.26, i + 0.26], [med, med], color=sc, lw=2.6, zorder=6,
                solid_capstyle="round")
        ax.plot([i, i], [lo, hi], color=sc, lw=1.2, zorder=5)
        ax.annotate(f"{med:.1f}\u00d7", (i + 0.30, med), va="center", fontsize=6,
                    color=sc, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(RATIO_XLABELS, fontsize=5.6)
    yt = YTICKS[state]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:g}" for v in yt])
    ax.set_xlim(-0.6, len(cols) - 0.25)
    ax.set_ylabel(YLABEL[state])


def fold_decomp(ax, dd, ylabel):
    """Stacked % variance of log10(dose ratio), two bars = the two population axes."""
    labels = ["arith-mean", "log-mean"]
    xs = np.array([0, 1])
    bottom = np.zeros(2)
    for comp in fs.FACTOR_ORDER:
        col = fs.FACTOR_COLUMN[comp]
        vals = np.array([dd[a][col] for a in labels])
        ax.bar(xs, vals, bottom=bottom, width=0.62, color=fs.FACTOR[comp], label=comp,
               edgecolor="white", lw=0.5)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 8:
                ax.text(xs[xi], b + v / 2, f"{v:.0f}", ha="center", va="center",
                        fontsize=5.2,
                        color="#333333" if comp == "Unexplained" else "white",
                        fontweight="bold")
        bottom += vals
    # Same invariant as Figure 3: shares sum to 100 by construction.
    assert np.allclose(bottom, 100.0, atol=0.5), f"stacks sum to {bottom}"
    ax.set_xticks(xs)
    ax.set_xticklabels(RATIO_XLABELS, fontsize=5.4)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, frameon=False,
              fontsize=fs.FS_LEGEND, handlelength=1.0, columnspacing=0.7)


# --- figure ----------------------------------------------------------------
def build(wt, cis, medians, dec_p, dec_c):
    """Assemble the 8-panel figure.  Returns the Figure; writes nothing."""
    rng = np.random.default_rng(JITTER_SEED)
    fig = plt.figure(figsize=(13.5, 7.4))
    gs = fig.add_gridspec(2, 4, hspace=0.58, wspace=0.5)

    # TOP ROW - partial inhibition (panels A-D).
    a = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ic_scatter(a[0], wt, "mean", "partial", "Arithmetic-mean inhibition\nIC50 (nM)",
               "Partial-inhibition IC50 (nM)", "Arithmetic mean", leg=True)
    ic_scatter(a[1], wt, "log-mean", "partial", "Log-mean inhibition\nIC50 (nM)", "", "Log-mean")
    cols, keys = ROW_RATIOS["partial"]
    ratio_panel(a[2], wt, cols, {k: cis[k] for k in keys},
                {k: medians[k] for k in keys}, "partial", rng)
    fold_decomp(a[3], dec_p, "Variance (%)")

    # BOTTOM ROW - complete inhibition (panels E-H).
    b = [fig.add_subplot(gs[1, i]) for i in range(4)]
    ic_scatter(b[0], wt, "mean", "complete", "Arithmetic-mean inhibition\nIC50 (nM)",
               "Complete-inhibition IC50 (nM)", "Arithmetic mean")
    ic_scatter(b[1], wt, "log-mean", "complete", "Log-mean inhibition\nIC50 (nM)", "", "Log-mean")
    cols, keys = ROW_RATIOS["complete"]
    ratio_panel(b[2], wt, cols, {k: cis[k] for k in keys},
                {k: medians[k] for k in keys}, "complete", rng)
    fold_decomp(b[3], dec_c, "Variance (%)")

    for ax, letter in zip(a + b, "ABCDEFGH"):
        fs.panel_letter(ax, letter, case="upper")
    fs.block_header(fig, "Partial inhibition", 0.965, "partial")
    fs.block_header(fig, "Complete inhibition", 0.475, "complete")
    return fig


def main() -> int:
    fs.apply_style()
    wt = load_profiles()
    cis = load_fold_cis()
    medians = load_fold_medians()
    dec_p = load_decomposition("partial")
    dec_c = load_decomposition("complete")

    assert len(wt) == 24, f"profiles: {len(wt)}"
    # The four published medians, to the precision the figure prints.
    expected = {"fold_partial_over_arith": 1.7, "fold_partial_over_logmean": 0.8,
                "fold_complete_over_arith": 7.4, "fold_complete_over_logmean": 2.4}
    for q, want in expected.items():
        got = round(medians[q], 1)
        assert got == want, f"{q}: figure would print {got}x, published {want}x"

    fig = build(wt, cis, medians, dec_p, dec_c)
    collisions = fs.text_overlaps(fig)
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "Figure_4.png"
    fig.savefig(out, dpi=fs.DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out.relative_to(REPO)}")
    print("medians: " + ", ".join(f"{q.split('_over_')[0].replace('fold_','')}"
                                  f"/{q.split('_over_')[1]}={medians[q]:.2f}x"
                                  for q in expected))
    print(f"text collisions: {len(collisions)}")
    for x, y in collisions:
        print(f"  {x!r} <-> {y!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
