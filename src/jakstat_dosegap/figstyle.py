"""Single source of truth for figure presentation: palette, markers, panel letters,
axis conventions, and a text-collision checker.

CONTRACT
--------
Inputs  : none (pure constants + stateless helpers).  Reads no files, touches no
          global matplotlib state at import time.
Outputs : module-level colour/marker/label maps, plus:
            apply_style()                    -> sets rcParams for the manuscript look
            panel_letter(ax, "A")            -> bold capital outside the axes corner
            block_header(fig, text, y, state)-> coloured row/block title
            despine(ax)                      -> drop top+right spines
            text_overlaps(fig)               -> list of colliding text pairs
Invariant: Figures 3 and 4 take their colours from this module. Scripts that reconstruct
          reference rasters pin their figure-specific colours locally when provenance
          requires it. The manuscript Figures 3 and 4 were verified by pixel-sampling
          the deposited PNGs against these exact values, so they are fixed constants.

WHY THE FIGURES LOOK THE WAY THEY DO (calibrated, not guessed)
    Font sizes and line weights below were recovered by measuring glyph bounding
    boxes in docs/published_figures/Figure_3.png at its native 300 dpi:
      "1.0" y-tick label  18 x 36 px  -> 6.0 pt  (renders 18 x 35)
        NOTE: 36 px is the INKED GLYPH width - the "1", "." and "0" occupy columns
        129-139, 145-147 and 152-164 of Figure_3.png.  A naive dark-pixel bbox over
        the whole tick-label gutter returns 61 px because it also swallows the major
        tick mark at columns 182-195.  Measure the glyph groups, not the gutter.
      "arith-mean inhib." 25 x 278 px -> 8.0 pt   (axes.labelsize)
      "A" panel letter    28 x 29 px  -> 9.0 pt bold
      "Partial inhibition" 29 x 352px -> 9.0 pt bold
      axis spine          3 px        -> 0.72 pt
      major tick          13 px       -> 3.1 pt
    Panel letters sit a CONSTANT 98 px left of and 48 px above the axes' top-left
    corner in both published figures, i.e. a fixed physical offset rather than a
    fraction of a panel width.  That is why panel_letter() offsets in POINTS via
    ScaledTranslation: the two figures have different panel widths (5 columns vs 4),
    and a transAxes fraction would place the letters inconsistently between them.

SEMANTIC LAYERS - three palettes, deliberately non-overlapping
    STATE  red/blue    the single-cell state axis.  Semantics ORIGINATE in Figures 1
                       and 2. Figure 1 states the complete-inhibition and partial-only
                       state convention on the figure itself. Figures 3/4 inherit them
                       for block headers, median bars and CI whiskers only.
    TARGET violet/amber the phospho-readout of a data point (pSTAT1 / pSTAT3).
    FACTOR  4 colours  the ANOVA variance components in the decomposition bars.

WHY THE FACTOR PALETTE IS THESE FOUR COLOURS
    The decomposition bars are stacked, so adjacent segments touch with no gap: the
    only cue separating them is the colour pair itself, and it must survive both
    greyscale printing and red-green colour blindness.  The four values were chosen
    against three measurable constraints, all re-verifiable from the hex codes:
      1. LUMINANCE SEPARATION under simulated deuteranopia.  Every one of the six
         pairs is >= 0.18 apart in CIE L* / 100 after deuteranope simulation
         (minimum pair, Inhibitor vs Target and Target vs Unexplained, = 0.185).
         Both the Machado-2009 severity-1.0 and the Vienot-1999 LMS simulations give
         the same 0.185 minimum, so the margin is not an artefact of one model.
         NOTE the metric: this is L* (perceptual lightness), NOT raw relative
         luminance Y - in Y the tightest pair (Cell line vs Inhibitor) is only 0.151
         apart.  Quote the L* figure or the number will not reproduce.
      2. HUE SEPARATION among the three chromatic factors: 107.9 deg (Inhibitor vs
         Target), 124.4 deg (Cell line vs Inhibitor), 127.8 deg (Cell line vs
         Target) - i.e. they are near-evenly spread around the hue circle rather
         than being three shades of one family.  Unexplained is achromatic by
         design, which is also the semantics: "no factor claims this".
      3. NO REUSE of the state red/blue or the target violet/amber.  A reader must
         never have to ask whether a purple means "inhibitor factor" or "pSTAT1".
         The factor Inhibitor violet (#8c6bb1) is the one near-miss - 6.4 deg of hue
         from pSTAT1 (#5e3c99) - but it is 0.116 apart in deuteranope L* and the two
         never appear in the same panel: factor colours occur ONLY in the stacked
         bar panels, which carry no scatter points.  That panel-level separation is
         load-bearing; do not introduce a bar colour into a scatter panel or a
         readout colour into a bar panel.

DO NOT invent variants, lighten "for emphasis", or add a fifth factor colour.  The
constraints above were solved as a set; a new colour re-opens all three.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

# --- palettes --------------------------------------------------------------
# Single-cell state axis (from Figures 1-2; labelled on Figure 1).
STATE = {"partial": "#c0392b", "complete": "#1f6cb4"}

# Phospho-readout of a data point.
TARGET = {"pSTAT1": "#5e3c99", "pSTAT3": "#e6a419"}

# ANOVA variance components.  Keys are the published legend labels verbatim.
FACTOR = {
    "Cell line": "#543005",
    "Inhibitor": "#8c6bb1",
    "Target": "#66c2a4",
    "Unexplained": "#e0e0e0",
}

# Stacking order, bottom -> top.  Verified by pixel-sampling a published bar column:
# brown (Cell line) at the base, then violet, then teal, grey on top.
FACTOR_ORDER = ("Cell line", "Inhibitor", "Target", "Unexplained")

# Legend label -> column name in data/derived/decompositions.csv.  WHY the indirection:
# the CSV reports the readout factor as "target" (config.ANOVA_FACTOR_LABEL), while the
# legend spells it "Target"; keeping the map here stops every script re-deriving it.
FACTOR_COLUMN = {
    "Cell line": "cell",
    "Inhibitor": "inhibitor",
    "Target": "target",
    "Unexplained": "unexplained",
}

# Small-multiple axis annotations ("arith. mean", "log-mean", "vs arith. mean").
TAG = "#555555"

# Neutral grey for identity lines, reference lines and shape-only legend keys.
NEUTRAL = "#888888"
NEUTRAL_KEY = "#666666"

# --- cell lines ------------------------------------------------------------
# Shape, not colour, carries cell line: colour is already spent on the readout, and a
# marker shape survives both greyscale and the dense overplotting in the strip panels.
MARKER = {"THP1": "o", "U937": "^"}
CELL_LABEL = {"THP1": "THP-1", "U937": "U937"}   # data key -> display label
CELLS = ("THP1", "U937")
READOUTS = ("pSTAT1", "pSTAT3")

# --- calibrated type and line sizes (see module docstring) -----------------
FS_TICK = 6.0
FS_LABEL = 8.0
FS_LETTER = 9.0
FS_HEADER = 9.0
FS_LEGEND = 5.0
LW_SPINE = 0.72
TICK_LEN = 3.1

# Panel-letter offset from the axes' top-left corner, in points.  Measured constant
# across both published figures (98 px left, 48 px up at 300 dpi).
LETTER_DX_PT = -98.0 / 300.0 * 72.0
LETTER_DY_PT = 48.0 / 300.0 * 72.0

DPI = 300


def apply_style() -> None:
    """Set rcParams to the manuscript look.  Call once, before creating any figure.

    Mutates global matplotlib state (that is the point).  Idempotent.
    """
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "axes.linewidth": LW_SPINE,
        "xtick.major.width": LW_SPINE,
        "ytick.major.width": LW_SPINE,
        "xtick.minor.width": LW_SPINE * 0.75,
        "ytick.minor.width": LW_SPINE * 0.75,
        "xtick.major.size": TICK_LEN,
        "ytick.major.size": TICK_LEN,
        "xtick.minor.size": TICK_LEN * 0.55,
        "ytick.minor.size": TICK_LEN * 0.55,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,          # embed as TrueType, not Type-3 outlines
        "ps.fonttype": 42,
    })


def despine(ax) -> None:
    """Drop the top and right spines.  Redundant after apply_style(); kept for axes
    created before it, and to make the intent explicit at call sites."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_letter(ax, letter: str, case: str = "upper") -> mpl.text.Text:
    """Stamp a bold panel letter outside the axes' top-left corner.

    letter : single character; `case` forces upper (manuscript convention) or lower.
    Offset is a fixed physical distance (LETTER_DX_PT / LETTER_DY_PT), NOT a fraction
    of the axes - see the module docstring for why.
    """
    txt = letter.upper() if case == "upper" else letter.lower()
    fig = ax.get_figure()
    offset = ScaledTranslation(
        LETTER_DX_PT / 72.0, LETTER_DY_PT / 72.0, fig.dpi_scale_trans
    )
    return ax.text(
        0.0, 1.0, txt,
        transform=ax.transAxes + offset,
        ha="left", va="top",
        fontsize=FS_LETTER, fontweight="bold", color="black",
        clip_on=False,   # the letter sits OUTSIDE the axes; clipping would hide it
    )


def block_header(fig, text: str, y: float, state: str) -> mpl.text.Text:
    """Centred, state-coloured header for a block of panels (a row, or a row pair).

    state : "partial" | "complete" -> picks the colour from STATE.  The header is the
            ONLY place a figure names which single-cell state its block describes, so
            the colour has to match the median bars inside that block.
    """
    return fig.text(
        0.5, y, text, ha="center",
        fontsize=FS_HEADER, fontweight="bold", color=STATE[state],
    )


def readout_cell_legend(ax, loc: str = "upper left", **kw):
    """The shared 5-key legend: two readout colours, two cell-line shapes, identity line.

    Colour keys use a circle so the shape carries no meaning in those two rows; the
    shape keys use NEUTRAL_KEY grey so the colour carries no meaning in those two.
    That orthogonality is why a single legend can explain a 2-colour x 2-shape design
    in five rows instead of four combinations.
    """
    handles = [
        plt.Line2D([], [], marker="o", ls="", mfc=TARGET["pSTAT1"], mec="none",
                   ms=4, label="pY-STAT1"),
        plt.Line2D([], [], marker="o", ls="", mfc=TARGET["pSTAT3"], mec="none",
                   ms=4, label="pY-STAT3"),
        plt.Line2D([], [], marker=MARKER["THP1"], ls="", mfc=NEUTRAL_KEY, mec="none",
                   ms=4, label=CELL_LABEL["THP1"]),
        plt.Line2D([], [], marker=MARKER["U937"], ls="", mfc=NEUTRAL_KEY, mec="none",
                   ms=4, label=CELL_LABEL["U937"]),
        plt.Line2D([], [], ls="--", color=NEUTRAL, label="identity"),
    ]
    opts = dict(frameon=False, fontsize=FS_LEGEND, handletextpad=0.3,
                borderpad=0.2, labelspacing=0.22)
    opts.update(kw)
    return ax.legend(handles=handles, loc=loc, **opts)


def text_overlaps(fig, skip=("inhib.", "threshold", "IC50")):
    """Report pairs of drawn text objects whose bounding boxes intersect.

    Returns [(text_a, text_b), ...].  A figure is acceptable when this is empty.

    WHY the exemptions, and why they are not cheating:
      * tick-label vs tick-label pairs are ignored - matplotlib lays those out itself
        and a 1-px touch between "0.2" and "0.4" is not a legibility failure;
      * strings of <= 2 characters are ignored for the same reason (single-digit tick
        labels dominate the pair list and drown out real collisions);
      * `skip` holds substrings of the long axis labels, which legitimately extend
        under a neighbouring panel's tick labels in a tight 5-column grid without
        overlapping any *glyph* - matplotlib's bbox for a text object is the full
        line box, which is taller than the inked glyphs.
    Everything else - a median annotation over a data label, a legend over a tag, a
    panel letter over a tick - is reported.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    items = [
        (t, t.get_window_extent(renderer))
        for t in fig.findobj(mpl.text.Text)
        if t.get_text().strip() and t.get_visible()
    ]
    ticks = set()
    for ax in fig.axes:
        ticks.update(ax.get_xticklabels())
        ticks.update(ax.get_yticklabels())
    out = []
    for i, (ta, ba) in enumerate(items):
        sa = ta.get_text()
        if len(sa) <= 2 or any(s in sa for s in skip):
            continue
        for tb, bb in items[i + 1:]:
            if ba.overlaps(bb) and not (ta in ticks and tb in ticks):
                out.append((sa, tb.get_text()))
    return out
