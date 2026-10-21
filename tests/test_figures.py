"""Tests for the figure layer: the style module's invariants, and the structure of the
two main manuscript figures.

SCOPE
    Style-module tests are pure and always run.  Figure-building tests need the derived
    tables and are skipped without them.  Nothing here re-derives a scientific value -
    that is scripts/99_validate_derived.py's job.  These tests pin the things a figure
    can get wrong SILENTLY: a swapped block order, a dropped panel, a palette colour
    that stops appearing, a mis-keyed decomposition column, non-determinism.

WHY THE PALETTE HAS ITS OWN QUANTITATIVE TEST
    figstyle's docstring makes three measurable claims about the factor colours
    (deuteranope L* separation, hue spread, no reuse across semantic layers).  A claim
    in a comment rots; the same claim as an assertion cannot.  If someone edits a hex
    value, these fail and name the constraint that broke.
"""

from __future__ import annotations

import colorsys
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from jakstat_dosegap import figstyle as fs   # noqa: E402

DERIVED = REPO / "data" / "derived"
REQUIRED = ("perdose.csv", "crossings.csv", "decompositions.csv",
            "ic50_per_profile.csv", "ic50_fold_summary.csv")
has_derived = all((DERIVED / f).exists() for f in REQUIRED)
needs_derived = pytest.mark.skipif(not has_derived, reason="derived tables absent")

# The figure scripts are named for their pipeline ORDER (10_, 11_), which is not a legal
# module name, so they are loaded by path and cached under a short alias.  Importing them
# is side-effect-free: everything happens in main(), which only runs under __main__.
_SCRIPTS = {"fig3": "10_figure_3.py", "fig4": "11_figure_4.py"}
_loaded: dict = {}


def load_script(alias: str):
    """Import a numbered figure script by path.  Cached, so repeated calls in one test
    session do not re-execute the module."""
    import importlib.util
    if alias not in _loaded:
        path = REPO / "scripts" / _SCRIPTS[alias]
        spec = importlib.util.spec_from_file_location(alias, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _loaded[alias] = mod
    return _loaded[alias]


def build_figure(alias: str):
    """Build one figure from the derived tables.  Centralised so a signature change is a
    one-line edit rather than a sweep through every test."""
    m = load_script(alias)
    if alias == "fig3":
        return m.build(m.load_perdose(),
                       m.load_crossings("partial"), m.load_crossings("complete"),
                       m.load_decomposition("partial"), m.load_decomposition("complete"))
    return m.build(m.load_profiles(), m.load_fold_cis(), m.load_fold_medians(),
                   m.load_decomposition("partial"), m.load_decomposition("complete"))


@needs_derived
@pytest.mark.parametrize("alias", ["fig3", "fig4"])
def test_endpoint_labels_use_full_words(alias):
    import re
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    fig = build_figure(alias)
    try:
        labels = [artist.get_text() for artist in fig.findobj(Text)]
        assert any("Arithmetic-mean inhibition" in label for label in labels)
        assert "pY-STAT1" in labels and "pY-STAT3" in labels
        abbreviated = re.compile(r"\b(?:arith[.-]|inhib\.|frac\.|sat\.|pop\.)|fully off")
        assert not [label for label in labels if abbreviated.search(label)]
    finally:
        plt.close(fig)


# ------------------------------------------------------------------ colour helpers
def _rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def _linearise(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# Machado et al. 2009 deuteranomaly matrix at severity 1.0 (full deuteranopia).
_M_DEUT = np.array([[0.367322, 0.860646, -0.227968],
                    [0.280085, 0.672501, 0.047413],
                    [-0.011820, 0.042940, 0.968881]])


def _deut_lstar(h):
    """CIE L* (0-100) of a hex colour as seen by a deuteranope."""
    y = float(np.dot(np.clip(_M_DEUT @ _linearise(_rgb(h)), 0, 1),
                     [0.2126, 0.7152, 0.0722]))
    return 116 * y ** (1 / 3) - 16 if y > 0.008856 else 903.3 * y


def _hue(h):
    r, g, b = _rgb(h)
    return colorsys.rgb_to_hsv(r, g, b)[0] * 360


def _hue_gap(a, b):
    d = abs(_hue(a) - _hue(b))
    return min(d, 360 - d)


# ------------------------------------------------------------------ palette claims
def test_factor_palette_survives_deuteranopia():
    """Every factor pair keeps >= 0.18 in L*/100 under simulated deuteranopia.

    The bars are stacked with no gap, so this separation is the ONLY thing keeping two
    adjacent segments distinguishable for a red-green colour-blind reader.
    """
    for a, b in itertools.combinations(fs.FACTOR, 2):
        sep = abs(_deut_lstar(fs.FACTOR[a]) - _deut_lstar(fs.FACTOR[b])) / 100
        assert sep >= 0.18, f"{a} vs {b}: deuteranope L* separation {sep:.3f}"


def test_factor_palette_hue_spread():
    """The three chromatic factors sit 100-135 deg apart on the hue circle, i.e. they are
    genuinely different hues rather than three shades of one family."""
    chromatic = [k for k in fs.FACTOR_ORDER if k != "Unexplained"]
    for a, b in itertools.combinations(chromatic, 2):
        gap = _hue_gap(fs.FACTOR[a], fs.FACTOR[b])
        assert 100 <= gap <= 135, f"{a} vs {b}: hue gap {gap:.1f} deg"


def test_unexplained_is_achromatic():
    """"Unexplained" must read as "no factor claims this" - a grey, not a fifth hue."""
    r, g, b = _rgb(fs.FACTOR["Unexplained"])
    assert max(abs(r - g), abs(g - b), abs(r - b)) < 0.01


def test_no_colour_reused_across_semantic_layers():
    """State, target and factor palettes must not share a colour: a reader must never
    have to ask whether a violet means "inhibitor factor" or "pSTAT1"."""
    state = set(fs.STATE.values())
    target = set(fs.TARGET.values())
    factor = set(fs.FACTOR.values())
    assert not state & target
    assert not state & factor
    assert not target & factor


def test_locked_hex_values():
    """The palette is pinned to the published figures (verified by pixel-sampling the
    shipped PNGs).  These are not preferences; changing one invalidates the figures."""
    assert fs.STATE == {"partial": "#c0392b", "complete": "#1f6cb4"}
    assert fs.TARGET == {"pSTAT1": "#5e3c99", "pSTAT3": "#e6a419"}
    assert fs.FACTOR == {"Cell line": "#543005", "Inhibitor": "#8c6bb1",
                         "Target": "#66c2a4", "Unexplained": "#e0e0e0"}
    assert fs.MARKER == {"THP1": "o", "U937": "^"}
    assert fs.TAG == "#555555"


def test_factor_order_puts_unexplained_on_top():
    """Stacking order is bottom -> top; the residual belongs at the top of the bar so the
    three attributed factors stack from a common baseline and stay comparable."""
    assert fs.FACTOR_ORDER[-1] == "Unexplained"
    assert set(fs.FACTOR_ORDER) == set(fs.FACTOR)


def test_factor_column_map_covers_every_component():
    assert set(fs.FACTOR_COLUMN) == set(fs.FACTOR)
    assert fs.FACTOR_COLUMN["Target"] == "target"     # CSV spells it lowercase


# ------------------------------------------------------------------ style helpers
def test_apply_style_is_idempotent():
    fs.apply_style()
    first = dict(matplotlib.rcParams)
    fs.apply_style()
    assert matplotlib.rcParams["axes.labelsize"] == first["axes.labelsize"]
    assert matplotlib.rcParams["axes.spines.top"] is False


def test_panel_letter_uppercases_and_sits_outside_axes():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    t = fs.panel_letter(ax, "c")
    assert t.get_text() == "C"
    assert t.get_fontweight() == "bold"
    # Offset is up-and-left of the axes' top-left corner, so x < 0 and y > 1 in axes
    # fraction terms once the point offset is resolved.
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    ab = ax.get_window_extent()
    assert bb.x1 <= ab.x0, "panel letter overlaps the axes horizontally"
    assert bb.y0 >= ab.y1, "panel letter is not above the axes"
    plt.close(fig)


def test_text_overlaps_detects_a_deliberate_collision():
    """Guard the guard: a checker that never fires is worthless."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "collide me", transform=ax.transAxes)
    ax.text(0.5, 0.5, "collide too", transform=ax.transAxes)
    assert fs.text_overlaps(fig), "checker missed an exact overlap"
    plt.close(fig)


# ------------------------------------------------------------------ Figure 3
@needs_derived
def test_figure3_structure_and_block_order():
    import matplotlib.pyplot as plt
    fig = build_figure("fig3")
    try:
        assert len(fig.axes) == 10
        letters = [t.get_text() for ax in fig.axes for t in ax.texts
                   if len(t.get_text()) == 1 and t.get_text().isalpha()]
        assert letters == list("ABCDEFGHIJ")

        # PARTIAL-FIRST is load-bearing: the partial header must be ABOVE the complete
        # one, and each header must carry its own state colour.
        headers = {t.get_text(): (t.get_position()[1], t.get_color())
                   for t in fig.texts if "inhibition" in t.get_text()}
        assert headers["Partial inhibition"][0] > headers["Complete inhibition"][0]
        assert headers["Partial inhibition"][1] == fs.STATE["partial"]
        assert headers["Complete inhibition"][1] == fs.STATE["complete"]
    finally:
        plt.close(fig)


@needs_derived
def test_figure3_panel_kinds_in_order():
    """Per block: two scatters, two 3-category strip plots, one 6-bar stack.  Panel KIND
    is checked structurally (collection/container counts) so a reordered build() fails
    here instead of shipping a figure whose letters no longer match the caption."""
    import matplotlib.pyplot as plt
    fig = build_figure("fig3")
    try:
        for block in (0, 5):
            scat_a, scat_b = fig.axes[block], fig.axes[block + 1]
            strip_a, strip_b = fig.axes[block + 2], fig.axes[block + 3]
            bars = fig.axes[block + 4]
            for ax in (scat_a, scat_b):
                assert len(ax.collections) == 4, "expect 2 cells x 2 readouts"
                assert len(ax.containers) == 0
            for ax in (strip_a, strip_b):
                assert len(ax.get_xticks()) == 3, "one category per threshold"
            assert len(bars.containers) == 4, "one bar container per factor"
            assert len(bars.containers[0]) == 6, "3 thresholds x 2 axes"
    finally:
        plt.close(fig)


@needs_derived
def test_figure3_has_no_text_collisions():
    import matplotlib.pyplot as plt
    fig = build_figure("fig3")
    try:
        assert fs.text_overlaps(fig) == []
    finally:
        plt.close(fig)


@needs_derived
def test_figure3_decomposition_keyed_on_threshold_not_target():
    """Regression: `target` is BOTH the Target variance-share column and a natural name
    for the threshold.  Writing the threshold into it silently replaced the Target share
    with ~0.2-0.8, so the teal segment vanished from the rendered bars while every other
    colour stayed correct.  Pin the loader's columns.
    """
    dec = load_script("fig3").load_decomposition("partial")
    assert "threshold" in dec.columns
    # The Target column must still hold percentages, not threshold values.
    assert dec["target"].min() > 1.0
    shares = dec[["cell", "inhibitor", "target", "unexplained"]].sum(axis=1)
    assert np.allclose(shares, 100.0, atol=0.5), f"shares sum to {shares.tolist()}"


@needs_derived
def test_figure3_crossing_profile_counts():
    """Profiles drop out at high thresholds; the pivot must not invent rows for the
    unobserved experiment x cell-line combinations."""
    m = load_script("fig3")
    expected = {0.2: (24, 24), 0.5: (24, 23), 0.8: (23, 22)}
    for state in ("partial", "complete"):
        cross = m.load_crossings(state)
        for thr, want in expected.items():
            w = cross[thr]
            got = (int(w[f"{state}_at_mean"].notna().sum()),
                   int(w[f"{state}_at_lm"].notna().sum()))
            assert got == want, f"{state} @ {thr}: {got} != {want}"


@needs_derived
def test_figure3_strip_medians_match_published_table():
    """The strip panels' median bars are the published crossing medians.  Cross-checked
    against data/derived/crossing_medians.csv, which the pipeline writes independently."""
    import pandas as pd
    m = load_script("fig3")
    table = pd.read_csv(DERIVED / "crossing_medians.csv")
    for state in ("partial", "complete"):
        cross = m.load_crossings(state)
        for _, row in table.iterrows():
            col = f"{state}_at_mean" if row.axis == "arith" else f"{state}_at_lm"
            got = np.median(cross[row.threshold][col].dropna())
            assert abs(got - row[f"{state}_median"]) < 5e-4, \
                f"{state} {row.axis} @ {row.threshold}: {got} vs {row[f'{state}_median']}"


# ------------------------------------------------------------------ Figure 4
@needs_derived
def test_figure4_structure_and_block_order():
    import matplotlib.pyplot as plt
    fig = build_figure("fig4")
    try:
        assert len(fig.axes) == 8
        letters = [t.get_text() for ax in fig.axes for t in ax.texts
                   if len(t.get_text()) == 1 and t.get_text().isalpha()]
        assert letters == list("ABCDEFGH")
        headers = {t.get_text(): (t.get_position()[1], t.get_color())
                   for t in fig.texts if "inhibition" in t.get_text()}
        assert headers["Partial inhibition"][0] > headers["Complete inhibition"][0]
        assert headers["Partial inhibition"][1] == fs.STATE["partial"]
        assert headers["Complete inhibition"][1] == fs.STATE["complete"]
    finally:
        plt.close(fig)


@needs_derived
def test_figure4_published_medians():
    """The four ratios the manuscript quotes, to the precision the figure prints."""
    med = load_script("fig4").load_fold_medians()
    assert round(med["fold_partial_over_arith"], 1) == 1.7
    assert round(med["fold_partial_over_logmean"], 1) == 0.8
    assert round(med["fold_complete_over_arith"], 1) == 7.4
    assert round(med["fold_complete_over_logmean"], 1) == 2.4


@needs_derived
def test_figure4_cis_bracket_their_medians():
    """A bootstrap CI whisker that does not contain its own median bar would be drawn as
    a whisker floating off the bar - cheap to assert, impossible to miss visually."""
    m = load_script("fig4")
    med, cis = m.load_fold_medians(), m.load_fold_cis()
    for q, (lo, hi) in cis.items():
        assert lo < med[q] < hi, f"{q}: median {med[q]} outside CI ({lo}, {hi})"


@needs_derived
def test_figure4_ratio_panels_are_log_scaled():
    """A dose ratio is multiplicative: 0.5x and 2x are equal-sized opposite effects and
    only a log axis renders them symmetrically about the 1x line."""
    import matplotlib.pyplot as plt
    fig = build_figure("fig4")
    try:
        for idx in (2, 6):        # panels C and G in row-major order
            assert fig.axes[idx].get_yscale() == "log"
        for idx in (0, 1, 4, 5):  # the IC50 scatters are log-log
            assert fig.axes[idx].get_xscale() == "log"
            assert fig.axes[idx].get_yscale() == "log"
    finally:
        plt.close(fig)


@needs_derived
def test_figure4_ratio_panels_show_unity_reference():
    """The 1x line is the panel's entire argument: everything above it is dose the
    population average did not ask for."""
    import matplotlib.pyplot as plt
    fig = build_figure("fig4")
    try:
        for idx in (2, 6):
            ys = [ln.get_ydata()[0] for ln in fig.axes[idx].lines
                  if len(set(ln.get_ydata())) == 1]
            assert any(abs(y - 1.0) < 1e-9 for y in ys), "no unity reference line"
    finally:
        plt.close(fig)


@needs_derived
def test_figure4_has_no_text_collisions():
    import matplotlib.pyplot as plt
    fig = build_figure("fig4")
    try:
        assert fs.text_overlaps(fig) == []
    finally:
        plt.close(fig)


@needs_derived
def test_figure4_decomposition_shares_sum_to_100():
    m = load_script("fig4")
    for state in ("partial", "complete"):
        dd = m.load_decomposition(state)
        for axis, comps in dd.items():
            total = sum(comps.values())
            assert abs(total - 100.0) < 0.5, f"{state}/{axis} sums to {total}"


# ------------------------------------------------------------------ determinism
@needs_derived
@pytest.mark.parametrize("alias", ["fig3", "fig4"])
def test_figures_are_deterministic(alias, tmp_path):
    """Two builds in the same process must be byte-identical.  Jitter is the only random
    element and it is seeded per build, so a difference here means a seed leaked or the
    rng is being shared across builds."""
    import matplotlib.pyplot as plt

    def render(tag):
        fig = build_figure(alias)
        path = tmp_path / f"{alias}_{tag}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path.read_bytes()

    assert render("a") == render("b")


@needs_derived
@pytest.mark.parametrize("alias,expected", [("fig3", (16.5, 7.4)), ("fig4", (13.5, 7.4))])
def test_figure_canvas_size(alias, expected):
    """Figure size is part of the published layout, not a rendering detail: the panel
    aspect ratios and the calibrated font sizes were chosen together at these inches."""
    import matplotlib.pyplot as plt
    fig = build_figure(alias)
    try:
        assert tuple(round(v, 2) for v in fig.get_size_inches()) == expected
        assert fs.DPI == 300
    finally:
        plt.close(fig)
