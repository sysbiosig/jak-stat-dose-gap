"""Tests for the per-figure source-table path (Figures 1, 2, S1, S2).

CONTRACT
    These tests guard the CLAIMS in docs/FIGURE_TABLES.md, not just the code paths.
    Each one pins a number that document states, so a change in convention that
    silently improves or degrades agreement fails here rather than going unnoticed.

    They read data/derived/figure_tables/ and data/reference/published_tables/ and skip
    (not fail) when the derived tables are absent, so a fresh clone without a pipeline
    run still collects cleanly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))   # package is not installed; matches test_pipeline
TABLES = REPO / "data" / "derived" / "figure_tables"
PUBLISHED_TABLES = REPO / "data" / "reference" / "published_tables"

# The two documented, unreproducible conventions and their measured sizes.  These are
# UPPER bounds: if a future change reduces them, the assertion still passes and the
# provenance doc should be updated.  If it makes them worse, the test fails.
MEAN_RESIDUAL_MAX = 0.24        # published-figure arithmetic-mean saturated reference
KZ226_LOGMEAN_MAX = 9e-3        # KZ226 log-mean saturated well subset, dose > 0
EXACT = 1e-9                    # float64 CSV round-trip


def _need(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"{path.relative_to(REPO)} absent - "
                    "run scripts/05_build_figure_tables.py")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def report() -> pd.DataFrame:
    return _need(TABLES / "rederivation_report.csv")


def test_all_thirteen_source_tables_are_covered(report):
    """Every published source table has a verdict - no file silently unexamined."""
    published = {p.name for p in PUBLISHED_TABLES.glob("*.csv")}
    assert len(published) == 13, f"expected 13 published CSVs, found {len(published)}"
    assert published == set(report["file"]), (
        f"missing verdicts: {published - set(report['file'])}")


def test_no_verdict_claims_a_match_it_cannot_support(report):
    """A row graded 'exact' must actually have max_abs_diff at round-trip precision.

    This guard makes it impossible to relabel a differing file as
    exact without the number moving too.
    """
    exact = report[report["match"] == "exact"]
    assert len(exact) >= 1
    bad = exact[exact["max_abs_diff"] > 1e-6]
    assert bad.empty, f"graded exact but differs: {bad[['file', 'max_abs_diff']]}"


def test_every_residual_has_a_named_root_cause(report):
    """No unexplained differences: a non-exact row must name why."""
    unexplained = report[report["match"].str.startswith("differs")]
    assert unexplained.empty, (
        "rows differing for undiagnosed reasons:\n"
        f"{unexplained[['file', 'match', 'root_cause']]}")


def test_figure1_source_data_reproduces_exactly():
    """The Figure 1 per-cell table is fully re-derivable from raw - the strongest
    claim in the provenance doc, so it gets the strictest test."""
    generated = _need(TABLES / "Figure_1_source_data.csv")
    reference = pd.read_csv(PUBLISHED_TABLES / "Figure_1_source_data.csv")
    assert len(generated) == len(reference) == 103667

    keys = ["panel", "experiment", "readout", "condition_value"]
    for df in (generated, reference):
        df["condition_value"] = df["condition_value"].astype(str)
    gn = generated.groupby(keys)["intensity_raw"].apply(lambda s: np.sort(s.values))
    go = reference.groupby(keys)["intensity_raw"].apply(lambda s: np.sort(s.values))
    assert set(gn.index) == set(go.index), "condition groups differ"
    worst = max(float(np.abs(gn[k] - go[k]).max()) for k in gn.index)
    assert worst < 1e-9, f"per-cell intensities differ by {worst:.3e}"


def test_complete_and_partial_reproduce_exactly():
    """`complete` is the measure the paper's conclusion rests on.  It must reproduce
    on every plate, including KZ223 under the saturation substitution."""
    generated = _need(TABLES / "Figure_S1_observed_points.csv")
    reference = pd.read_csv(PUBLISHED_TABLES / "Figure_S1_observed_points.csv")
    keys = ["experiment_id", "inhibitor_label", "readout", "dose", "type"]
    m = generated.merge(reference, on=keys, suffixes=("_generated", "_reference"))
    for measure in ("complete", "partial"):
        g = m[m["type"] == measure]
        worst = float((g["inhibition_generated"] - g["inhibition_reference"]).abs().max())
        assert worst < EXACT, f"{measure} differs by {worst:.3e}"


def test_mean_residual_is_bounded_and_confined_to_mean():
    """Cause 1 is real but must not have grown, and must not leak into other columns."""
    generated = _need(TABLES / "Figure_S1_observed_points.csv")
    reference = pd.read_csv(PUBLISHED_TABLES / "Figure_S1_observed_points.csv")
    keys = ["experiment_id", "inhibitor_label", "readout", "dose", "type"]
    m = generated.merge(reference, on=keys, suffixes=("_generated", "_reference"))
    m["ad"] = (m["inhibition_generated"] - m["inhibition_reference"]).abs()
    worst = float(m[m["type"] == "mean"]["ad"].max())
    assert worst < MEAN_RESIDUAL_MAX, f"mean residual grew to {worst:.4f}"
    # log-mean is exact everywhere EXCEPT KZ226 (cause 2), tested separately.
    lm = m[(m["type"] == "log-mean") & (m["experiment_id"] != "KZ226")]
    assert float(lm["ad"].max()) < 1e-6


def test_kz226_logmean_residual_is_bounded():
    """Cause 2: KZ226 log-mean, dose > 0.  Dose-0 rows are excluded because the
    published file carries duplicate dose-0 keys with two different reference values -
    a structural quirk documented in the provenance doc, not a computation difference."""
    generated = _need(TABLES / "Figure_S1_observed_points.csv")
    reference = pd.read_csv(PUBLISHED_TABLES / "Figure_S1_observed_points.csv")
    keys = ["experiment_id", "inhibitor_label", "readout", "dose", "type"]
    m = generated.merge(reference, on=keys, suffixes=("_generated", "_reference"))
    g = m[(m["experiment_id"] == "KZ226") & (m["type"] == "log-mean")
          & (m["dose"] > 0)]
    assert len(g) > 0
    worst = float((g["inhibition_generated"] - g["inhibition_reference"]).abs().max())
    assert worst < KZ226_LOGMEAN_MAX, f"KZ226 log-mean residual grew to {worst:.3e}"


def test_kz223_saturation_substitution_is_load_bearing():
    """Without the substitution KZ223 log-mean is off by >0.2; with it, ~1e-08.
    Guards against someone 'simplifying' figure_tables.py by dropping the special case."""
    from jakstat_dosegap import figure_tables

    with_sub = figure_tables.published_inhibition_table(
        ["KZ223"], kz223_saturation_substitution=True)
    without = figure_tables.published_inhibition_table(
        ["KZ223"], kz223_saturation_substitution=False)
    reference = pd.read_csv(PUBLISHED_TABLES / "Figure_S1_observed_points.csv")
    reference = reference[(reference["experiment_id"] == "KZ223") & (reference["type"] == "log-mean")]
    keys = ["inhibitor_label", "readout", "dose", "type"]

    def worst(df):
        m = df[df["type"] == "log-mean"].merge(
            reference, on=keys, suffixes=("_generated", "_reference")
        )
        return float(
            (m["inhibition_generated"] - m["inhibition_reference"]).abs().max()
        )

    assert worst(with_sub) < 1e-6
    assert worst(without) > 0.2, "substitution no longer changes the result"


def test_figure_scripts_emit_nontrivial_pngs():
    """Each figure exists, has its documented dimensions, and is not a blank canvas."""
    from PIL import Image

    expected = {
        "Figure_1.png": (3150, 3540),
        "Figure_2.png": (3000, 2400),
        "Figure_S1.png": (4488, 3023),
        "Figure_S2_selectivity.png": (3286, 1188),
    }
    for name, size in expected.items():
        path = REPO / "figures" / name
        if not path.exists():
            pytest.skip(f"{name} not rendered - run scripts/12..15")
        im = Image.open(path)
        assert im.size == size, f"{name}: {im.size} != {size}"
        arr = np.asarray(im.convert("L"))
        # A blank figure has near-zero variance; every real one has ink.
        assert arr.std() > 5.0, f"{name} looks blank (std {arr.std():.2f})"
        assert (arr < 128).mean() > 0.001, f"{name} has almost no dark pixels"


def test_state_colour_sources_are_explicit():
    """Reference-raster scripts pin their hexes; Figure 1 uses the state palette."""
    from jakstat_dosegap import config

    published_complete, published_partial = "#2171b5", "#cb181d"
    assert config.PALETTE["complete"] != published_complete
    assert config.PALETTE["partial"] != published_partial

    scripts = ["13_figure_2.py", "14_figure_S1.py", "15_figure_S2.py"]
    for name in scripts:
        src = (REPO / "scripts" / name).read_text()
        assert published_complete in src, f"{name} lost the published complete hex"
    src1 = (REPO / "scripts" / "12_figure_1.py").read_text()
    assert 'COMPLETE = config.PALETTE["complete"]' in src1
    assert 'PARTIAL = config.PALETTE["partial"]' in src1


def test_figure1_states_the_colour_convention_on_the_figure():
    """The Figure 1 legend states the operational color convention for state occupancy."""
    src = (REPO / "scripts" / "12_figure_1.py").read_text()
    assert "Blue = complete-inhibition state" in src
    assert "red = partial-only state" in src
    assert "residual pSTAT-positive cells" in src
