"""Integration tests: the real raw data must reproduce the published values.

These tests are SKIPPED when the raw files or the derived tables are absent, so the
unit suite still runs on a checkout without the ~260 MB of raw data.

Structure/consistency checks read the derived tables (fast).  The one test that
reloads the raw data is marked `slow` - deselect with `-m "not slow"`.

Tolerances are the same as scripts/99_validate_derived.py and are rounding allowances
for values published to 2-3 decimals, NOT fit windows.  A failure here is a pipeline
discrepancy to diagnose, never a tolerance to widen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jakstat_dosegap import config, crossings, decomposition, fitting, io, metrics

REF_JSON = config.DATA_REFERENCE / "published_values.json"
TOL_FRACTION = 0.01
TOL_FOLD = 0.05
TOL_ANOVA_PP = 1.0

FOLD_KEYS = {
    "fold_complete_over_logmean": "complete_over_logmean",
    "fold_complete_over_arith": "complete_over_arithmean",
    "fold_partial_over_arith": "partial_over_arithmean",
    "fold_partial_over_logmean": "partial_over_logmean",
}


def _table(name: str) -> pd.DataFrame:
    path = config.DATA_DERIVED / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"{path.name} not built; run scripts/01-04 first")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def targets() -> dict:
    if not REF_JSON.exists():
        pytest.skip("reference target values not staged in data/reference/")
    return json.loads(REF_JSON.read_text())


@pytest.fixture(scope="module")
def perdose() -> pd.DataFrame:
    return _table("perdose")


@pytest.fixture(scope="module")
def ic50() -> pd.DataFrame:
    return _table("ic50_per_profile")


@pytest.fixture(scope="module")
def cross() -> pd.DataFrame:
    return _table("crossings")


# ---------------------------------------------------------------- structure
def test_perdose_has_242_rows_and_24_profiles(perdose):
    assert len(perdose) == 242
    assert perdose.groupby(["experiment", "inhibitor", "readout"]).ngroups == 24


def test_perdose_covers_the_expected_design(perdose):
    assert set(perdose["experiment"]) == set(config.PANEL_EXPERIMENTS)
    assert set(perdose["readout"]) == set(config.READOUTS)
    assert set(perdose["inhibitor"]) == {
        "solcitinib", "BMS-911543", "cerdulatinib",
        "baricitinib", "ruxolitinib", "tofacitinib"}
    assert set(perdose["cell"]) == {"THP1", "U937"}


def test_perdose_has_no_zero_doses_and_enough_cells(perdose):
    assert (perdose["dose"] > 0).all()
    assert (perdose["n"] >= config.MIN_CELLS).all()


def test_perdose_state_fractions_are_in_unit_interval(perdose):
    for col in config.STATE_MEASURES:
        assert perdose[col].between(0.0, 1.0).all()


def test_perdose_partial_bound_never_below_complete_bound(perdose):
    assert (perdose["partial_bound"] >= perdose["complete_bound"]).all()


def test_perdose_partial_fraction_never_below_complete_fraction(perdose):
    assert (perdose["partial"] >= perdose["complete"] - 1e-12).all()


def test_bounds_are_constant_within_experiment_and_readout(perdose):
    """Bounds come from each plate's own controls, so they must not vary by dose."""
    g = perdose.groupby(["experiment", "readout"])[["complete_bound", "partial_bound"]]
    assert (g.nunique() == 1).all().all()


def test_all_4pl_fits_converged(ic50):
    for m in config.ALL_MEASURES:
        assert ic50[f"fitok_{m}"].all(), f"{m} has non-converged fits"


# ---------------------------------------------------------------- published values
def test_ic50_fold_medians_match_published(ic50, targets):
    s = decomposition.fold_ratio_summary(ic50).set_index("quantity")
    for col, key in FOLD_KEYS.items():
        assert s.loc[col, "median"] == pytest.approx(
            targets["ic50_fold_medians"][key], abs=TOL_FOLD), key


def test_ic50_fold_bootstrap_cis_match_published(ic50, targets):
    s = decomposition.fold_ratio_summary(ic50).set_index("quantity")
    for col, key in FOLD_KEYS.items():
        lo, hi = targets["ic50_fold_ci95"][key]
        assert s.loc[col, "ci_lo"] == pytest.approx(lo, abs=TOL_FOLD), f"{key} lo"
        assert s.loc[col, "ci_hi"] == pytest.approx(hi, abs=TOL_FOLD), f"{key} hi"


def test_bootstrap_settings_match_published(targets):
    assert config.BOOTSTRAP_N == targets["bootstrap"]["n"]
    assert config.BOOTSTRAP_SEED == targets["bootstrap"]["seed"]


def test_crossing_medians_and_counts_match_published(cross, targets):
    med = crossings.crossing_medians(cross)
    seen = 0
    for _, r in med.iterrows():
        t, a = f"{r['threshold']:g}", r["axis"]
        assert r["complete_median"] == pytest.approx(
            targets["complete_at_crossing_medians"][t][a], abs=TOL_FRACTION), f"comp {t}/{a}"
        assert r["partial_median"] == pytest.approx(
            targets["partial_at_crossing_medians"][t][a], abs=TOL_FRACTION), f"part {t}/{a}"
        assert int(r["n_profiles"]) == targets["n_profiles"][t][a], f"n {t}/{a}"
        seen += 1
    assert seen == 6


def test_state_fraction_decompositions_match_published(cross, targets):
    dec = decomposition.decomposition_table(cross)
    for state, tkey in (("complete", "complete_fraction_decomposition_pct"),
                        ("partial", "partial_fraction_decomposition_pct")):
        for t, per_axis in targets[tkey].items():
            for a, exp in per_axis.items():
                got = dec[(dec["quantity"] == f"{state}_fraction")
                          & (dec["axis"] == a)
                          & np.isclose(dec["threshold"], float(t))].iloc[0]
                for f in ("cell", "inhibitor", "target", "unexplained"):
                    assert got[f] == pytest.approx(exp[f], abs=TOL_ANOVA_PP), \
                        f"{state} {t}/{a} {f}"


def test_dose_penalty_decompositions_match_published(ic50, targets):
    dp = decomposition.dose_penalty_decomposition(ic50, transform="log10")
    for state, per_axis in targets["dose_penalty_decomposition_pct"].items():
        for a, exp in per_axis.items():
            got = dp[(dp["state"] == state) & (dp["axis"] == a)].iloc[0]
            for f in ("cell", "inhibitor", "target", "unexplained"):
                assert got[f] == pytest.approx(exp[f], abs=TOL_ANOVA_PP), \
                    f"penalty {state}/{a} {f}"


def test_linear_transform_does_not_reproduce_published_penalties(ic50, targets):
    """Guard on the log10 convention: decomposing raw ratios must visibly disagree, so
    a future refactor cannot silently switch scale and still appear to pass."""
    dp = decomposition.dose_penalty_decomposition(ic50, transform="linear")
    got = dp[(dp["state"] == "complete") & (dp["axis"] == "log")].iloc[0]
    exp = targets["dose_penalty_decomposition_pct"]["complete"]["log"]
    assert abs(got["cell"] - exp["cell"]) > 5 * TOL_ANOVA_PP


def test_conventions_match_published(perdose, targets):
    n = targets["normalization"]
    assert config.NF == n["divisor"]
    assert config.RESP["pSTAT1"] == n["pSTAT1_col"]
    assert config.RESP["pSTAT3"] == n["pSTAT3_col"]


# ---------------------------------------------------------------- raw-data reload
@pytest.mark.slow
def test_pipeline_rebuilds_perdose_from_raw_identically(perdose):
    """End-to-end determinism: reloading the raw files must give the same table."""
    for e in config.PANEL_EXPERIMENTS:
        if not config.raw_csv(e).exists():
            pytest.skip(f"raw data for {e} not present")
    frames = {e: io.load_experiment(e) for e in config.PANEL_EXPERIMENTS}
    fresh = metrics.perdose_table(frames)
    assert len(fresh) == len(perdose)
    cols = list(config.ALL_MEASURES) + ["dose", "n"]
    a = fresh.sort_values(["experiment", "readout", "inhibitor", "dose"]
                          ).reset_index(drop=True)[cols]
    b = perdose.sort_values(["experiment", "readout", "inhibitor", "dose"]
                            ).reset_index(drop=True)[cols]
    pd.testing.assert_frame_equal(a, b, atol=1e-9, check_dtype=False)


#: Rejected KZ198 exclusion set from the histogram workflow. Kept as a literal so the
#: test documents why the dose-response analysis must not use it.
REJECTED_KZ198_EXCLUSIONS = ["A12", "B09", "B10", "B11", "B12", "E02"]


@pytest.mark.slow
def test_rejected_kz198_exclusion_set_destroys_the_controls():
    """The rejected KZ198 set removes every reference well, so no bounds can be formed.

    This is the concrete reason the registry resolves conflicts by analysis validity
    instead of set size.  B11 and B12 are KZ198's only unstimulated wells and B09/B10
    its only inhibitor-free saturated wells; the rejected set drops all four.
    """
    if not config.raw_csv("KZ198").exists():
        pytest.skip("raw data for KZ198 not present")
    unstim, sat = io.reference_wells("KZ198")
    dropped = set(REJECTED_KZ198_EXCLUSIONS)
    assert unstim and sat, "KZ198 must have both reference distributions before exclusion"
    assert not (unstim - dropped), "rejected set should remove every unstimulated well"
    assert not (sat - dropped), "rejected set should remove every saturated well"
    assert io.exclusion_conflicts_block_controls("KZ198", REJECTED_KZ198_EXCLUSIONS)
    # ...and the downstream consequence: KZ198 contributes nothing.
    frames = {"KZ198": io.load_experiment("KZ198", exclude_wells=REJECTED_KZ198_EXCLUSIONS)}
    assert metrics.perdose_table(frames).empty


def test_registry_set_preserves_controls_for_every_experiment():
    """Every experiment's shipped exclusion set must leave its reference wells intact.

    Guards the whole registry, not just KZ198: this is the invariant whose violation
    silently drops an experiment from the analysis.
    """
    for exp_id in config.ALL_EXPERIMENTS:
        if not config.raw_csv(exp_id).exists():
            pytest.skip(f"raw data for {exp_id} not present")
        assert not io.exclusion_conflicts_block_controls(exp_id), (
            f"{exp_id}: registry exclusion set removes a needed reference distribution")
