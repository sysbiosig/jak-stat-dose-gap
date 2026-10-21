"""Unit tests for the jakstat_dosegap pipeline.

SCOPE
    These tests check the LOGIC of each module against hand-constructed inputs whose
    correct answer is known analytically - they do not touch the raw data and run in
    under a second.  Reproduction of the published numbers from the real raw files is
    the job of scripts/99_validate_derived.py (and test_reproduction.py, which is
    skipped unless the derived tables exist).

WHY synthetic fixtures: a test that recomputes a published number from the raw data
proves the pipeline is self-consistent but cannot localise a fault.  These tests pin
each module's contract independently, so a regression names its own culprit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jakstat_dosegap import config, crossings, decomposition, fitting, io, metrics


# ---------------------------------------------------------------- metrics: gating
def test_gate_drops_out_of_range_and_nan():
    x = metrics.gate([0.5, 1.0, 2.0, 100.0, 1e4, 2e4, np.nan])
    # Gate is strict on both sides: 1.0 and 1e4 are excluded, 0.5/2e4/NaN dropped.
    assert np.allclose(x, [2.0, 100.0])


def test_gate_accepts_series_and_array():
    a = metrics.gate(pd.Series([2.0, 5.0]))
    b = metrics.gate(np.array([2.0, 5.0]))
    assert np.allclose(a, b)


# ---------------------------------------------------------------- metrics: bounds
def test_complete_bound_is_95th_percentile_of_unstimulated():
    unstim = np.linspace(2, 102, 1001)
    sat = np.full(500, 500.0)
    b = metrics.reference_bounds(unstim, sat)
    assert b.complete == pytest.approx(np.percentile(unstim, 95))


def test_partial_bound_clamped_to_complete_when_distributions_overlap():
    """The clamp is load-bearing: without it partial would be STRICTER than complete."""
    unstim = np.linspace(2, 1000, 1000)   # wide: 95th pct is high
    sat = np.linspace(2, 100, 1000)       # overlapping and lower: 5th pct is low
    b = metrics.reference_bounds(unstim, sat)
    assert np.percentile(sat, 5) < b.complete       # the unclamped value is lower
    assert b.partial == pytest.approx(b.complete)   # so the clamp took effect


def test_partial_bound_is_5th_pct_of_saturated_when_well_separated():
    unstim = np.linspace(2, 10, 1000)
    sat = np.linspace(100, 500, 1000)
    b = metrics.reference_bounds(unstim, sat)
    assert b.partial == pytest.approx(np.percentile(sat, 5))
    assert b.partial > b.complete


def test_partial_bound_never_below_complete_bound_random():
    rng = np.random.default_rng(0)
    for _ in range(50):
        unstim = rng.lognormal(2.0, rng.uniform(0.2, 1.5), 500)
        sat = rng.lognormal(rng.uniform(2.0, 5.0), rng.uniform(0.2, 1.5), 500)
        b = metrics.reference_bounds(np.clip(unstim, 1.1, 9e3),
                                     np.clip(sat, 1.1, 9e3))
        assert b.partial >= b.complete


def test_reference_bounds_raises_on_too_few_control_cells():
    with pytest.raises(ValueError):
        metrics.reference_bounds([2.0, 3.0], np.full(500, 100.0))


# ---------------------------------------------------------------- metrics: measures
@pytest.fixture
def bounds():
    """Well-separated controls: unstim ~ 10, saturated ~ 1000."""
    return metrics.reference_bounds(np.full(500, 10.0), np.full(500, 1000.0))


def test_mean_inhibition_endpoints(bounds):
    # At the saturated level inhibition is 0; at the unstimulated level it is 1.
    assert metrics.mean_inhibition(np.full(100, 1000.0), bounds) == pytest.approx(0.0)
    assert metrics.mean_inhibition(np.full(100, 10.0), bounds) == pytest.approx(1.0)


def test_logmean_inhibition_endpoints_and_midpoint(bounds):
    assert metrics.logmean_inhibition(np.full(100, 1000.0), bounds) == pytest.approx(0.0)
    assert metrics.logmean_inhibition(np.full(100, 10.0), bounds) == pytest.approx(1.0)
    # 100 is the geometric midpoint of 10 and 1000, so the log-mean measure reads 0.5
    # while the arithmetic measure does not - this is the whole point of having both.
    assert metrics.logmean_inhibition(np.full(100, 100.0), bounds) == pytest.approx(0.5)
    assert metrics.mean_inhibition(np.full(100, 100.0), bounds) != pytest.approx(0.5)


def test_state_fractions_count_cells_below_bounds(bounds):
    # Both bounds are STRICT (<), and with constant controls the complete bound sits
    # exactly at 10.0, so test cells are placed just below/above it rather than on it.
    x = np.concatenate([np.full(30, 9.0), np.full(70, 1000.0)])
    assert metrics.complete_fraction(x, bounds) == pytest.approx(0.30)
    assert metrics.partial_fraction(x, bounds) == pytest.approx(0.30)


def test_state_fraction_bounds_are_strict_inequalities(bounds):
    """A cell exactly AT the bound is not counted as within the non-stimulated range."""
    assert metrics.complete_fraction(np.full(100, bounds.complete), bounds) == 0.0
    assert metrics.partial_fraction(np.full(100, bounds.partial), bounds) == 0.0


def test_partial_fraction_at_least_complete_fraction(bounds):
    rng = np.random.default_rng(1)
    x = np.clip(rng.lognormal(4.0, 1.5, 2000), 1.1, 9e3)
    assert metrics.partial_fraction(x, bounds) >= metrics.complete_fraction(x, bounds)


def test_all_measures_agrees_with_individual_functions(bounds):
    rng = np.random.default_rng(2)
    x = np.clip(rng.lognormal(4.5, 1.0, 1000), 1.1, 9e3)
    m = metrics.all_measures(x, bounds)
    assert m["mean_inh"] == pytest.approx(metrics.mean_inhibition(x, bounds))
    assert m["logmean_inh"] == pytest.approx(metrics.logmean_inhibition(x, bounds))
    assert m["complete"] == pytest.approx(metrics.complete_fraction(x, bounds))
    assert m["partial"] == pytest.approx(metrics.partial_fraction(x, bounds))
    assert m["n"] == len(metrics.gate(x))


def test_measures_are_not_clipped_to_unit_interval(bounds):
    """A cell population brighter than saturated gives negative inhibition, on purpose:
    clipping here would bias the 4PL fit."""
    assert metrics.mean_inhibition(np.full(100, 5000.0), bounds) < 0


# ---------------------------------------------------------------- fitting
def test_four_pl_reduces_to_plateaus_and_midpoint():
    # With hill=1 the curve at x=ic50 sits exactly halfway between the plateaus.
    assert fitting.four_pl(10.0, 0.0, 1.0, 10.0, 1.0) == pytest.approx(0.5)
    # x -> 0 gives `top`, x -> inf gives `bottom` (see fitting.four_pl docstring).
    assert fitting.four_pl(1e-9, 0.0, 1.0, 10.0, 1.0) == pytest.approx(1.0, abs=1e-6)
    assert fitting.four_pl(1e9, 0.0, 1.0, 10.0, 1.0) == pytest.approx(0.0, abs=1e-6)


def test_increasing_response_is_encoded_as_bottom_above_top():
    """Pins the orientation convention: inhibition rising with dose means bottom > top
    and a POSITIVE hill, not a negative exponent.  The hill bounds are [0.2, 5.0], so a
    negative hill is not even representable."""
    assert config.FIT_BOUNDS_LOW[3] > 0
    rising = fitting.four_pl(np.array([0.01, 1.0, 100.0]), 1.0, 0.0, 1.0, 1.5)
    assert np.all(np.diff(rising) > 0)


def test_fit_recovers_known_4pl_parameters():
    """Exact-data round trip: the fit must return the parameters that generated it."""
    dose = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    true = (1.0, 0.0, 0.5, 1.5)   # bottom > top => inhibition rises with dose
    y = fitting.four_pl(dose, *true)
    fit = fitting.fit_4pl(dose, y)
    assert fit.ok
    assert fit.ic50 == pytest.approx(0.5, rel=1e-3)
    assert fit.params[3] == pytest.approx(1.5, rel=1e-3)
    assert fit.params[0] > fit.params[1]
    # atol=1e-4, not 1e-6: the generating `top` is 0.0, which sits exactly ON the
    # optimiser's lower bound, so curve_fit converges to ~6e-6 rather than to 0 and the
    # residuals inherit that offset.  This is solver convergence on a synthetic round
    # trip; it is unrelated to the published-value tolerances in 99_validate_derived.py.
    assert np.allclose(fit.predict(), y, atol=1e-4)


def test_fit_recovers_ic50_under_noise():
    rng = np.random.default_rng(3)
    dose = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    y = fitting.four_pl(dose, 1.0, 0.0, 1.0, 1.0) + rng.normal(0, 0.02, len(dose))
    fit = fitting.fit_4pl(dose, y)
    assert fit.ok
    assert fit.ic50 == pytest.approx(1.0, rel=0.25)


def test_fit_fails_gracefully_with_too_few_doses():
    fit = fitting.fit_4pl([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    assert not fit.ok
    assert np.isnan(fit.ic50)
    assert "distinct doses" in fit.message
    assert np.all(np.isnan(fit.predict()))


def test_fit_ignores_nonpositive_and_nonfinite_doses():
    dose = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, np.nan]
    y = [0.0, 0.1, 0.3, 0.5, 0.8, 0.95, 0.5]
    fit = fitting.fit_4pl(dose, y)
    assert fit.ok
    assert len(fit.dose) == 5 and fit.dose.min() > 0


def test_fit_ic50_stays_inside_scaled_dose_bounds():
    dose = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    # A response that never rises: the optimiser cannot place a real IC50, but the
    # bounds must still contain whatever it returns.
    fit = fitting.fit_4pl(dose, np.full(5, 0.05))
    if fit.ok:
        assert dose.min() * 0.1 <= fit.ic50 <= dose.max() * 10.0


def test_fit_dose_vector_is_sorted():
    fit = fitting.fit_4pl([10.0, 0.1, 1.0, 0.01, 100.0], [0.9, 0.1, 0.5, 0.02, 0.99])
    assert np.all(np.diff(fit.dose) > 0)


# ---------------------------------------------------------------- fixture: profiles
def _synth_perdose(ic50s: dict[str, float], doses=(0.01, 0.1, 1.0, 10.0, 100.0),
                   experiments=("E1", "E2"), inhibitors=("inhA", "inhB", "inhC"),
                   readouts=("pSTAT1", "pSTAT3")) -> pd.DataFrame:
    """Synthetic per-dose table from known 4PL curves; one profile per combination."""
    rows = []
    cells = {"E1": "THP1", "E2": "U937"}
    for e in experiments:
        for i_idx, inh in enumerate(inhibitors):
            for r_idx, ro in enumerate(readouts):
                shift = 1.0 + 0.3 * i_idx + 0.1 * r_idx
                for d in doses:
                    row = {"experiment": e, "cell": cells[e], "inhibitor": inh,
                           "readout": ro, "dose": float(d), "n": 500,
                           "complete_bound": 20.0, "partial_bound": 40.0}
                    for measure, ic in ic50s.items():
                        # bottom=1, top=0, positive hill -> inhibition rises with dose,
                        # matching the orientation the real fits take.
                        row[measure] = float(
                            fitting.four_pl(d, 1.0, 0.0, ic * shift, 1.5))
                    rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def synth():
    # complete needs ~4x the dose of the log-mean axis and ~10x the arithmetic axis.
    return _synth_perdose({"mean_inh": 0.1, "logmean_inh": 0.25,
                           "complete": 1.0, "partial": 0.5})


def test_ic50_table_shape_and_fold_ratios(synth):
    tab = fitting.ic50_table(synth)
    assert len(tab) == 2 * 3 * 2
    assert all(tab[f"fitok_{m}"].all() for m in config.ALL_MEASURES)
    # Fold ratios must equal the ratio of the IC50s that generated the curves.
    assert tab["fold_complete_over_logmean"].median() == pytest.approx(4.0, rel=0.02)
    assert tab["fold_complete_over_arith"].median() == pytest.approx(10.0, rel=0.02)
    assert tab["fold_partial_over_arith"].median() == pytest.approx(5.0, rel=0.02)
    assert tab["fold_partial_over_logmean"].median() == pytest.approx(2.0, rel=0.02)


def test_ic50_fold_ratio_is_quotient_of_reported_ic50s(synth):
    tab = fitting.ic50_table(synth)
    r = tab.iloc[0]
    assert r["fold_complete_over_arith"] == pytest.approx(
        r["ic50_complete"] / r["ic50_mean_inh"])


# ---------------------------------------------------------------- crossings
def test_crossing_returns_a_measured_dose_not_an_interpolated_one(synth):
    g = synth[(synth.experiment == "E1") & (synth.inhibitor == "inhA")
              & (synth.readout == "pSTAT1")]
    res = crossings.crossing_for_profile(g, 0.5, "mean_inh")
    assert res is not None
    assert res["dose"] in set(g["dose"])            # snap-to-measured-dose
    assert res["pop_fitted"] >= 0.5                 # the FITTED curve cleared the bar


def test_crossing_takes_the_first_qualifying_dose(synth):
    g = synth[(synth.experiment == "E1") & (synth.inhibitor == "inhA")
              & (synth.readout == "pSTAT1")].sort_values("dose")
    res = crossings.crossing_for_profile(g, 0.2, "mean_inh")
    fitted = fitting.fit_4pl(g["dose"].values, g["mean_inh"].values).predict()
    assert res["dose_index"] == int(np.where(fitted >= 0.2)[0][0])


def test_crossing_state_fractions_are_observed_values_at_that_dose(synth):
    g = synth[(synth.experiment == "E2") & (synth.inhibitor == "inhB")
              & (synth.readout == "pSTAT3")]
    res = crossings.crossing_for_profile(g, 0.5, "logmean_inh")
    at = g[g["dose"] == res["dose"]].iloc[0]
    assert res["complete"] == pytest.approx(at["complete"])
    assert res["partial"] == pytest.approx(at["partial"])


def test_crossing_returns_none_when_curve_never_reaches_threshold():
    """Profile attrition at high thresholds is by design, not an error."""
    weak = _synth_perdose({"mean_inh": 1e5, "logmean_inh": 1e5,
                           "complete": 1e5, "partial": 1e5})
    g = weak[(weak.experiment == "E1") & (weak.inhibitor == "inhA")
             & (weak.readout == "pSTAT1")]
    assert crossings.crossing_for_profile(g, 0.8, "mean_inh") is None


def test_crossing_dose_is_monotone_in_threshold(synth):
    g = synth[(synth.experiment == "E1") & (synth.inhibitor == "inhA")
              & (synth.readout == "pSTAT1")]
    doses = [crossings.crossing_for_profile(g, t, "mean_inh")["dose"]
             for t in (0.2, 0.5, 0.8)]
    assert doses == sorted(doses)


def test_crossings_table_covers_thresholds_and_both_axes(synth):
    tab = crossings.crossings_table(synth)
    assert set(tab["threshold"]) == set(config.CROSSING_THRESHOLDS)
    assert set(tab["axis"]) == {"arith", "log"}
    assert set(tab.columns) >= {"complete", "partial", "pop_fitted", "pop_observed"}


def test_crossing_medians_profile_counts_match_row_counts(synth):
    tab = crossings.crossings_table(synth)
    med = crossings.crossing_medians(tab)
    for _, r in med.iterrows():
        sub = tab[(tab.threshold == r["threshold"]) & (tab.axis == r["axis"])]
        assert int(r["n_profiles"]) == len(sub)
        assert r["complete_median"] == pytest.approx(sub["complete"].median())


def test_log_axis_crossing_is_never_earlier_than_arith_axis(synth):
    """The log-mean axis rises later here (larger IC50), so it must cross no earlier."""
    tab = crossings.crossings_table(synth)
    piv = tab.pivot_table(index=["experiment", "inhibitor", "readout", "threshold"],
                          columns="axis", values="dose")
    assert (piv["log"] >= piv["arith"]).all()


# ---------------------------------------------------------------- decomposition
def test_variance_shares_sum_to_100():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({
        "cell": ["THP1", "U937"] * 12,
        "inhibitor": [f"i{k}" for k in range(6)] * 4,
        "readout": ["pSTAT1"] * 12 + ["pSTAT3"] * 12,
        "y": rng.normal(size=24),
    })
    s = decomposition.variance_shares(df, "y")
    assert sum(s[k] for k in ("cell", "inhibitor", "target", "unexplained")) \
        == pytest.approx(100.0)


def _crossed_design() -> pd.DataFrame:
    """Fully crossed 2 cells x 6 inhibitors x 2 readouts = 24 rows.

    WHY not ["THP1","U937"]*12 with [f"i{k}" for k in range(6)]*4: those two cycles have
    periods 2 and 6, so every inhibitor level pairs with exactly one cell line and the
    factors are perfectly collinear.  Type-II SS then splits a pure cell effect across
    the aliased factors, which looks like a decomposition bug but is the design's fault.
    The real 24-profile design IS fully crossed, so the fixture must be too.
    """
    return pd.DataFrame([
        {"cell": c, "inhibitor": f"i{k}", "readout": r}
        for c in ("THP1", "U937") for k in range(6) for r in ("pSTAT1", "pSTAT3")
    ])


def test_crossed_design_fixture_is_not_collinear():
    df = _crossed_design()
    assert len(df) == 24
    # Each inhibitor must appear with BOTH cell lines, else the factors are aliased.
    assert (df.groupby("inhibitor")["cell"].nunique() == 2).all()
    assert (df.groupby("cell")["inhibitor"].nunique() == 6).all()


def test_variance_shares_attribute_a_pure_cell_effect_to_cell():
    df = _crossed_design()
    df["y"] = np.where(df["cell"] == "THP1", 0.0, 10.0)   # cell explains everything
    s = decomposition.variance_shares(df, "y")
    assert s["cell"] == pytest.approx(100.0, abs=1e-6)
    assert s["inhibitor"] == pytest.approx(0.0, abs=1e-6)
    assert s["unexplained"] == pytest.approx(0.0, abs=1e-6)


def test_variance_shares_attribute_a_pure_readout_effect_to_target():
    """The readout factor is reported under the published label "target"."""
    df = _crossed_design()
    df["y"] = np.where(df["readout"] == "pSTAT1", 1.0, 5.0)
    s = decomposition.variance_shares(df, "y")
    assert s["target"] == pytest.approx(100.0, abs=1e-6)


def test_variance_shares_split_additive_two_factor_effect():
    """An additive cell+inhibitor effect must be attributed to both, not to noise."""
    df = _crossed_design()
    inh_effect = {f"i{k}": float(k) for k in range(6)}
    df["y"] = (np.where(df["cell"] == "THP1", 0.0, 10.0)
               + df["inhibitor"].map(inh_effect))
    s = decomposition.variance_shares(df, "y")
    assert s["unexplained"] == pytest.approx(0.0, abs=1e-6)
    assert s["cell"] > 0 and s["inhibitor"] > 0
    assert s["target"] == pytest.approx(0.0, abs=1e-6)


def test_variance_shares_reports_pure_noise_as_unexplained():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "cell": ["THP1", "U937"] * 60,
        "inhibitor": [f"i{k}" for k in range(6)] * 20,
        "readout": ["pSTAT1", "pSTAT3"] * 60,
        "y": rng.normal(size=120),
    })
    s = decomposition.variance_shares(df, "y")
    assert s["unexplained"] > 80.0


def test_variance_shares_is_order_independent():
    """Type-II SS must not depend on the order the factors are listed."""
    rng = np.random.default_rng(6)
    df = pd.DataFrame({
        "cell": ["THP1", "U937"] * 12,
        "inhibitor": [f"i{k}" for k in range(6)] * 4,
        "readout": ["pSTAT1"] * 12 + ["pSTAT3"] * 12,
        "y": rng.normal(size=24),
    })
    a = decomposition.variance_shares(df, "y", factors=("cell", "inhibitor", "readout"))
    b = decomposition.variance_shares(df, "y", factors=("readout", "inhibitor", "cell"))
    for k in ("cell", "inhibitor", "target", "unexplained"):
        assert a[k] == pytest.approx(b[k])


def test_variance_shares_nan_on_insufficient_data():
    df = pd.DataFrame({"cell": ["A", "B"], "inhibitor": ["x", "y"],
                       "readout": ["p", "q"], "y": [1.0, 2.0]})
    s = decomposition.variance_shares(df, "y")
    assert np.isnan(s["cell"]) and np.isnan(s["unexplained"])


def test_bootstrap_median_ci_is_deterministic_for_a_seed():
    rng = np.random.default_rng(7)
    v = rng.lognormal(1.0, 0.6, 24)
    a = decomposition.bootstrap_median_ci(v, n=2000, seed=7)
    b = decomposition.bootstrap_median_ci(v, n=2000, seed=7)
    assert a == b
    assert a[1] <= a[0] <= a[2]
    assert a[0] == pytest.approx(np.median(v))


def test_bootstrap_median_ci_is_stable_across_seeds():
    """Seed choice must not move the published interval materially.

    NOTE: the resampled median of n=24 values takes only a few discrete values, so the
    2.5/97.5 percentiles frequently land on the SAME number for different seeds - an
    inequality test would be testing the discreteness, not the estimator.  What matters
    for reproducibility is that the interval is seed-INSENSITIVE, which is asserted here.
    """
    rng = np.random.default_rng(8)
    v = rng.lognormal(1.0, 0.6, 24)
    cis = [decomposition.bootstrap_median_ci(v, n=4000, seed=s) for s in (7, 8, 42, 123)]
    los = [c[1] for c in cis]
    his = [c[2] for c in cis]
    assert max(los) - min(los) < 0.25 * np.median(v)
    assert max(his) - min(his) < 0.25 * np.median(v)


def test_bootstrap_median_ci_narrows_as_sample_grows():
    """Sanity check that the interval reflects sample size, not just the seed."""
    rng = np.random.default_rng(9)
    small = decomposition.bootstrap_median_ci(rng.lognormal(1.0, 0.6, 15), n=3000, seed=7)
    large = decomposition.bootstrap_median_ci(rng.lognormal(1.0, 0.6, 600), n=3000, seed=7)
    assert (large[2] - large[1]) < (small[2] - small[1])


def test_bootstrap_median_ci_drops_nans():
    v = [1.0, 2.0, 3.0, np.nan]
    med, lo, hi = decomposition.bootstrap_median_ci(v, n=500, seed=7)
    assert med == pytest.approx(2.0)
    assert np.isfinite(lo) and np.isfinite(hi)


def test_dose_penalty_table_log10_column_matches_penalty(synth):
    ic50 = fitting.ic50_table(synth)
    pen = decomposition.dose_penalty_table(ic50)
    assert len(pen) == len(ic50) * len(fitting.FOLD_DEFS)
    assert np.allclose(pen["log10_penalty"], np.log10(pen["penalty"]))


def test_dose_penalty_decomposition_shares_sum_to_100(synth):
    ic50 = fitting.ic50_table(synth)
    dp = decomposition.dose_penalty_decomposition(ic50)
    for _, r in dp.iterrows():
        assert sum(r[k] for k in ("cell", "inhibitor", "target", "unexplained")) \
            == pytest.approx(100.0)


def test_fold_ratio_summary_covers_all_four_penalties(synth):
    s = decomposition.fold_ratio_summary(fitting.ic50_table(synth), n=500)
    assert set(s["quantity"]) == set(fitting.FOLD_DEFS)
    assert (s["ci_lo"] <= s["median"]).all() and (s["median"] <= s["ci_hi"]).all()


# ---------------------------------------------------------------- config integrity
def test_every_inhibitor_code_has_a_dose_column_and_a_label():
    for code in config.DOSECOL:
        assert code in config.LABELMAP
        assert config.DOSECOL[code] in config.DOSE_COLUMNS


def test_label_map_yields_six_distinct_compounds():
    assert len({config.LABELMAP[c] for c in config.DOSECOL}) == 6


def test_panel_experiments_all_have_a_cell_line():
    assert set(config.PANEL_EXPERIMENTS) == set(config.CELL)
    assert set(config.CELL.values()) == {"THP1", "U937"}


def test_axis_keys_cover_the_population_measures():
    assert set(config.AXIS_KEY) == set(config.POP_MEASURES)
    assert set(config.AXIS_KEY.values()) == {"arith", "log"}


def test_analysis_exclusions_cover_every_experiment():
    assert set(config.ANALYSIS_EXCLUSIONS) == set(config.ALL_EXPERIMENTS)


def test_kz198_analysis_exclusions_keep_its_control_wells():
    """Regression guard for the registry/analysis conflict: B10-B12 are KZ198's only
    control wells, so excluding them removes the reference distributions and drops the
    experiment.  See config.ANALYSIS_EXCLUSIONS."""
    excl = set(config.ANALYSIS_EXCLUSIONS["KZ198"])
    assert not (excl & {"B10", "B11", "B12"})


def test_histogram_only_experiments_are_not_in_the_panel():
    assert not set(config.HISTOGRAM_ONLY_EXPERIMENTS) & set(config.PANEL_EXPERIMENTS)


# ---------------------------------------------------------------- io helpers
def test_registry_exclusions_parse_space_separated_wells():
    excl = io.registry_exclusions()
    assert excl["KZ214"][:3] == ["A01", "A02", "A03"]
    assert excl["KZ195"] == []          # empty field -> empty list, not [""]


def test_registry_carries_the_analysis_valid_kz198_set():
    """The registry must ship the exclusion set the dose-response analysis can actually
    use.

    WHY THIS TEST EXISTS: the per-plate records disagree about which KZ198 wells to
    drop, and the repo's first registry build resolved the conflict by taking the
    LARGEST set (A12 B09 B10 B11 B12 E02). That was wrong -- see
    test_rejected_kz198_exclusion_set_destroys_the_controls below. The registry was
    corrected to resolve by analysis validity instead, and this test pins that
    correction so a future regeneration of the registry cannot silently revert it.
    """
    reg = io.registry_exclusions()
    assert set(reg["KZ198"]) == {"A12", "B05", "B09", "E02"}
    # KZ198 must retain at least one unstimulated and one saturated reference well.
    assert not io.exclusion_conflicts_block_controls("KZ198")


def test_strip_frame_removes_cr_from_names_and_values():
    df = pd.DataFrame({"well.name\r": ["A01\r", "B02\r"], "x": [1, 2]})
    out = io._strip_frame(df.copy())
    assert list(out.columns) == ["well.name", "x"]
    assert list(out["well.name"]) == ["A01", "B02"]
