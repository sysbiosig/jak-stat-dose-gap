"""Pins the Figure S1 "missing partial curve" explanation.

Three KZ223 / U937 / pSTAT3 profiles show no visible partial-inhibition curve in
Figure S1.  The reason is a threshold collapse, not missing data, and these tests
pin every link in that chain so a future change to gating, exclusions or the
percentile choice cannot quietly turn the explanation false.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jakstat_dosegap import config, io, metrics  # noqa: E402


@pytest.fixture(scope="module")
def diag():
    p = config.derived("threshold_separation_diagnostic.csv")
    if not p.exists():
        pytest.skip("run scripts/07_threshold_separation.py first")
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def perdose():
    p = config.derived("perdose.csv")
    if not p.exists():
        pytest.skip("perdose.csv not built; run scripts/01_build_perdose.py first")
    return pd.read_csv(p)


def test_exactly_one_experiment_readout_cell_clamps(diag):
    """The collapse is confined to KZ223 pSTAT3 -- if it spreads, Figure S1 changes."""
    clamped = diag.loc[diag.clamp_fires, ["experiment", "readout"]]
    assert set(map(tuple, clamped.values)) == {("KZ223", "pSTAT3")}


def test_clamp_is_an_inversion_not_a_tie(diag):
    """sat p05 must be strictly BELOW unstim p95 -- i.e. the references invert."""
    r = diag[(diag.experiment == "KZ223") & (diag.readout == "pSTAT3")].iloc[0]
    assert r.sat_p05 < r.unstim_p95


def test_partial_equals_complete_exactly_where_clamped(perdose):
    """The observable consequence: three profiles whose two curves coincide."""
    same = (perdose.groupby(["experiment", "cell", "inhibitor", "readout"])
            .apply(lambda d: np.allclose(d.partial.values, d.complete.values),
                   include_groups=False))
    coincident = set(k[:1] + k[3:] for k in same[same].index)
    assert coincident == {("KZ223", "pSTAT3")}, "collapse moved off KZ223/pSTAT3"
    assert int(same.sum()) == 3, f"expected 3 affected profiles, got {int(same.sum())}"


def test_bounds_are_equal_only_for_the_clamped_cell(perdose):
    eq = perdose.groupby(["experiment", "readout"]).apply(
        lambda d: np.isclose(d.complete_bound.iloc[0], d.partial_bound.iloc[0]),
        include_groups=False)
    assert set(eq[eq].index) == {("KZ223", "pSTAT3")}


def test_clamp_survives_bootstrap_of_the_reference_cells():
    """The inversion is not a sampling accident: P(clamp) == 1 over 2000 resamples.

    KZ223's unstimulated reference is a SINGLE well, so this is the check that the
    finding does not rest on a handful of cells.
    """
    frame = io.load_experiment("KZ223")
    unstim_wells, sat_wells = io.reference_wells("KZ223")
    well = frame["well"].astype(str).str.strip()

    def gate(wells):
        x = pd.to_numeric(frame.loc[well.isin(wells), "pSTAT3"], errors="coerce")
        return x[(x > config.SIGNAL_MIN) & (x < config.SIGNAL_MAX)].dropna().values

    u, s = gate(unstim_wells), gate(sat_wells)
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    bu = np.percentile(rng.choice(u, (2000, len(u)), replace=True), 95, axis=1)
    bs = np.percentile(rng.choice(s, (2000, len(s)), replace=True), 5, axis=1)
    assert np.mean(bs <= bu) == 1.0


def test_the_clamp_is_what_reference_bounds_actually_does():
    """Guard the mechanism directly: partial must never fall below complete."""
    unstim = np.concatenate([np.full(400, 10.0), np.full(100, 30.0)])   # p95 high
    saturated = np.concatenate([np.full(50, 12.0), np.full(450, 100.0)])  # p05 low
    b = metrics.reference_bounds(unstim, saturated)
    assert b.partial == b.complete == pytest.approx(np.percentile(unstim, 95))


def test_kz223_unstimulated_reference_rests_on_one_well(diag):
    """Documents the one-well reference so it remains visible during maintenance."""
    r = diag[(diag.experiment == "KZ223") & (diag.readout == "pSTAT3")].iloc[0]
    assert r.unstim_wells == 1
    assert r.unstim_well_ids.strip() == "H03"
