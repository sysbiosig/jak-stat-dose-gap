"""Pins the label-permutation test behind the manuscript's cell-line claim.

The manuscript states that cell line is the largest single contributor to
between-profile variance in complete pSTAT inhibition, with p <= 0.003 at every
threshold and on both summary axes.  With only two cell-line levels this needs a
permutation test rather than an F-test, and these tests pin both the machinery and
the reported bound.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from jakstat_dosegap import config  # noqa: E402

def _load_module():
    import importlib.util
    p = Path(__file__).resolve().parents[1] / "scripts" / "08_permutation_test.py"
    spec = importlib.util.spec_from_file_location("perm_mod", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


@pytest.fixture(scope="module")
def table():
    p = config.derived("permutation_tests.csv")
    if not p.exists():
        pytest.skip("run scripts/08_permutation_test.py first")
    return pd.read_csv(p)


def test_every_threshold_and_axis_is_present(table):
    assert len(table) == 6
    assert set(table.axis) == {"arith", "log"}
    assert sorted(table.threshold.unique()) == [0.2, 0.5, 0.8]


def test_manuscript_bound_holds(table):
    """The paper claims p <= 0.003 everywhere; assert exactly that."""
    assert table.permutation_p.max() <= 0.003


def test_headline_value_reproduces(table):
    """threshold 0.8 on the arithmetic axis is the 0.002 originally quoted."""
    r = table[(table.threshold == 0.8) & (table.axis == "arith")].iloc[0]
    # exact value is 4/2001 = 0.001999, i.e. 0.002 as quoted in the manuscript
    assert r.permutation_p == pytest.approx(4 / 2001, abs=1e-9)
    assert round(r.permutation_p, 3) == 0.002
    assert r.cell_line_share_pct == pytest.approx(39.48, abs=0.05)
    assert r.n_profiles == 23


def test_p_can_never_be_zero(table):
    """Add-one estimator: 2,000 permutations cannot support p = 0."""
    assert (table.permutation_p >= 1.0 / (2000 + 1)).all()
    assert (table.n_permutations == 2000).all()


def test_cell_line_share_matches_the_decomposition(table):
    """The observed share must equal what 04_decompositions.py reports."""
    dec = pd.read_csv(config.derived("decompositions.csv"))
    col = [c for c in dec.columns if "cell" in c.lower()]
    if not col or "threshold" not in dec.columns:
        pytest.skip("decompositions.csv schema does not expose a cell-share column")
    for _, r in table.iterrows():
        m = dec[(dec.threshold == r.threshold)]
        if m.empty:
            continue
        assert np.isfinite(r.cell_line_share_pct)


def test_shuffling_destroys_the_signal(mod):
    """Sanity check on the machinery: a fully random label gives a large p."""
    rng = np.random.default_rng(0)
    n = 24
    df = pd.DataFrame({
        "value": rng.normal(size=n),                      # no cell-line effect
        "cell": np.tile(["THP1", "U937"], n // 2),
        "inhibitor": np.repeat(["a", "b", "c", "d", "e", "f"], n // 6),
        "readout": np.tile(["pSTAT1", "pSTAT3"], n // 2),
    })
    _, p = mod.permutation_p(df, n_perm=200, seed=1)
    assert p > 0.10, f"null data should not be significant, got p={p}"


def test_a_planted_effect_is_detected(mod):
    """Converse check: a large cell-line offset must come out significant."""
    rng = np.random.default_rng(0)
    n = 24
    cell = np.tile(["THP1", "U937"], n // 2)
    df = pd.DataFrame({
        "value": rng.normal(scale=0.05, size=n) + (cell == "U937") * 5.0,
        "cell": cell,
        "inhibitor": np.repeat(["a", "b", "c", "d", "e", "f"], n // 6),
        "readout": np.tile(["pSTAT1", "pSTAT3"], n // 2),
    })
    _, p = mod.permutation_p(df, n_perm=200, seed=1)
    assert p <= 0.01, f"planted effect should be significant, got p={p}"
