"""Pin the extrapolation disclosure quoted in Methods."""
import numpy as np
import pandas as pd
import pytest
from jakstat_dosegap import config

MAX_TESTED_NM = 10000.0


@pytest.fixture(scope="module")
def ic():
    p = config.derived("ic50_per_profile.csv")
    if not p.exists():
        pytest.skip("run scripts/02_fit_ic50.py first")
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def sens():
    p = config.derived("extrapolation_sensitivity.csv")
    if not p.exists():
        pytest.skip("run scripts/09_extrapolation_sensitivity.py first")
    return pd.read_csv(p)


def test_every_curve_converged(ic):
    # Methods states all 96 fits converged, so no fallback estimator was applied.
    for m in ("mean_inh", "logmean_inh", "partial", "complete"):
        assert ic[f"fitok_{m}"].all(), f"{m} has a failed fit; Methods claims none"


def test_counts_above_max_tested_match_methods(sens):
    c = sens[sens.kind == "extrapolation_count"].set_index("metric")
    assert c.loc["arithmetic mean", "n_above_max_tested"] == 0
    assert c.loc["log-mean", "n_above_max_tested"] == 1
    assert c.loc["partial", "n_above_max_tested"] == 1
    assert c.loc["complete", "n_above_max_tested"] == 2


def test_exactly_one_ic50_at_the_fitting_bound(ic):
    at_bound = (ic["ic50_complete"] >= MAX_TESTED_NM * 10 - 1).sum()
    assert at_bound == 1, f"expected 1 complete IC50 pinned at the bound, got {at_bound}"


def test_twenty_two_profiles_fully_bracketed(ic):
    br = np.ones(len(ic), dtype=bool)
    for m in ("mean_inh", "logmean_inh", "partial", "complete"):
        br &= (ic[f"ic50_{m}"] <= MAX_TESTED_NM).values
    assert br.sum() == 22


def test_headline_survives_bracketed_restriction(sens):
    # The Methods sentence quotes 2.2 and 7.1 for the bracketed-only medians.
    h = sens[sens.kind == "headline_sensitivity"].set_index("metric")
    assert round(h.loc["complete / log-mean", "median_bracketed"], 1) == 2.2
    assert round(h.loc["complete / arith-mean", "median_bracketed"], 1) == 7.1
    # and the full-panel values are unchanged
    assert round(h.loc["complete / log-mean", "median_all"], 2) == 2.39
    assert round(h.loc["complete / arith-mean", "median_all"], 2) == 7.42


def test_bracketed_ci_overlaps_full_panel(sens):
    # The claim is "essentially unchanged": the CIs must overlap.
    h = sens[sens.kind == "headline_sensitivity"].set_index("metric")
    for m in ("complete / log-mean", "complete / arith-mean"):
        r = h.loc[m]
        assert r.ci95_low_bracketed <= r.ci95_high_all
        assert r.ci95_low_all <= r.ci95_high_bracketed


# --- Manuscript-consistency guards -------------------------------------------
# Added after a review caught the Methods sentence saying "three exceeded" while
# its own parenthetical (1 log-mean + 1 partial + 2 complete) and this CSV both
# say four.  A string-presence check cannot catch an arithmetic contradiction, so
# these tests parse the numbers out of the manuscript and compare them to the data.

import re
from pathlib import Path

MANUSCRIPT = Path(__file__).resolve().parents[1] / "manuscript" / "main_all.tex"
_WORD = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _manuscript_text():
    if not MANUSCRIPT.exists():
        pytest.skip("manuscript/main_all.tex not present")
    return " ".join(MANUSCRIPT.read_text().split())


def test_methods_exceedance_total_matches_data(sens):
    """The headline count in Methods must equal the summed per-metric counts."""
    txt = _manuscript_text()
    m = re.search(r"of the 96 fits, (\w+) exceeded", txt)
    assert m, "extrapolation-disclosure sentence not found in Methods"
    stated = _WORD.get(m.group(1).lower())
    assert stated is not None, f"could not parse count {m.group(1)!r}"
    actual = int(sens[sens.kind == "extrapolation_count"].n_above_max_tested.sum())
    assert stated == actual, f"Methods says {stated} exceedances, data say {actual}"


def test_methods_exceedance_breakdown_sums_to_headline():
    """The parenthetical breakdown must sum to the stated headline count."""
    txt = _manuscript_text()
    m = re.search(r"of the 96 fits, (\w+) exceeded[^(]*\(([^)]*)\)", txt)
    assert m, "extrapolation-disclosure breakdown not found"
    stated = _WORD[m.group(1).lower()]
    parts = [_WORD[w] for w in re.findall(r"\b(one|two|three|four|five)\b", m.group(2).lower())]
    assert sum(parts) == stated, (
        f"breakdown {parts} sums to {sum(parts)} but sentence says {stated}")


def test_methods_bracketed_profile_count_matches_data(ic):
    txt = _manuscript_text()
    m = re.search(r"(\w+[- ]?\w*) of the 24 profiles have all four", txt)
    assert m, "bracketed-profile sentence not found"
    stated = {"twenty-two": 22, "twenty-three": 23, "twenty-one": 21,
              "twenty-four": 24}.get(m.group(1).lower())
    assert stated is not None, f"could not parse {m.group(1)!r}"
    br = np.ones(len(ic), dtype=bool)
    for k in ("mean_inh", "logmean_inh", "partial", "complete"):
        br &= (ic[f"ic50_{k}"] <= MAX_TESTED_NM).values
    assert stated == int(br.sum())
