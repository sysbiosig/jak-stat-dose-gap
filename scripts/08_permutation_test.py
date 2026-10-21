"""CONTRACT
Purpose: label-permutation test for the significance of the CELL-LINE contribution
         to between-profile variance in complete pSTAT inhibition.

WHY this exists: the manuscript reports "permutation p = 0.002" for the cell-line
         variance share (Results and the Figure 3 caption) and describes the
         procedure in Methods as shuffling cell-line labels over 2,000
         permutations.  Until this script was added, NOTHING in the repository
         implemented it and no p-value was recorded anywhere -- the number was
         reproducible only by hand.  With only two cell lines the share could be
         inflated by the small number of levels, which is exactly why the test
         matters and why it must ship with the analysis.

Method : the same main-effects type-II ANOVA used in 04_decompositions.py
         (cell + inhibitor + phospho-target), applied to the observed complete
         fraction at each population-inhibition threshold crossing.  The
         cell-line labels are permuted; the share of total sum of squares
         attributed to cell line is recomputed on each permutation.  p is the
         proportion of permutations reaching the observed share, using the
         add-one estimator (ge + 1) / (n + 1) so p is never reported as zero.

Input  : data/derived/crossings.csv
Output : data/derived/permutation_tests.csv - one row per (threshold, axis):
         observed cell-line share, permutation p, n_profiles, n_permutations.
Runtime: ~55 s (6 threshold/axis combinations x 2,000 ANOVA refits).

Usage:  python scripts/08_permutation_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jakstat_dosegap import config

N_PERM = 2000
FORMULA = "value ~ C(cell) + C(inhibitor) + C(readout)"


def cell_line_share(frame):
    """Percent of total sum of squares attributed to cell line (type-II ANOVA).

    eval_env is pinned to an EMPTY namespace: patsy resolves `C(...)` against the
    caller's scope by default, so any local named C silently shadows the
    categorical-coding helper and the fit fails with a confusing TypeError.
    """
    model = smf.ols(FORMULA, data=frame, eval_env=patsy.EvalEnvironment({})).fit()
    table = anova_lm(model, typ=2)
    return 100.0 * table.loc["C(cell)", "sum_sq"] / table["sum_sq"].sum()


def permutation_p(frame, n_perm=N_PERM, seed=config.BOOTSTRAP_SEED):
    observed = cell_line_share(frame)
    rng = np.random.default_rng(seed)
    shuffled = frame.copy()
    labels = frame["cell"].to_numpy()
    at_least = 0
    for _ in range(n_perm):
        shuffled["cell"] = rng.permutation(labels)
        if cell_line_share(shuffled) >= observed:
            at_least += 1
    # add-one estimator: never reports p = 0, which 2,000 permutations cannot support
    return observed, (at_least + 1) / (n_perm + 1)


def build():
    crossings = pd.read_csv(config.derived("crossings.csv"))
    rows = []
    for threshold in sorted(crossings["threshold"].unique()):
        for axis in ("arith", "log"):
            sub = crossings[(crossings.threshold == threshold)
                            & (crossings.axis == axis)].dropna(subset=["complete"])
            sub = sub.rename(columns={"complete": "value"})
            observed, p = permutation_p(sub)
            rows.append({"threshold": threshold, "axis": axis,
                         "quantity": "complete_fraction",
                         "cell_line_share_pct": observed,
                         "permutation_p": p,
                         "n_profiles": len(sub),
                         "n_permutations": N_PERM,
                         "seed": config.BOOTSTRAP_SEED})
            print(f"  threshold {threshold} {axis:5s}: n={len(sub):2d} "
                  f"cell share {observed:5.2f}%  p={p:.4f}")
    return pd.DataFrame(rows)


def main():
    out = build()
    dest = config.derived("permutation_tests.csv")
    out.round(6).to_csv(dest, index=False)
    headline = out[(out.threshold == 0.8)]
    print("\nthe value quoted in the manuscript (threshold 0.8):")
    for _, r in headline.iterrows():
        print(f"  {r.axis:5s} axis: cell-line share {r.cell_line_share_pct:.2f}%, "
              f"p = {r.permutation_p:.4f}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
