#!/usr/bin/env python3
"""99: assert the derived tables reproduce the published values.

CONTRACT
    Input  : all four tables under data/derived/ (run 01-04 first) and the published
             reference values at data/reference/published_values.json.
    Output : a printed expected-vs-actual comparison table for EVERY checked quantity,
             plus data/derived/validation_report.csv with the same content.
             Exit status 0 iff every check passes.

TOLERANCES (stated, fixed, and deliberately not negotiable)
    fractions        0.01  absolute  - state fractions and crossing medians; the
                                       published values are quoted to 3 decimals
    fold-ratios      0.05  absolute  - dose penalties and their CI bounds
    ANOVA shares     1.0   pp        - variance shares; the published values for the
                                       complete-fraction and dose-penalty blocks are
                                       rounded to whole percent
    profile counts   exact           - integer counts must match exactly

    These are ROUNDING allowances, not fit windows.  If a quantity misses, the cause is
    a pipeline discrepancy to be diagnosed - NEVER widen a tolerance to force a pass.
    The published values live in the reference JSON only; nothing in
    src/jakstat_dosegap/ reads this file, so no check here can be satisfied by the
    pipeline having been told the answer.

Usage:  python scripts/99_validate_derived.py [--reference PATH] [-v]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jakstat_dosegap import config, decomposition  # noqa: E402

TOL_FRACTION = 0.01
TOL_FOLD = 0.05
TOL_ANOVA_PP = 1.0

# fold-ratio column in ic50_per_profile.csv  ->  key in the reference JSON
FOLD_KEYS = {
    "fold_complete_over_logmean": "complete_over_logmean",
    "fold_complete_over_arith": "complete_over_arithmean",
    "fold_partial_over_arith": "partial_over_arithmean",
    "fold_partial_over_logmean": "partial_over_logmean",
}
FACTORS = ("cell", "inhibitor", "target", "unexplained")


class Checker:
    """Accumulates expected-vs-actual comparisons and reports pass/fail."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, group: str, name: str, expected, actual, tol: float,
              unit: str = "") -> bool:
        if actual is None or (isinstance(actual, float) and not np.isfinite(actual)):
            ok, diff = False, np.nan
        else:
            diff = float(actual) - float(expected)
            ok = abs(diff) <= tol
        self.rows.append({"group": group, "quantity": name, "expected": expected,
                          "actual": actual, "difference": diff, "tolerance": tol,
                          "unit": unit, "pass": ok})
        return ok

    def check_exact(self, group: str, name: str, expected: int, actual: int) -> bool:
        ok = int(expected) == int(actual)
        self.rows.append({"group": group, "quantity": name, "expected": int(expected),
                          "actual": int(actual),
                          "difference": int(actual) - int(expected), "tolerance": 0,
                          "unit": "count", "pass": ok})
        return ok

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def report(self, verbose: bool = False) -> bool:
        df = self.frame()
        for group, g in df.groupby("group", sort=False):
            n_fail = int((~g["pass"]).sum())
            print(f"\n--- {group}  ({len(g) - n_fail}/{len(g)} pass) ---")
            print(f"  {'quantity':44s} {'expected':>10s} {'actual':>10s} "
                  f"{'diff':>9s} {'tol':>7s}  ")
            for _, r in g.iterrows():
                if not verbose and r["pass"] and n_fail == 0 and len(g) > 14:
                    continue  # keep large all-passing blocks readable
                mark = "ok  " if r["pass"] else "FAIL"
                act = "  n/a" if not np.isfinite(float(r["actual"])) \
                    else f"{float(r['actual']):10.3f}"
                print(f"  {r['quantity']:44s} {float(r['expected']):10.3f} {act} "
                      f"{float(r['difference']):9.3f} {float(r['tolerance']):7.3f}  {mark}")
            if not verbose and n_fail == 0 and len(g) > 14:
                print(f"  (all {len(g)} within tolerance; -v to list individually)")
        n_fail = int((~df["pass"]).sum())
        print(f"\n{'=' * 78}")
        print(f"TOTAL: {len(df) - n_fail}/{len(df)} checks pass"
              + ("" if n_fail == 0 else f"  --  {n_fail} FAILURES"))
        print("=" * 78)
        return n_fail == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default=None,
                    help="path to published_values.json")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="list every check, including passing ones in large blocks")
    args = ap.parse_args()

    ref_path = Path(args.reference) if args.reference else \
        config.DATA_REFERENCE / "published_values.json"
    if not ref_path.exists():
        print(f"missing reference values at {ref_path}")
        return 2
    T = json.loads(ref_path.read_text())

    need = {n: config.DATA_DERIVED / f"{n}.csv" for n in
            ("perdose", "ic50_per_profile", "crossings", "decompositions")}
    missing = [str(p) for p in need.values() if not p.exists()]
    if missing:
        print("missing derived tables (run scripts 01-04 first):")
        print("  " + "\n  ".join(missing))
        return 2
    perdose = pd.read_csv(need["perdose"])
    ic50 = pd.read_csv(need["ic50_per_profile"])
    cross = pd.read_csv(need["crossings"])
    dec = pd.read_csv(need["decompositions"])

    c = Checker()

    # --- structure -------------------------------------------------------
    c.check_exact("structure", "perdose rows", 242, len(perdose))
    c.check_exact("structure", "profiles (WT)", 24, len(ic50))
    c.check_exact("structure", "experiments", 4, perdose["experiment"].nunique())
    c.check_exact("structure", "inhibitors", 6, perdose["inhibitor"].nunique())
    c.check_exact("structure", "readouts", 2, perdose["readout"].nunique())

    # --- dose penalties: medians and bootstrap CIs ------------------------
    summary = decomposition.fold_ratio_summary(ic50).set_index("quantity")
    c.check_exact("bootstrap settings", "bootstrap n",
                  T["bootstrap"]["n"], config.BOOTSTRAP_N)
    c.check_exact("bootstrap settings", "bootstrap seed",
                  T["bootstrap"]["seed"], config.BOOTSTRAP_SEED)
    for col, key in FOLD_KEYS.items():
        r = summary.loc[col]
        c.check("IC50 fold medians", f"{key} median",
                T["ic50_fold_medians"][key], r["median"], TOL_FOLD, "x")
        lo, hi = T["ic50_fold_ci95"][key]
        c.check("IC50 fold CI95", f"{key} CI lo", lo, r["ci_lo"], TOL_FOLD, "x")
        c.check("IC50 fold CI95", f"{key} CI hi", hi, r["ci_hi"], TOL_FOLD, "x")

    # --- crossings: state medians and profile counts ----------------------
    med = (cross.groupby(["threshold", "axis"])
           .agg(complete=("complete", "median"), partial=("partial", "median"),
                n=("complete", "size")).reset_index())
    for _, r in med.iterrows():
        t, a = f"{r['threshold']:g}", r["axis"]
        c.check("crossing medians", f"complete @ {t} / {a}",
                T["complete_at_crossing_medians"][t][a], r["complete"], TOL_FRACTION)
        c.check("crossing medians", f"partial  @ {t} / {a}",
                T["partial_at_crossing_medians"][t][a], r["partial"], TOL_FRACTION)
        c.check_exact("profile counts", f"n profiles @ {t} / {a}",
                      T["n_profiles"][t][a], int(r["n"]))

    # --- variance decompositions -----------------------------------------
    def shares(mask) -> dict | None:
        sub = dec[mask]
        return None if len(sub) != 1 else sub.iloc[0].to_dict()

    for state, tkey in (("complete", "complete_fraction_decomposition_pct"),
                        ("partial", "partial_fraction_decomposition_pct")):
        for t, per_axis in T[tkey].items():
            for a, exp in per_axis.items():
                got = shares((dec["quantity"] == f"{state}_fraction")
                             & (dec["axis"] == a)
                             & (np.isclose(dec["threshold"].astype(float), float(t))))
                for f in FACTORS:
                    c.check(f"ANOVA {state}_fraction", f"{t}/{a} {f}",
                            exp[f], None if got is None else got[f],
                            TOL_ANOVA_PP, "pp")

    for state, per_axis in T["dose_penalty_decomposition_pct"].items():
        for a, exp in per_axis.items():
            got = shares((dec["quantity"] == f"dose_penalty_{state}")
                         & (dec["axis"] == a))
            for f in FACTORS:
                c.check("ANOVA dose penalty", f"{state}/{a} {f}",
                        exp[f], None if got is None else got[f], TOL_ANOVA_PP, "pp")

    # --- conventions actually used ---------------------------------------
    n = T["normalization"]
    c.check_exact("conventions", "normalization divisor",
                  int(n["divisor"]), int(config.NF))
    c.check_exact("conventions", "pSTAT1 column matches",
                  1, int(config.RESP["pSTAT1"] == n["pSTAT1_col"]))
    c.check_exact("conventions", "pSTAT3 column matches",
                  1, int(config.RESP["pSTAT3"] == n["pSTAT3_col"]))
    c.check_exact("conventions", "partial bound >= complete bound (all rows)",
                  len(perdose), int((perdose["partial_bound"]
                                     >= perdose["complete_bound"]).sum()))

    ok = c.report(verbose=args.verbose)
    out = config.derived("validation_report.csv")
    c.frame().to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(config.REPO_ROOT)}")
    if not ok:
        print("\nValidation failed; inspect the reported discrepancies before changing tolerances.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
