#!/usr/bin/env python3
"""01: raw per-nucleus CSVs -> data/derived/perdose.csv

CONTRACT
    Input  : data/raw/<EXPID>/ShrinkedNuclei.csv for the four inhibitor-panel plates
             (KZ214, KZ198, KZ226, KZ223).  KZ195/KZ216 carry no inhibitor columns and
             are not used here.
    Output : data/derived/perdose.csv - one row per
             (experiment, cell, inhibitor, readout, dose) with all four inhibition
             measures, the gated cell count and the two state bounds.
             242 rows / 24 profiles expected.
             data/derived/exclusion_audit.csv - registry-vs-analysis exclusion diff,
             written ONLY when the two sets disagree.  They currently agree, so this
             file is absent on a normal run; absence is the pass condition.
    Runtime: ~5 s, ~570 MB peak (the four raw files total ~260 MB).

Usage:  python scripts/01_build_perdose.py [--registry-exclusions]

    --registry-exclusions  use metadata/experiment_registry.csv exclusion sets instead
                           of config.ANALYSIS_EXCLUSIONS.  Provided for auditing only;
                           it drops KZ198 entirely (see the ANALYSIS_EXCLUSIONS note in
                           config.py) and will NOT reproduce the published numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jakstat_dosegap import config, io, metrics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry-exclusions", action="store_true",
                    help="audit mode: use registry exclusion sets (drops KZ198)")
    args = ap.parse_args()

    audit = io.exclusion_conflicts()
    if not audit.empty:
        print("NOTE: registry and analysis exclusion sets differ:")
        print(audit.to_string(index=False))
        print("  -> see config.ANALYSIS_EXCLUSIONS for the provenance decision.\n")
        audit.to_csv(config.derived("exclusion_audit.csv"), index=False)

    overrides = io.registry_exclusions() if args.registry_exclusions else None

    frames = {}
    for exp_id in config.PANEL_EXPERIMENTS:
        excl = None if overrides is None else overrides.get(exp_id, [])
        frames[exp_id] = io.load_experiment(exp_id, exclude_wells=excl)
        print(f"  loaded {exp_id} ({config.CELL[exp_id]}): "
              f"{len(frames[exp_id]):>7,} nuclei after {len(excl) if excl is not None else len(config.ANALYSIS_EXCLUSIONS[exp_id])} well exclusions")

    perdose = metrics.perdose_table(frames)
    out = config.derived("perdose.csv")
    perdose.to_csv(out, index=False)

    n_profiles = perdose.groupby(["experiment", "inhibitor", "readout"]).ngroups
    print(f"\nwrote {out.relative_to(config.REPO_ROOT)}: "
          f"{len(perdose)} rows, {n_profiles} profiles, "
          f"{perdose['n'].sum():,} gated nuclei")
    if len(perdose) != 242 or n_profiles != 24:
        print(f"WARNING: expected 242 rows / 24 profiles, got {len(perdose)} / {n_profiles}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
