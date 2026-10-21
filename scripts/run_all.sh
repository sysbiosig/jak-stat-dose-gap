#!/usr/bin/env bash
# Rebuild every derived table from the raw single-cell CSVs and validate the result.
#
# CONTRACT
#   Input  : data/raw/<EXPID>/ShrinkedNuclei.csv  (6 of the 10 deposited experiments:
#            the 4 dose-response panels KZ214/KZ198/KZ226/KZ223 plus the 2 stimulation
#            source-table plates KZ195/KZ216 used by Figure 1; the 4 replicate experiments
#            KZ191/KZ203/KZ222/KZ227 are deposited but read by no stage)
#   Output : data/derived/*.csv, then figures/Figure_{1,2,3,4,S1}.png and
#            figures/Figure_S2_selectivity.png
#   Exit 0 iff all 105 published-value checks pass and every figure builds clean.
#   Runtime: ~90-145 s total (the permutation test is ~55 s of it), dominated by 01 (reads ~260 MB of raw CSVs).
#
# Usage:  bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

for step in 01_build_perdose 02_fit_ic50 03_crossings 04_decompositions; do
    echo "=== ${step} ==="
    python "scripts/${step}.py"
done

echo "=== 99_validate_derived ==="
python scripts/99_validate_derived.py

# Figures consume ONLY data/derived/*.csv, so they run after validation: a figure built
# from unvalidated tables would look plausible and be wrong.
for step in 10_figure_3 11_figure_4; do
    echo "=== ${step} ==="
    python "scripts/${step}.py"
done

# PER-FIGURE SOURCE TABLES (Figures 1, 2, S1, S2).  Separate from the block above because these
# figures come from the published-figure conventions rather than the validated
# pipeline: 05 re-derives their inputs from raw and REPORTS where the published-figure
# conventions cannot be reproduced, without failing the build for it.  See
# docs/FIGURE_TABLES.md for what does and does not reproduce, and why.
echo "=== 05_build_figure_tables ==="
python scripts/05_build_figure_tables.py

echo "=== 07_threshold_separation (why 3 Figure S1 profiles show no partial curve) ==="
python scripts/07_threshold_separation.py

echo "=== 08_permutation_test (cell-line significance; ~1 min) ==="
python scripts/08_permutation_test.py

echo "=== 09_extrapolation_sensitivity (bracketed-only headline check; ~2 s) ==="
python3 scripts/09_extrapolation_sensitivity.py

for step in 12_figure_1 13_figure_2 14_figure_S1 15_figure_S2; do
    echo "=== ${step} ==="
    python "scripts/${step}.py"
done

echo "=== 16_compare_figures ==="
python scripts/16_compare_figures.py

echo "=== tests ==="
python -m pytest -q
