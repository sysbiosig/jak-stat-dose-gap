# Verification

The verification results below come from `bash scripts/run_all.sh` using the deposited
raw measurements. Derived tables and figures are rebuilt by the workflow; manuscript
quantities are not entered into the analysis code.

Environment used for the recorded run: macOS 14 (arm64), CPython 3.11.15, numpy 2.4.6,
pandas 2.3.3, scipy 1.17.1, matplotlib 3.11.0, and statsmodels 0.14.6. Dependencies are
listed in `requirements.txt`.

Recorded clean-run performance: 90--145 s wall time, approximately 693 MB peak memory,
and 766 MB of raw input.

## Reported values

`scripts/99_validate_derived.py` compares generated quantities with the 105 values
reported in the manuscript and its source tables. Validation targets are stored in
`data/reference/published_values.json`; no module under `src/` imports that file.

| Group | Checks | Largest deviation | Tolerance |
|---|---:|---:|---:|
| IC50 fold medians | 4 | 0.0042-fold | 0.05-fold |
| IC50 fold 95% confidence-interval bounds | 8 | 0.0030-fold | 0.05-fold |
| Crossing medians | 12 | 0.00049 | 0.01 |
| Profile counts | 6 | exact | exact |
| ANOVA, complete-inhibition occupancy | 24 | 0.487 percentage points | 1.0 percentage point |
| ANOVA, partial-inhibition occupancy | 24 | 0.0050 percentage points | 1.0 percentage point |
| ANOVA, dose penalty | 16 | 0.385 percentage points | 1.0 percentage point |
| Structure, fitting conventions, and bootstrap settings | 11 | exact | exact |

All 105 checks pass. Numerical tolerances reflect the precision with which values are
reported in the manuscript; they are not fitting windows.

| Quantity | Manuscript | Pipeline output |
|---|---:|---:|
| complete/log-mean IC50 ratio | 2.4 (95% CI 1.9--3.1) | 2.39 (95% CI 1.86--3.10) |
| complete/arithmetic-mean IC50 ratio | 7.4 (95% CI 3.5--16) | 7.42 (95% CI 3.50--16.12) |
| partial/arithmetic-mean IC50 ratio | 1.7 | 1.72 (95% CI 1.55--2.93) |
| partial/log-mean IC50 ratio | 0.8 | 0.80 (95% CI 0.52--1.74) |
| complete occupancy at log-mean inhibition = 0.8 | 0.63 | 0.628 |
| complete occupancy at arithmetic-mean inhibition = 0.8 | 0.24 | 0.244 |
| evaluable profiles at benchmarks 0.2/0.5/0.8 | 24/24, 24/23, 23/22 | identical |

## Figures

Figures 1, 2, 3, 4, and Supplementary Figure S1 have executable generation scripts.
Supplementary Figure S2 is supplied as a manuscript raster together with its plotted
IC50 table and independent-replicate raw measurements.

| Figure | Verification |
|---|---|
| Figure 1 | generated from deposited per-cell intensities and reference distributions |
| Figure 2 | matching panel structure and dimensions; 88.4% of pixels within 8 grey levels |
| Figure 3 | generated directly from deposited tables; manuscript and reference copies match pipeline output |
| Figure 4 | generated directly from deposited tables; manuscript and reference copies match pipeline output |
| Supplementary Figure S1 | matching structure and dimensions; 90.2% of pixels within 8 grey levels |
| Supplementary Figure S2 | plotted values recomputed directly from the deposited IC50 table |

Raster differences in line plots are sensitive to font rendering, antialiasing, and
sub-pixel placement. The test suite therefore also checks panel counts, panel order, axis
scales, colors, and printed summary values. Detailed source-table coverage is provided in
`docs/FIGURE_TABLES.md`.

For Supplementary Figure S2, log10-scale Pearson correlations recompute to 0.9255,
0.9524, and 0.8722 for arithmetic-mean, log-mean, and complete-inhibition IC50,
respectively (24 paired profiles). These values round to 0.93, 0.95, and 0.87 as shown in
the caption. See `docs/BIOLOGICAL_REPLICATE.md` for experiment pairing and sensitivity to
extrapolated fits.

## Raw-data integrity

Ten CellProfiler exports are deposited through Git LFS. SHA256 checksums and file sizes
are recorded in `metadata/raw_data_manifest.json`; experiment roles and excluded wells
are recorded in `metadata/experiment_registry.csv`.

| Experiment | Cell line | Role |
|---|---|---|
| KZ214 | THP-1 | dose-response panel |
| KZ198 | U937 | dose-response panel |
| KZ226 | THP-1 | dose-response panel |
| KZ223 | U937 | dose-response panel |
| KZ195 | U937 | stimulation distributions |
| KZ216 | U937 | stimulation distributions |
| KZ191 | THP-1 | independent replicate of KZ214 |
| KZ203 | U937 | independent replicate of KZ198 |
| KZ222 | THP-1 | independent replicate of KZ226 |
| KZ227 | U937 | independent replicate of KZ223 |

Approximately 215,000 nuclei enter the 24 dose-response profiles after channel-specific
intensity gating. Because pSTAT1 and pSTAT3 are gated separately, their gated counts must
not be added as independent cells.

## Tests

The repository contains 139 tests:

- unit tests for metrics and four-parameter logistic fitting;
- regression tests for the manuscript quantities;
- provenance tests for exclusions, source-table reconstruction, and figure structure.

Run them with:

```bash
python -m pytest -q
```
