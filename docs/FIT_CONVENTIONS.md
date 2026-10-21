# Analysis and fitting conventions

This document defines the canonical calculations used for the manuscript results. The
implementation is centralized in `src/jakstat_dosegap/`, and the numerical settings are
recorded in `src/jakstat_dosegap/config.py`.

## Response references

Reference distributions are computed separately for each experiment and phospho-STAT
readout.

- The fully stimulated control defines zero inhibition for the population measures.
- The non-stimulated control defines complete population inhibition.
- Arithmetic-mean inhibition uses the mean raw response within each dose group.
- Log-mean inhibition uses the mean of log10-transformed single-cell responses within
  each dose group.
- The complete-inhibition threshold is the 95th percentile of the non-stimulated
  distribution.
- The partial-inhibition threshold is the larger of the complete-inhibition threshold
  and the 5th percentile of the fully stimulated distribution.

Signals are gated to `1 < intensity < 1e4` before aggregation. Population inhibition is
not clipped to `[0, 1]`, because clipping would change the fitted dose-response curve.

## Dose-response fitting

Each profile is fitted with a bounded four-parameter logistic curve using
`scipy.optimize.curve_fit` and `maxfev=20000`. The Hill coefficient is constrained to
`[0.2, 5.0]`, and the IC50 search interval spans one decade below and above the measured
dose range. These settings are applied identically to arithmetic-mean inhibition,
log-mean inhibition, partial-inhibition occupancy, and complete-inhibition occupancy.

Fold differences are calculated per inhibitor-cell-line-readout profile. Confidence
intervals are percentile-bootstrap intervals for the median across profiles, using
10,000 resamples and seed 7. Variance decomposition of dose ratios is performed on the
log10 scale.

## Threshold crossing

Figure 3 relates a population-inhibition benchmark to observed state occupancy. For each
profile and benchmark (0.2, 0.5, or 0.8), the procedure is:

1. Fit the population-inhibition curve.
2. Identify the lowest measured dose at which the fitted curve reaches the benchmark.
3. Read the observed state occupancy at that measured dose.

Profiles whose fitted curve does not reach a benchmark are omitted for that benchmark.
The procedure intentionally reports measurements at tested concentrations rather than
interpolated state fractions between doses.

## Reference files

Files under `data/reference/` are read-only validation and figure-provenance resources.
The analysis pipeline does not use them to calculate its results. The canonical derived
tables are generated from `data/raw/` by `scripts/01_build_perdose.py` through
`scripts/04_decompositions.py` and checked by `scripts/99_validate_derived.py`.

Some deposited figure-source tables lack sufficient metadata to reconstruct every
population reference constant byte for byte. Reconstruction residuals are therefore
reported rather than corrected by fitted constants. Complete-inhibition values, which
drive the central dose-gap result, are recovered to numerical precision. The largest
documented residuals affect the arithmetic-mean figure-source columns and one KZ226
log-mean column; their magnitude and figure-level consequences are summarized in
`docs/FIGURE_TABLES.md`.
