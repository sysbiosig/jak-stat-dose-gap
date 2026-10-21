# Analysis pipeline

Raw single-cell CSVs in, derived tables out. Reproduces the values reported in
"Single-cell profiling quantifies the dose gap between population-average and complete pY-STAT1/3 suppression by JAK inhibitors".

## Run it

```bash
bash scripts/run_all.sh        # all stages, validation, tests, figures (~90-145 s)
```

Or step by step:

| Script | Reads | Writes |
|---|---|---|
| `01_build_perdose.py` | `data/raw/<EXPID>/ShrinkedNuclei.csv` | `perdose.csv` (242 rows); `exclusion_audit.csv` only if registry and analysis exclusion sets disagree |
| `02_fit_ic50.py` | `perdose.csv` | `ic50_per_profile.csv` (24), `ic50_fold_summary.csv` |
| `03_crossings.py` | `perdose.csv` | `crossings.csv` (140), `crossing_medians.csv` |
| `04_decompositions.py` | `crossings.csv`, `ic50_per_profile.csv` | `decompositions.csv` (16), `dose_penalty_ratios.csv` |
| `99_validate_derived.py` | all of the above + `data/reference/published_values.json` | `validation_report.csv` |
| `10_figure_3.py` | `perdose.csv`, `crossings.csv`, `decompositions.csv` | `figures/Figure_3.png` (10 panels) |
| `11_figure_4.py` | `ic50_per_profile.csv`, `ic50_fold_summary.csv`, `decompositions.csv` | `figures/Figure_4.png` (8 panels) |

All derived tables land in `data/derived/` (gitignored — rebuild rather than commit).
A clean rebuild is byte-identical: verified by deleting `data/derived/` and rerunning.

## Figures

Figures 3 and 4 are deterministic scripts over the derived tables — they read no raw data
and no manuscript source table. `src/jakstat_dosegap/figstyle.py` is their source of truth
for the palette, markers, panel-letter stamping and axis conventions. The same-layout
reconstruction scripts for Figures 2, S1 and the S2 selectivity atlas intentionally pin
reference-raster colors locally so their outputs remain comparable to those rasters.

**Block order is load-bearing.** Both figures put **partial** inhibition in the top block
and **complete** in the bottom. This order reflects the manuscript narrative:
partial inhibition agrees with the population average, complete departs from it, so the
reader meets the agreement before the discrepancy. `tests/test_figures.py` pins the
order.

Determinism: the only stochastic element is strip-plot x jitter, seeded per figure
(seed 3 for Figure 3, 5 for Figure 4) and consumed by the jittered panels in layout
order — so re-ordering those calls changes the jitter even though the data are unchanged.
Two builds in one process are byte-identical (pinned by test).

Agreement with the manuscript rasters in `docs/published_figures/` (mean absolute pixel difference on a common
1200-px-wide resampled grid, 0–255 scale):

| Figure | Dimensions (generated / reference) | MAD | pixels >10 | pixels >32 | Pearson *r* |
|---|---|---|---|---|---|
| Figure 3 | 4060×1887 / 4060×1887 | 0.78 (0.31%) | 2.3% | 0.30% | 0.9924 |
| Figure 4 | 3335×2192 / 3334×2192 | 0.96 (0.38%) | 2.3% | 0.99% | 0.9820 |

The residual is uniform across all panels (per-panel MAD 0.65–1.13), which is the
signature of font rasterisation and antialiasing rather than a structural difference.
Figure 4 is 1 px wider than the reference raster — `bbox_inches="tight"` rounding, not a layout
change. Structure was verified independently: panel counts (10 / 8), row-major letter
order (A–J / A–H), panel-letter positions within 1–5 px of the reference, and the red partial
header above the blue complete header in both.

## Package layout

`src/jakstat_dosegap/` — each module's docstring states its contract; read that
rather than the body.

| Module | Responsibility |
|---|---|
| `config.py` | every numerical convention, label map, path and palette. Single source of truth. |
| `io.py` | raw CSV → tidy per-nucleus frame; CR line endings, missing inhibitor columns, well exclusions |
| `metrics.py` | the two single-cell state bounds and the four inhibition measures |
| `fitting.py` | 4PL fit, IC50 extraction, the four dose-penalty fold-ratios |
| `crossings.py` | the validated crossing rule |
| `decomposition.py` | type-II ANOVA variance shares, bootstrap CIs |
| `figstyle.py` | the locked palette, markers, panel letters, axis conventions, text-collision checker |

## The analysis contract

Conventions are not free parameters — they are the manuscript analysis definitions,
and changing any of them breaks reproduction.

- Intensities: raw CellProfiler value × 65535 → counts. `pSTAT1 = Alexa488`,
  `pSTAT3 = Alexa555`.
- Per-cell gate: `1 < signal < 1e4`; groups need ≥ 10 gated cells.
- **Complete bound** = 95th percentile of unstimulated cells. A cell below it is
  indistinguishable from unstimulated.
- **Partial bound** = `max(complete_bound, 5th percentile of saturated cells)`. The
  clamp is load-bearing: when the two control distributions overlap, an unclamped 5th
  percentile falls *below* the complete bound and "partial" would become the stricter
  of the two.
- Bounds and population references are computed **per (experiment, readout)** from
  that plate's own controls — absolute intensities are not comparable across stainings.
- Four measures per dose group: arithmetic-mean and log-mean population inhibition
  (both normalised so 0 = saturated, 1 = unstimulated), plus the complete and partial
  state fractions. Measures are **not** clipped to [0,1]; clipping would bias the fit.
- 4PL: `scipy.optimize.curve_fit`, bounded, `maxfev=20000`. IC50 bounds are scaled to
  each profile's own dose window (`dose.min()/10 … dose.max()×10`).
  Orientation: with `hill > 0`, `top` is the low-dose plateau and `bottom` the
  high-dose one, so a rising inhibition curve has `bottom > top`. Hill is bounded
  `[0.2, 5.0]` — strictly positive.
- **Crossing rule**: fit a 4PL to dose → population inhibition, take the **first
  measured dose** at which the **fitted** curve reaches the threshold, and read the
  **observed** state fractions at that dose. Interpolating instead does *not* reproduce
  the reported values; the snap-to-measured-dose discreteness is part of the
  definition. Profiles whose fitted curve never reaches the threshold drop out, which
  is why the profile count falls from 24 to 22–23 at threshold 0.8.
- ANOVA: type II (the design is unbalanced, so type-I shares would be order-dependent),
  additive `cell + inhibitor + readout`, no interactions (one observation per cell, so
  interaction variance lands in "unexplained"). The `readout` factor is reported under
  the manuscript label **target**.
- Dose penalties are decomposed on the **log10** scale. This is not cosmetic:
  decomposing raw ratios shifts the cell-line share of the complete-state penalty by
  more than 20 percentage points. `tests/test_reproduction.py` pins the disagreement so
  the scale cannot be switched silently.
- Bootstrap: percentile bootstrap of the **median** over profiles, n=10000, seed=7.
  The median rather than the mean because fold-ratios are right-skewed.

## Well exclusions

`config.ANALYSIS_EXCLUSIONS` holds the sets used here, taken from the per-plate exclusion
`scripts_InhibitionPlotting_*.R` scripts that produced the manuscript dose-response
panels. The same sets are recorded in `metadata/experiment_registry.csv`.

KZ198 requires particular care because the deposited filtration records contain an
additional six-well candidate set:

| Source | Set | n |
|---|---|---|
| Rejected six-well set | `A12 B09 B10 B11 B12 E02` | 6 |
| Analysis set | `A12 B05 B09 E02` | 4 |

The six-well set removes both KZ198 unstimulated wells (B11 and B12) and the only
surviving inhibitor-free saturated well (B10, because B09 is excluded by both sets).
Neither reference distribution can then be defined, and KZ198 drops out of the analysis.
The four-well set preserves the required controls and is therefore the analysis set.

The registry records this four-well decision (`n_excluded=4`), and
`metadata/exclusion_resolution_audit.csv` records the supporting plate-level evidence.
`io.exclusion_conflicts()` provides a programmatic check that the registry and analysis
sets agree. `scripts/01_build_perdose.py` writes `data/derived/exclusion_audit.csv` only
if a conflict is detected.

## Validation

`scripts/99_validate_derived.py` compares 105 quantities against
`data/reference/published_values.json` and prints expected vs actual for each.
Nothing in `src/jakstat_dosegap/` reads that JSON, so no check can be satisfied by the
pipeline having been told the answer.

Tolerances are rounding allowances for values reported to 2–3 decimals, **not** fit
windows: 0.01 absolute on fractions, 0.05 on fold-ratios, 1 percentage point on ANOVA
shares, exact on integer counts. A failure is a discrepancy to diagnose — do not widen
a tolerance to force a pass.

**105 of 105 checks pass.** Largest deviations are 0.005 on fold medians,
0.0005 on crossing medians, and 0.49 pp on ANOVA shares (the manuscript
complete-fraction and dose-penalty shares are quoted to whole percent, so sub-0.5 pp
deviations are pure rounding).

## Tests

```bash
python -m pytest -q                 # 139 tests, ~30 s
python -m pytest -q -m "not slow"   # skip the raw-data reload (135 tests, ~22 s)
```

`tests/test_pipeline.py` checks each module's logic against hand-constructed inputs
whose answer is known analytically — so a regression names its own culprit rather than
just failing to reproduce a reported number. `tests/test_reproduction.py` checks the
real derived tables against the manuscript values and skips itself if they are absent.
`tests/test_figures.py` (27 tests) pins the things a figure gets wrong *silently*: block
order, panel count and kind, log-scaled ratio axes, the four reported dose ratios, the
locked palette, determinism, and the three quantitative colour-separation claims that
`figstyle.py` documents. The palette and style tests are pure and run without derived
data; the figure-building tests skip themselves when `data/derived/` is absent.

## Why three Figure S1 profiles show no separate partial curve

`07_threshold_separation.py` exists to answer a question that looks like a bug.

In Figure S1 the three KZ223 / U937 / pSTAT3 profiles (baricitinib, ruxolitinib,
tofacitinib) appear to have **no partial-inhibition curve**. They do. It is drawn
exactly underneath the complete-inhibition curve.

The two bounds are

    complete = 95th percentile of the UNSTIMULATED cells
    partial  = max(complete, 5th percentile of the SATURATED cells)

On that plate and that readout the saturated 5th percentile (12.71) falls *below*
the unstimulated 95th percentile (13.81). The `max()` clamp in
`metrics.reference_bounds` fires, both bounds become 13.81, the two measures count
the same cells, and `partial == complete` at every dose to machine precision.

It is a property of the (experiment, readout) pair, not of the inhibitor -- which
is why all three inhibitors on that plate are affected and no others anywhere are.

Why that plate: KZ223 pSTAT3 has the weakest stimulation in the panel, a median
saturated/unstimulated fold of **3.07x** against 5.1-10.3x for pSTAT3 elsewhere
and up to 129x for pSTAT1. 6.62% of its saturated cells fall below the
unstimulated 95th percentile, versus 0.00-2.63% everywhere else. The overall
separation is still good (AUC 0.969); it is specifically the tails that cross.

The effect is not a sampling accident -- bootstrap over the reference cells gives
P(clamp fires) = 1.000, with non-overlapping 95% CIs ([13.52, 14.41] for the
unstimulated p95, [12.44, 13.03] for the saturated p05). But it *is* marginal by
construction: the saturated p10 is 15.93, comfortably above the bound, so the
inversion hangs on an 8% gap between two adjacent percentiles.

**Reference-cell limitation.** KZ223's unstimulated reference is a single well
(H03, n=948); H04 and H12 appear in the plate design but contribute no gated
cells. KZ198 likewise has only one surviving saturated well. The affected plate is
therefore also the one with the thinnest reference. The consequence is documented
above and pinned by tests; it is a property of the deposited data, not a defect.

Pinned by `tests/test_threshold_separation.py` (7 tests), which asserts the
collapse stays confined to KZ223 / pSTAT3, that exactly 3 profiles are affected,
that the bootstrap probability remains 1.0, and that `reference_bounds` still
clamps on synthetic inverted input.
