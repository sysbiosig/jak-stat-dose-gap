# Single-cell profiling quantifies the dose gap between population-average and complete pY-STAT1/3 suppression by JAK inhibitors

Reproducible data and analysis companion for the manuscript *"Single-cell profiling quantifies the dose gap between population-average and complete pY-STAT1/3 suppression by JAK inhibitors"*: raw per-nucleus exports,
plate metadata, source tables, executable figure scripts, manuscript files and tests.
The repository is designed so the reported results can be regenerated and audited, not
just downloaded as static tables.

**Raw experimental CSVs in, derived tables and manuscript figures out — one command,
about two minutes.** No manual steps, no hand-edited intermediates, no configuration.

```bash
bash scripts/run_all.sh
```

## What you get

- **Roughly 215,000 gated nuclei** per channel: nuclear pY-STAT1 and pY-STAT3
  reported as per-nucleus CellProfiler image-analysis exports. These are the source
  measurements used by the deposited analysis pipeline, not the original microscopy images.
- **24 complete dose-response profiles**: 6 JAK/TYK2 inhibitors x 2 myeloid cell lines
  (THP-1, U937) x 2 phospho-targets, 10-10,000 nM (baricitinib in U937 adds 0.5 nM).
- **A pipeline that rebuilds Figures 1, 2, 3, 4 and S1 from those raw files**, plus every
  derived table behind the numbers in the text.
- **139 tests** that check the regenerated values against the values reported in the manuscript, so you know
  immediately whether a change you made altered a result.

If you want the measurements rather than the pipeline, go straight to
`data/derived/ic50_per_profile.csv` (IC50 per profile for all four measures) and
`data/derived/perdose.csv` (the per-dose response table). Both are rebuilt by the command
above; neither needs to be trusted on faith, because `99_validate_derived.py` re-checks
105 reported values on every run.

## Getting started

Raw CSVs are stored with Git Large File Storage. **Install it before cloning** — without
it you get 130-byte pointer files instead of data, and the first stage fails with a CSV
tokenizing error.

```bash
brew install git-lfs                        # or: apt install git-lfs
git lfs install
git clone https://github.com/sysbiosig/jak-stat-dose-gap.git
cd jak-stat-dose-gap
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_all.sh                     # ~90-145 s, writes data/derived/ and figures/
python -m pytest -q                         # 139 tests, ~30 s
```

Sanity check on the clone: `data/raw/KZ214/ShrinkedNuclei.csv` should be about 30 MB. If
it is 130 bytes, LFS was not active — run `git lfs pull`.

Requirements: Python 3.11+, 766 MB disk for the raw data, ~700 MB peak RSS. No GPU, no
cluster, no licensed software.

No `pip install -e .` and no `PYTHONPATH` are needed. Scripts and the test suite locate
`src/` themselves (`conftest.py` at the repository root does it for pytest), so the two
commands above work in a bare clone from any working directory.

## The result, in three numbers

Population averages and single-cell state occupancy report different things about
inhibitor efficacy:

- **Partial-inhibition occupancy**, the fraction leaving the stimulated range, has
  an apparent IC50 close to the log-mean IC50 (median ratio **0.80x**).
- **Complete-inhibition occupancy**, the fraction entering the non-stimulated range,
  has an apparent IC50 a median **2.4x** above the log-mean IC50 (95% CI 1.9-3.1)
  and **7.4x** above the arithmetic-mean IC50 (CI 3.5-16).
- At the lowest tested dose where fitted population inhibition reaches or exceeds 80%,
  a median **37%** of cells remain above the non-stimulated pY-STAT range for the
  log-mean benchmark, and **76%** for the arithmetic-mean benchmark.

The fold differences compare fitted endpoint midpoints, not doses that suppress every
cell. Cell line has the largest identifiable association with the complete-inhibition
dose ratio within this panel; 37-39% of its variability is unexplained by the three
factors. Batch is confounded with cell line and inhibitor panel, so these associations
are descriptive. Profile-bootstrap intervals do not estimate biological-replicate
uncertainty. The numerical results are regenerated and checked by the pipeline.

## What is in data/raw

Ten CellProfiler exports, in three roles:

| role | experiments | used for |
|---|---|---|
| dose-response panels | KZ214, KZ198, KZ226, KZ223 | the 24 profiles behind Figures 2, 3, 4, S1 |
| stimulation histograms | KZ195, KZ216 | Figure 1 cytokine-dose/reference distributions |
| biological replicate | KZ191, KZ203, KZ222, KZ227 | replicate concordance, Supplementary Figure S2 |

The replicate plates pair one-to-one with the panels (KZ191<->KZ214, KZ203<->KZ198,
KZ222<->KZ226, KZ227<->KZ223) and are the source measurements behind Supplementary Figure S2,
which reports IC50 concordance between the two replicates. The pipeline's figure stages
analyse the shown replicate only, so these four plates are not read by `run_all.sh`; their
concordance values are tabulated in
`data/reference/replicate_ic50_shown_vs_replicate.csv` and documented in
[docs/BIOLOGICAL_REPLICATE.md](docs/BIOLOGICAL_REPLICATE.md).

Every plate's role, checksum and excluded wells is in `metadata/experiment_registry.csv`.

## Reusing this for something else

The measurements are more general than the question the paper asks. Three entry points,
in increasing depth:

1. **Per-profile IC50s** — `data/derived/ic50_per_profile.csv`, 24 profiles x 4 measures
   with fit parameters and quality flags. Enough for a potency comparison across
   inhibitors or cell lines.
2. **Per-dose response** — `data/derived/perdose.csv`, 242 rows. The population summaries
   and single-cell state fractions at every measured dose. Enough to define your own
   efficacy metric and refit.
3. **Per-nucleus intensities** — `data/raw/`. Cell-level distributions, so you can set
   different gates, different reference bounds, or ask distributional questions the paper
   does not.

`src/jakstat_dosegap/` is importable and each module opens with a CONTRACT docstring
stating inputs with units, outputs and invariants — read those rather than the bodies.
`config.py` holds every threshold and convention in one place, so changing the gating or
the reference percentiles is a one-line edit that the test suite will immediately tell
you about.

## Pipeline stages

| script | does | output |
|---|---|---|
| `01_build_perdose.py` | load raw, gate, normalize, aggregate per dose | `perdose.csv` (242 rows, 24 profiles) |
| `02_fit_ic50.py` | 4PL fit + IC50 for all four measures | `ic50_per_profile.csv` |
| `03_crossings.py` | threshold crossings at 0.2/0.5/0.8, both axes | `crossings.csv` |
| `04_decompositions.py` | type-II ANOVA variance shares, bootstrap CIs | `decompositions.csv` |
| `05_build_figure_tables.py` | rebuild the per-figure source tables from raw | `data/derived/figure_tables/` |
| `07_threshold_separation.py` | reference-distribution separation diagnostics | `threshold_separation_diagnostic.csv` |
| `08_permutation_test.py` | label-permutation test on the cell-line variance share | `permutation_tests.csv` |
| `09_extrapolation_sensitivity.py` | re-tests the headline on bracketed-only estimates | `extrapolation_sensitivity.csv` |
| `10_figure_3.py` | Figure 3, 10 panels | `figures/Figure_3.png` |
| `11_figure_4.py` | Figure 4, 8 panels | `figures/Figure_4.png` |
| `12..15_figure_*.py` | Figures 1, 2, S1, and the S2 selectivity atlas | `figures/*.png` |
| `16_compare_figures.py` | pixel comparison against the manuscript reference PNGs | comparison CSV |
| `99_validate_derived.py` | verify the 105 reported values | `validation_report.csv` |

Figure-to-script map: Fig 1 -> 12, Fig 2 -> 13, **Fig 3 -> 10**, **Fig 4 -> 11**,
Fig S1 -> 14. Supplementary Figure S2 is a replicate-concordance scatter whose plotted
values and replicate raw files are deposited; script 15 produces a different figure, the
selectivity atlas, under its own name `Figure_S2_selectivity.png`.

## The analysis in one screen

Four measures are computed per (experiment, cell line, inhibitor, target, dose) from
per-nucleus intensities, multiplied by 65535 and gated to `1 < signal < 1e4`:

- **arithmetic-mean** and **log-mean** population inhibition — the two population summaries
- **partial inhibition** — fraction of cells below the fully stimulated range
  (bound: `max(complete_bound, 5th percentile of fully stimulated cells)`)
- **complete inhibition**: fraction within the non-stimulated response range
  (bound: 95th percentile of non-stimulated control cells)

pY-STAT1 is read on Alexa488, pY-STAT3 on Alexa555. Occupancy counts cells strictly
below each bound and is not baseline-corrected; the non-stimulated reference has
approximately 95% complete-inhibition occupancy by construction.

**One rule is easy to get wrong.** To find the state occupancy at a given level of
population inhibition: fit the 4PL to dose -> population inhibition, take the *first
measured dose* at which the *fitted* curve reaches the target, and read the *observed*
state fraction at that dose. Interpolating between doses gives different numbers. See
`src/jakstat_dosegap/crossings.py`.

## Repository layout

```
data/raw/<EXPID>/ShrinkedNuclei.csv    10 per-nucleus CellProfiler exports (LFS)
data/derived/                          everything the pipeline computes (gitignored)
data/reference/published_tables/       the 13 deposited per-figure source tables
                                       (read-only reference files)
data/reference/published_values.json   the 105 manuscript values validated against
src/jakstat_dosegap/                   the pipeline package
scripts/                               numbered entry points, run in order
tests/                                 139 tests
metadata/
  experiment_registry.csv              per-plate roles, exclusions, checksums
  raw_data_manifest.json               the unique raw files with sha256 and sizes
  exclusion_resolution_audit.csv       how the KZ198/KZ214 exclusion sets were resolved
  filtration_docs/                     the per-plate exclusion documents
docs/                                  pipeline contract and comparison reports
docs/published_figures/                manuscript PNGs used as comparison targets
figures/                               regenerated output (gitignored)
manuscript/                            LaTeX source, compiled PDF, figures
```

## Documentation

- **[docs/PIPELINE.md](docs/PIPELINE.md)** — the stage-by-stage contract: each script's
  inputs and outputs, the normalization and gating conventions, and the well-exclusion
  decisions. Start here to modify the analysis.
- **[VERIFICATION.md](VERIFICATION.md)** — 105/105 reported values recovered from the pipeline,
  with raw-data checks and figure-level pixel agreement, including reconstruction residuals.
- **[docs/FIGURE_TABLES.md](docs/FIGURE_TABLES.md)** — source-table coverage, measured
  reconstruction residuals, and figure-specific reproducibility limits.
- **[docs/FIT_CONVENTIONS.md](docs/FIT_CONVENTIONS.md)** — the canonical normalization,
  fitting, crossing, and uncertainty conventions used for the manuscript results.
- **[docs/BIOLOGICAL_REPLICATE.md](docs/BIOLOGICAL_REPLICATE.md)** — the replicate plates:
  pairing, exclusions, and the concordance values with a sensitivity check.

## The manuscript

`manuscript/` holds the LaTeX source, the compiled PDF and the six figures it embeds, so
the text and the analysis travel together. `bash manuscript/build.sh` recompiles it.

## Conventions if you extend this

- Every module and script opens with a **CONTRACT** docstring: inputs with units, outputs,
  invariants.
- `fig.savefig(...)`, never `plt.savefig(...)`. Figures are deterministic; jitter is seeded.
- Color conventions are explicit in `src/jakstat_dosegap/config.py`,
  `src/jakstat_dosegap/figstyle.py` and the individual figure scripts. Blue denotes
  complete-inhibition state occupancy and red denotes partial-inhibition state occupancy;
  Figures 2, S1 and the S2 selectivity atlas retain their locked curve colors for
  reproducibility.
- `data/raw/` and `data/reference/` are read-only. Nothing in the pipeline writes to them.

## Citation

If you use these data or code, please cite the manuscript. Grant: National Science Centre
Poland 2020/39/B/NZ2/03259.
