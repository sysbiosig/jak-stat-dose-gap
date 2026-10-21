# Figure source tables and reproducibility

This document maps the manuscript figures to their deposited inputs and executable
scripts. Files in `data/reference/` are read-only provenance and validation resources;
generated tables are written to `data/derived/figure_tables/`.

## Regenerating the figure inputs

```bash
python scripts/05_build_figure_tables.py -v
python scripts/12_figure_1.py
python scripts/13_figure_2.py
python scripts/14_figure_S1.py
python scripts/15_figure_S2.py
python scripts/16_compare_figures.py
```

The complete workflow, including the canonical analysis tables, Figures 3 and 4,
validation, and tests, is available through:

```bash
bash scripts/run_all.sh
```

## Figure map

| Manuscript figure | Deposited input | Executable output |
|---|---|---|
| Figure 1 | per-cell intensity table and raw reference distributions | `scripts/12_figure_1.py` -> `figures/Figure_1.png` |
| Figure 2 | observed dose points and fitted-curve table | `scripts/13_figure_2.py` -> `figures/Figure_2.png` |
| Figure 3 | `perdose.csv`, `crossings.csv`, `decompositions.csv` | `scripts/10_figure_3.py` -> `figures/Figure_3.png` |
| Figure 4 | `ic50_per_profile.csv`, fold summaries, decompositions | `scripts/11_figure_4.py` -> `figures/Figure_4.png` |
| Supplementary Figure S1 | observed dose points and fit summaries | `scripts/14_figure_S1.py` -> `figures/Figure_S1.png` |
| Supplementary Figure S2 | replicate IC50 table and replicate raw files | manuscript raster; source values deposited |

`scripts/15_figure_S2.py` produces the exploratory pSTAT1-minus-pSTAT3 selectivity
atlas, saved as `figures/Figure_S2_selectivity.png`. It is intentionally named
separately because Supplementary Figure S2 in the manuscript is the biological-replicate
concordance scatter.

## Reconstruction coverage

All 13 deposited per-figure source tables can be traced to the raw measurements. The
reconstruction report is written to
`data/derived/figure_tables/rederivation_report.csv` and is checked by
`tests/test_figure_tables.py`.

| Component | Agreement with deposited source material | Interpretation |
|---|---|---|
| Figure 1 per-cell intensities | numerical precision across all 28 condition groups | fully reconstructed |
| Complete-inhibition columns | numerical precision across dose-response plates | fully reconstructed |
| Partial-inhibition columns | numerical precision except where inherited reference metadata are incomplete | central state definitions preserved |
| Arithmetic-mean figure-source columns | residual up to 0.235 inhibition units | saturated reference constant is not fully recoverable from the deposited table metadata |
| KZ226 log-mean figure-source column | residual up to 0.0085 inhibition units | small saturated-reference residual |
| Supplementary Figure S2 values | all plotted IC50 values deposited | values reproducible; exact plotting layout is not reconstructed |

No fitted correction is applied to force agreement with a reference table. The pipeline
uses the fully specified response references in `src/jakstat_dosegap/config.py` and
reports measured residuals. The complete-inhibition IC50 values are recovered to
numerical precision, and the documented population-reference residuals do not alter the
ordering of the inhibition endpoints or the manuscript's dose-gap conclusions.

## Figure comparison

`scripts/16_compare_figures.py` compares generated rasters with the manuscript reference
PNGs where panel layouts are directly comparable. It writes
`data/derived/figure_tables/figure_pixel_comparison.csv`.

| Figure | Comparison result |
|---|---|
| Figure 1 | generated directly from the deposited per-cell source data |
| Figure 2 | matching panel content and dimensions; 88.4% of pixels within 8 grey levels |
| Figure 3 | generated directly from deposited tables; manuscript and reference copies match pipeline output |
| Figure 4 | generated directly from deposited tables; manuscript and reference copies match pipeline output |
| Supplementary Figure S1 | matching panel content and dimensions; 90.2% of pixels within 8 grey levels |
| Supplementary Figure S2 | source values checked directly rather than by raster comparison |

Pixel differences in dense line art are dominated by font rasterization, antialiasing,
and sub-pixel curve placement. Structural checks therefore also verify panel counts,
panel order, axis scales, colors, and the values printed in the figures.

## Supplementary Figure S2

`data/reference/replicate_ic50_shown_vs_replicate.csv` contains 48 rows: 24 profiles for
the replicate shown in the main figures and 24 profiles for the independent replicate.
The four independent-replicate experiments are deposited under `data/raw/` as KZ191,
KZ203, KZ222, and KZ227.

For the arithmetic mean, log-mean, and complete-inhibition endpoints, log10-scale Pearson
correlations are 0.9255, 0.9524, and 0.8722, respectively, which round to the values in
the figure caption. The source table therefore supports direct reconstruction of the
plotted quantities. Panel geometry and styling are retained in the deposited manuscript
raster because the notebook that assembled that raster is not part of the deposit.

Further replicate details and the sensitivity analysis excluding extrapolated IC50
estimates are provided in `docs/BIOLOGICAL_REPLICATE.md`.

## Read-only reference boundary

Nothing in the pipeline writes to `data/reference/`, and figure scripts do not obtain
manuscript result values from the validation targets. Analysis outputs are regenerated
under `data/derived/`; reference tables and rasters are used only to document provenance
and verify the generated results.
