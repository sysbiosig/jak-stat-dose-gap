# Biological replicates

The imaging experiments were performed in two independent biological replicates. The
main figures use four primary-replicate plates, and Supplementary Figure S2 compares their
IC50 estimates with four independently acquired plates.

| Cell line | Inhibitors | Primary replicate | Independent replicate |
|---|---|---|---|
| THP-1 | solcitinib, BMS-911543, cerdulatinib | KZ214 | KZ191 |
| U937 | solcitinib, BMS-911543, cerdulatinib | KZ198 | KZ203 |
| THP-1 | baricitinib, ruxolitinib, tofacitinib | KZ226 | KZ222 |
| U937 | baricitinib, ruxolitinib, tofacitinib | KZ223 | KZ227 |

Each pair has the same cell line, inhibitor set, and pSTAT1/pSTAT3 readouts, giving 24
matched inhibitor-cell-line-readout profiles.

## Role in the workflow

Figures 1, 2, 3, 4, and Supplementary Figure S1 are generated from the primary-replicate
experiments KZ214, KZ198, KZ226, and KZ223. Supplementary Figure S2 reports concordance
with KZ191, KZ203, KZ222, and KZ227.

The replicate IC50 values plotted in Supplementary Figure S2 are deposited in
`data/reference/replicate_ic50_shown_vs_replicate.csv`. The table uses `shown` for the
primary-replicate values and `replicate` for the independent-replicate values. It contains
48 rows: 24 matched profiles for each role.

## Recomputing concordance

Supplementary Figure S2 contains three log10-scale scatter plots: arithmetic-mean,
log-mean, and complete-inhibition IC50. The corresponding Pearson correlations can be
recomputed directly:

```python
import numpy as np
import pandas as pd

d = pd.read_csv("data/reference/replicate_ic50_shown_vs_replicate.csv")
w = d.pivot_table(
    index=["cell", "inhibitor", "readout"],
    columns="role",
    values=["mean", "log-mean", "complete", "partial"],
)

for metric in ["mean", "log-mean", "complete"]:
    primary = np.log10(w[(metric, "shown")])
    independent = np.log10(w[(metric, "replicate")])
    print(metric, np.corrcoef(primary, independent)[0, 1])
```

The resulting correlations are 0.9255, 0.9524, and 0.8722, which round to the values
shown in the figure caption (0.93, 0.95, and 0.87). Complete-inhibition IC50 exceeds
log-mean IC50 in all 24 independent-replicate profiles.

The exact raster is deposited under `manuscript/figures/Figure_S2.png`. The plotted values
and raw measurements are available, but the notebook-specific panel geometry and styling
are not reconstructed by the numbered pipeline. `scripts/15_figure_S2.py` generates a
separate pSTAT1-minus-pSTAT3 selectivity atlas under the distinct filename
`figures/Figure_S2_selectivity.png`.

## Exclusions

Independent-replicate exclusions are recorded in the plate metadata and filtration
documents:

| Experiment | Excluded wells | n |
|---|---|---:|
| KZ191 | A02 A06 C01 E02 E05 | 5 |
| KZ203 | A02 B01 B02 H01 | 4 |
| KZ222 | A03 A05 A06 A09 B01 B09 C05 D05 E01 E05 F01 G01 G10 H01 | 14 |
| KZ227 | A12 B09 E02 | 3 |

The corresponding files are under `metadata/filtration_docs/`, and experiment-level
roles and exclusions are summarized in `metadata/experiment_registry.csv`.

## Extrapolation sensitivity

Four of the 24 matched profiles include at least one IC50 estimate above the maximum
tested concentration of 10,000 nM. Restricting the concordance calculation to the 20
fully bracketed profile pairs gives:

| Metric | All 24 pairs | Bracketed 20 pairs |
|---|---:|---:|
| arithmetic mean | 0.925 | 0.824 |
| log-mean | 0.952 | 0.959 |
| complete inhibition | 0.872 | 0.914 |

The arithmetic-mean correlation is sensitive to the extrapolated estimates, whereas the
log-mean and complete-inhibition correlations are not reduced by their exclusion.
Complete-inhibition IC50 remains greater than log-mean IC50 in all 20 bracketed pairs.
