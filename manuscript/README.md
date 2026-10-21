# Manuscript source

LaTeX source and compiled PDF for the manuscript, with the figures it embeds.

```
main_all.tex      the manuscript source
main_all.pdf      the compiled manuscript
refs.bib          external bibliography
figures/*.png     the six figures, as relative paths
build.sh          compile: bash build.sh  ->  main_all.pdf
```

## Compiling

```bash
cd manuscript
bash build.sh
```

Runs pdflatex, bibtex, then pdflatex twice, and reports undefined references and page
count. Expect **0 undefined references**. Requires a TeX distribution; on macOS
MacTeX installs `pdflatex` to `/Library/TeX/texbin`, which `build.sh` adds to `PATH`.

Figure paths are relative, so the folder compiles anywhere — a local TeX install, Overleaf,
or a co-author's machine. For Overleaf, upload the whole folder.

## Which figures these are

| file | source |
|---|---|
| Figure_1.png | produced by `../scripts/12_figure_1.py` and copied into this folder |
| Figure_2.png | manuscript raster; source data and reconstruction script are deposited via `../scripts/13_figure_2.py` |
| Figure_3.png, Figure_4.png | produced by the pipeline (`../scripts/10_figure_3.py`, `11_figure_4.py`) |
| Figure_S1.png | produced by the pipeline (`../scripts/14_figure_S1.py`) |
| Figure_S2.png | manuscript raster for the IC50 replicate-concordance scatter; plotted values and replicate measurements are deposited (see `../docs/BIOLOGICAL_REPLICATE.md`) |

Figures 1, 3, 4 and S1 here are the pipeline's output, so what you regenerate is what
the manuscript shows. Figures 2 and S2 are manuscript rasters with deposited source data
and provenance notes.

The pipeline reconstructs Figure 2 under `../figures/`. The raster is not byte-identical:
a different matplotlib build shifts thin
curves by sub-pixel amounts and changes tick density slightly. Against the manuscript
raster, 88.4% of Figure 2 pixels fall within 8 grey levels of 255; panel content, axis
labels and semantic colors are unchanged. Per-figure numbers are in `../VERIFICATION.md`.

To refresh the figures after changing the analysis, run `bash ../scripts/run_all.sh` from
the repository root and copy the results from `../figures/` into `figures/` here.
