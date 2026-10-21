"""jakstat_dosegap - reproduction pipeline for "Single-cell profiling quantifies
the dose gap between population-average and complete pY-STAT1/3 suppression by JAK
inhibitors".

CONTRACT
--------
Raw input : CellProfiler per-nucleus exports, one CSV per experiment, at
            data/raw/<EXPID>/ShrinkedNuclei.csv.
Final out : four tables under data/derived/ (see scripts/01..04).

The package is a straight port of the validated reference implementation that
reproduces the published crossing values to machine precision.  Every numerical
convention that the published numbers depend on lives in `config.py` as a named
constant; nothing downstream re-derives them.

MODULE MAP (read the module docstring, not the body)
    config         numerical conventions, label maps, paths, palette
    io             raw CSV -> tidy per-nucleus frame, registry well exclusions
    metrics        the four inhibition measures + the two single-cell bounds
    fitting        4PL dose-response fit and IC50 extraction
    crossings      the validated crossing rule at thresholds 0.2 / 0.5 / 0.8
    decomposition  type-II ANOVA variance shares + bootstrap CIs
    figstyle       presentation layer: locked palette, markers, panel letters, axis
                   conventions, text-collision checker.  Imported only by the figure
                   scripts; no analysis module depends on it.

IMPORT SIDE EFFECTS: none.  No module reads a file at import time.
"""

__version__ = "1.0.0"

from . import config, crossings, decomposition, fitting, io, metrics  # noqa: F401

__all__ = ["config", "io", "metrics", "fitting", "crossings", "decomposition"]
