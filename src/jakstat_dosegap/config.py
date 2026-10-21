"""Numerical conventions, label maps and repo paths.  Single source of truth.

CONTRACT
--------
Inputs  : none (pure constants + two path helpers).
Outputs : module-level constants, documented below with units.
Invariant: every magic number the published results depend on appears HERE and
           nowhere else.  If a downstream module contains a bare literal that
           affects a reported value, that is a bug.

WHY these particular values: they are the conventions of the original analysis
scripts, verified to reproduce the published numbers to machine precision.  They
are NOT free parameters - do not tune them.

KEY CONSTANTS
    NF            65535.0        raw CellProfiler intensities are 16-bit scaled to
                                 [0,1]; multiplying by NF returns them to counts.
    SIGNAL_MIN/MAX 1 / 1e4       per-cell gate in counts, applied per readout.
    RESP          readout -> CellProfiler intensity column.
    DOSECOL       inhibitor code -> the CSV column holding that inhibitor's dose.
                                 WHY: the plate layout encodes three inhibitors per
                                 experiment in three separate dose columns, so the
                                 dose of a cell depends on WHICH inhibitor it got.
    CELL          experiment id -> cell line, for the four inhibitor-panel plates.
    STIM_SATURATING 10           ng/mL IFN-alpha; the fully stimulated reference.
    STIM_UNSTIM     0            unstimulated reference.
    CROSSING_THRESHOLDS          population-inhibition targets for the crossing rule.
    POP_MEASURES / STATE_MEASURES  which of the four measures form which axis.
"""

from __future__ import annotations

from pathlib import Path

# --- repo layout -----------------------------------------------------------
# WHY resolve() and parents[2]: the package lives at <repo>/src/jakstat_dosegap,
# so the repo root is two levels up.  This keeps every script runnable from any cwd.
PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_DERIVED = REPO_ROOT / "data" / "derived"
DATA_REFERENCE = REPO_ROOT / "data" / "reference"
METADATA_DIR = REPO_ROOT / "metadata"
FIGURES_DIR = REPO_ROOT / "figures"
REGISTRY_CSV = METADATA_DIR / "experiment_registry.csv"

# --- intensity conventions -------------------------------------------------
NF = 65535.0                 # 16-bit rescale factor (dimensionless)
SIGNAL_MIN = 1.0             # counts, exclusive lower gate
SIGNAL_MAX = 1e4             # counts, exclusive upper gate
MIN_CELLS = 10               # minimum gated nuclei for a group to be usable

RESP = {
    "pSTAT1": "Intensity_MeanIntensity_Alexa488",
    "pSTAT3": "Intensity_MeanIntensity_Alexa555",
}
READOUTS = ("pSTAT1", "pSTAT3")

# --- plate-design columns --------------------------------------------------
COL_WELL = "well.name"
COL_CELLLINE = "cells.1.1"
COL_STIM = "stimulation.1.1"
COL_INHIB_NAME = "inhibitor.1.0"
DOSE_COLUMNS = ("inhibitor.1.1", "inhibitor.1.2", "inhibitor.1.3",
                "inhibitor.2.1", "inhibitor.2.2", "inhibitor.2.3")

STIM_UNSTIM = 0
STIM_SATURATING = 10

# Inhibitor code as written in `inhibitor.1.0` -> the dose column for that code.
DOSECOL = {
    "JAK1": "inhibitor.1.1", "JAK2": "inhibitor.1.2", "TYK2": "inhibitor.1.3",
    "Bar": "inhibitor.1.1", "Rux": "inhibitor.1.2", "Tof": "inhibitor.1.3",
}

# Inhibitor code -> published compound name.  Codes name the nominal JAK target on
# the KZ214/KZ198 plates and abbreviate the compound on the KZ226/KZ223 plates.
LABELMAP = {
    "JAK1": "solcitinib", "JAK2": "BMS-911543", "TYK2": "cerdulatinib",
    "Bar": "baricitinib", "Rux": "ruxolitinib", "Tof": "tofacitinib",
    "baricitinib": "baricitinib", "ruxolitinib": "ruxolitinib",
    "tofacitinib": "tofacitinib",
}

# Tokens in `inhibitor.1.0` that mean "no inhibitor" rather than a compound.
NO_INHIBITOR_TOKENS = frozenset({"0", "0.0", "nan", "", "NA", "none", "None"})

# The four plates carrying the inhibitor panel, and their cell lines.
CELL = {"KZ214": "THP1", "KZ198": "U937", "KZ226": "THP1", "KZ223": "U937"}
PANEL_EXPERIMENTS = ("KZ214", "KZ198", "KZ226", "KZ223")
# KZ195/KZ216 are histogram-only plates: they carry NO inhibitor columns at all.
HISTOGRAM_ONLY_EXPERIMENTS = ("KZ195", "KZ216")
ALL_EXPERIMENTS = PANEL_EXPERIMENTS + HISTOGRAM_ONLY_EXPERIMENTS

# --- well exclusions -------------------------------------------------------
# Wells excluded from the dose-response analysis, taken from the per-plate records for
# the manuscript dose-response panels. The constants are kept alongside the registry so
# io.exclusion_conflicts() can verify that metadata and analysis agree. KZ198 has a
# rejected six-well candidate set in its filtration history; the four-well analysis set
# is required because the larger set removes the reference controls:
#
#   KZ198 registry set  = A12 B09 B10 B11 B12 E02  (n=6, from a histogram-only record)
#   KZ198 analysis set  = A12 B05 B09 E02          (n=4, from the dose-response record)
#
# B10, B11 and B12 are KZ198's CONTROL wells.  Verified against the raw file: B11 and
# B12 are the plate's ONLY unstimulated wells (stimulation == 0), and its inhibitor-free
# saturated wells (stimulation == 10, inhibitor.1.0 == 0) are B09 and B10 - of which B09
# is excluded by BOTH candidate sets.  So under the registry set every unstimulated well
# AND every surviving saturated well is removed, leaving neither reference distribution:
# the complete/partial bounds become undefined and KZ198 drops out of the analysis
# entirely (24 profiles -> 18; tests/test_reproduction.py pins this).  The
# histogram-only convention could drop them because it only drew vertical lines on other wells'
# histograms; the dose-response analysis cannot.  For the other three plates the
# registry set and the analysis set agree exactly.
#
# Use io.exclusion_conflicts() to re-audit this against the registry.
ANALYSIS_EXCLUSIONS = {
    "KZ214": ["A01", "A02", "A03", "A04", "A05", "A06", "A12", "B12", "C05", "C06",
              "C07", "C08", "C09", "C10", "C12", "E09", "E10", "E11", "E12", "F01",
              "F02"],
    "KZ198": ["A12", "B05", "B09", "E02"],
    "KZ226": ["A01", "A02", "A03", "A04", "B11", "C01", "C02", "C03", "C04", "E01",
              "E02", "E03", "E04", "E07", "E09", "E11"],
    "KZ223": ["A05", "A09", "A11", "B05", "B09", "C01", "C05", "C09", "D01", "D05",
              "D09", "E01", "E05", "E09", "F01", "F05", "F09", "G01", "G05", "G09",
              "H01", "H04", "H05", "H07", "H08", "H09", "H11", "H12"],
    # Histogram-only plates: no exclusions recorded for these plates.
    "KZ195": [],
    "KZ216": [],
}

# --- measures --------------------------------------------------------------
# Population-level ("average") axes: what a plate-reader style assay would report.
POP_MEASURES = ("mean_inh", "logmean_inh")
# Single-cell state fractions: what fraction of cells lie in the non-stimulated range.
STATE_MEASURES = ("complete", "partial")
ALL_MEASURES = POP_MEASURES + STATE_MEASURES
# Short axis keys used in the published tables / target-value JSON.
AXIS_KEY = {"mean_inh": "arith", "logmean_inh": "log"}

CROSSING_THRESHOLDS = (0.2, 0.5, 0.8)

# --- fitting ---------------------------------------------------------------
FIT_MAXFEV = 20000
# 4PL parameter order is (bottom, top, ic50, hill).
# WHY these bounds: responses are inhibition fractions in [0,1] (1.2 allows a little
# overshoot from noise); ic50 is constrained to within a decade of the measured dose
# range so an unconverged fit cannot report an IC50 outside the tested window; the
# Hill floor of 0.2 stops the optimiser flattening the curve into a straight line.
FIT_BOUNDS_LOW = (0.0, 0.0, 0.1, 0.2)     # ic50 low  is scaled by dose.min()
FIT_BOUNDS_HIGH = (1.2, 1.2, 10.0, 5.0)   # ic50 high is scaled by dose.max()

# --- statistics ------------------------------------------------------------
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 7
ANOVA_FACTORS = ("cell", "inhibitor", "readout")
# The published tables label the readout factor "target" (the phospho-target read).
ANOVA_FACTOR_LABEL = {"cell": "cell", "inhibitor": "inhibitor", "readout": "target"}

# --- presentation ----------------------------------------------------------
PALETTE = {
    "partial": "#c0392b", "complete": "#1f6cb4",
    "pSTAT1": "#5e3c99", "pSTAT3": "#e6a419",
    "factor_cell": "#543005", "factor_inhibitor": "#8c6bb1",
    "factor_target": "#66c2a4", "factor_unexplained": "#e0e0e0",
    "tag": "#555555",
}
MARKERS = {"THP1": "o", "U937": "^"}


def raw_csv(experiment_id: str) -> Path:
    """Path to one experiment's raw per-nucleus CSV.  Does not check existence."""
    return DATA_RAW / experiment_id / "ShrinkedNuclei.csv"


def derived(name: str) -> Path:
    """Path under data/derived/, creating the directory if needed."""
    DATA_DERIVED.mkdir(parents=True, exist_ok=True)
    return DATA_DERIVED / name
