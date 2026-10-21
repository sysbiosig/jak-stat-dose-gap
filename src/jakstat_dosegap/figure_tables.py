"""Re-derivation of the per-figure source tables from data/raw/.

CONTRACT
--------
figure1_source_data()            -> per-cell frame, 103667 rows, 8 columns
published_inhibition_table(exp_ids) -> per-dose inhibition table in the published-figure
                                    wide layout (one row per design group x type)
published_observed_long()           -> the atlas long table the S1/S2 figure scripts melt
published_fit_table(long)           -> 4PL fits in the published-figure (not pipeline) convention

Every function here reads ONLY data/raw/ + config, and returns a frame whose keys and
numeric columns line up with the corresponding file in
data/reference/published_tables/.  scripts/05_build_figure_tables.py does the
comparison and writes the verdicts.

WHY THIS MODULE EXISTS SEPARATELY FROM metrics.py / fitting.py
--------------------------------------------------------------
The published-figure convention and the validated pipeline compute the SAME quantities under
DIFFERENT conventions.  Three differences are established numerically (see
docs/FIGURE_TABLES.md for the measured residuals):

1. GATING.  metrics.gate() applies the per-cell window 1 < signal < 1e4 before every
   statistic.  The published-figure log-mean path applies NO gate: it takes log10 of every
   segmented nucleus.  On these six plates no nucleus actually falls outside the
   window on the log-mean path, so ungated and gated log-means agree to 1e-15 - the
   two conventions are numerically indistinguishable HERE, and this module uses the
   ungated form only to keep the published-figure definition explicit.

2. THE SATURATED REFERENCE FOR `mean`.  The published-figure arithmetic-mean inhibition is an
   exact affine function of the pipeline's (per experiment x readout, max residual
   4e-16 on KZ198/KZ214 pSTAT3), which pins the discrepancy to the two population
   reference constants and NOT to the per-dose aggregation.  The implied published-figure
   saturated mean is 0.8-17% ABOVE the pooled inhibitor-free saturated mean, and no
   candidate rule tested reproduces it on all four (experiment, readout) cells:
   well-level averaging, alternate exclusion sets, tail trimming at the 1st/5th/10th
   percentile, median, geometric mean, and gating on the partner channel were all
   tried.  So `published_inhibition_table` returns the PIPELINE `mean` and the script
   reports the residual as measured rather than fitting a constant to close it.

3. THE KZ223 SATURATION SUBSTITUTION.  KZ223's inhibitor-free saturated wells are
   unusable, and the published-figure convention silently relabels the baricitinib 0.5 nM
   wells as the saturated reference (`apply_saturation_logic`).  With that
   substitution KZ223 log-mean reproduces to 1.3e-08 and its complete/partial
   fractions to 1e-16; without it, log-mean is off by up to 0.46.  The pipeline does
   NOT make this substitution - which is why data/derived/ic50_per_profile.csv and
   the published Figure_3/S1 IC50 tables disagree by ~2x on KZ223 specifically.
   `published_inhibition_table(kz223_saturation_substitution=True)` reproduces the
   published-figure convention; the default False is the pipeline convention.

INVARIANTS
    * Nothing here is imported by scripts/01..04 or by the validated modules.  The
      published numbers come from metrics.py/fitting.py and are unaffected by this
      file.
    * Well exclusions come from config.ANALYSIS_EXCLUSIONS, EXCEPT Figure 1's
      cerdulatinib histogram panels, where the figure-source table fixes a THIRD set
      (A12 B09 E02, n=3) that matches neither the registry (n=6) nor the analysis set
      (n=4).  That set is reproduced literally in FIG1_KZ198_EXCLUSIONS; it is a
      histogram-only choice and does not touch the dose-response path.
    * The published-figure 4PL differs from fitting.fit_4pl in two ways that DO move IC50s:
      p0 = [max(0,min(y)), min(1,max(y)), median(x), 1] rather than [0, 1, median, 1],
      and maxfev = 10000 rather than 20000.  published_fit_table uses the published-figure form so
      the regenerated fit tables are comparable to the published ones.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from . import config

# --- published-table naming ------------------------------------------------------
# The published tables key inhibitors by their DOSE COLUMN, not by compound code, and
# the two plate families put different compounds in the same column.
PUBLISHED_DOSE_COLUMNS = ("inhibitor.1.1", "inhibitor.1.2", "inhibitor.1.3")

# dose column -> compound, per plate family.  KZ214/KZ198 carry the JAK1/JAK2/TYK2
# codes; KZ226/KZ223 carry Bar/Rux/Tof.
COLUMN_COMPOUND = {
    "KZ214": {"inhibitor.1.1": "solcitinib", "inhibitor.1.2": "BMS-911543",
              "inhibitor.1.3": "cerdulatinib"},
    "KZ198": {"inhibitor.1.1": "solcitinib", "inhibitor.1.2": "BMS-911543",
              "inhibitor.1.3": "cerdulatinib"},
    "KZ226": {"inhibitor.1.1": "baricitinib", "inhibitor.1.2": "ruxolitinib",
              "inhibitor.1.3": "tofacitinib"},
    "KZ223": {"inhibitor.1.1": "baricitinib", "inhibitor.1.2": "ruxolitinib",
              "inhibitor.1.3": "tofacitinib"},
}

# The deposited files use title case for the additional inhibitor panel and lowercase
# for the original panel. Labels are preserved so joins against the reference tables succeed.
COMPOUND_LABEL = {
    "solcitinib": "solcitinib", "BMS-911543": "BMS-911543",
    "cerdulatinib": "cerdulatinib", "baricitinib": "Baricitinib",
    "ruxolitinib": "Ruxolitinib", "tofacitinib": "Tofacitinib",
}

# type token used by the published tables <-> pipeline measure name
LEGACY_TYPE = {"mean": "mean_inh", "log-mean": "logmean_inh",
               "complete": "complete", "partial": "partial"}
LEGACY_TYPES = ("mean", "log-mean", "complete", "partial")

# Figure 1 cerdulatinib panels use their own KZ198 exclusion set - see module docstring.
FIG1_KZ198_EXCLUSIONS = ("A12", "B09", "E02")

# Figure 1's panel condition series, as fixed by the figure-source convention.
FIG1_IFNG_DOSES = (0, 0.1, 0.5, 1, 10, 50)         # KZ195, stimulation.1.1, ng/mL
FIG1_IL10_DOSES = (0, 5, 10, 50, 100, 500)         # KZ195, stimulation.1.2, ng/mL
FIG1_CERDULATINIB_DOSES = (0, 25, 100, 250, 500, 1000)   # KZ198, inhibitor.1.3, nM
FIG1_MIXED_LABEL = {
    "0": "control",
    "IFNG": "IFN-\u03b3 = 10 ng/mL",
    "IL10": "IL-10 = 100 ng/mL",
    "IFNG_IL10": "IFN-\u03b3 = 10 ng/mL, IL-10 = 100 ng/mL",
}

# Figure 2 plots KZ214 only: solcitinib then cerdulatinib, pSTAT1 then pSTAT3.
FIG2_PANELS = (
    ("A", "KZ214", "inhibitor.1.1", "pSTAT1"),
    ("B", "KZ214", "inhibitor.1.1", "pSTAT3"),
    ("C", "KZ214", "inhibitor.1.3", "pSTAT1"),
    ("D", "KZ214", "inhibitor.1.3", "pSTAT3"),
)
READOUT_LABEL = {"pSTAT1": "pY-STAT1", "pSTAT3": "pY-STAT3"}

# Figure S1/S2 panel order.
ATLAS_INHIBITORS = ("solcitinib", "BMS-911543", "cerdulatinib",
                    "Baricitinib", "Ruxolitinib", "Tofacitinib")
ATLAS_CELL_LINES = ("THP1", "U937")

# Number of points on an exported fitted curve.  Fixed by the published-figure convention.
CURVE_POINTS = 200


# --- raw loading (published-figure idiom) --------------------------------------------
def _read_design(exp_id: str, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Raw CSV -> design + intensity columns, intensities in counts, CR stripped.

    Differs from io._read_raw only in taking `extra` column names: the published figures
    need `stimulation.1.0` (KZ216's mixed-stimulation label) and `stimulation.1.2`,
    which the dose-response pipeline never looks at.
    """
    wanted = {config.COL_WELL, config.COL_CELLLINE, config.COL_STIM,
              config.COL_INHIB_NAME, "stimulation.1.2",
              *config.DOSE_COLUMNS, *config.RESP.values(), *extra}
    d = pd.read_csv(config.raw_csv(exp_id),
                    usecols=lambda c: str(c).strip() in wanted, low_memory=False)
    d.columns = [str(c).strip() for c in d.columns]
    for col in d.columns:
        if d[col].dtype == object:
            d[col] = d[col].str.strip()
    for readout, col in config.RESP.items():
        d[readout] = pd.to_numeric(d[col], errors="coerce") * config.NF
    return d


def _numeric(d: pd.DataFrame, cols) -> pd.DataFrame:
    """Coerce design columns to float, filling absent ones with 0.

    WHY fillna(0) rather than NaN: the published-figure `coerce_numeric_columns` does exactly
    this, and the "other inhibitor columns are zero" filter that defines a dose series
    depends on it - a NaN there would silently drop the whole series.
    """
    d = d.copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0) if c in d.columns else 0.0
    return d


# --- Figure 1: per-cell intensities ---------------------------------------
def figure1_source_data() -> pd.DataFrame:
    """The Figure 1 panel intensities, per cell.  Columns as the published file.

    Returns 103667 rows over 28 (panel, experiment, readout, condition) groups:
      2A  KZ195 pY-STAT1 IFN-gamma dose series  + KZ216 IL-10 / IFN-gamma+IL-10
      2B  KZ195 pY-STAT3 IL-10 dose series      + KZ216 IFN-gamma / IFN-gamma+IL-10
      2C  KZ198 pY-STAT1 cerdulatinib dose series
      2D  KZ198 pY-STAT3 cerdulatinib dose series
    NO gating and NO exclusions are applied to KZ195/KZ216 - the panels show the raw
    response distribution, which is the point of the figure.  KZ198 uses
    FIG1_KZ198_EXCLUSIONS.
    """
    rows: list[dict] = []

    def emit(panel, exp, readout, ctype, cvalue, clabel, values):
        v = np.asarray(values, dtype=float)
        rows.append(pd.DataFrame({
            "panel": panel, "experiment": exp, "readout": readout,
            "condition_type": ctype, "condition_value": str(cvalue),
            "condition_label": clabel,
            "intensity_raw": v,
            "intensity_log10": np.where(v > 0, np.log10(np.where(v > 0, v, np.nan)),
                                        np.nan),
        }))

    # --- KZ195 dose series (panels 2A / 2B) --------------------------------
    kz195 = _numeric(_read_design("KZ195"),
                     [config.COL_STIM, "stimulation.1.2"])
    # Panel 2A holds IFN-gamma dose with IL-10 off; 2B is the mirror image.
    a = kz195[kz195["stimulation.1.2"] == 0]
    for dose in FIG1_IFNG_DOSES:
        emit("2A", "KZ195", "pY-STAT1", "IFN-\u03b3 dose", dose,
             f"IFN-\u03b3 = {dose} ng/mL",
             a.loc[a[config.COL_STIM] == dose, "pSTAT1"].values)
    b = kz195[kz195[config.COL_STIM] == 0]
    for dose in FIG1_IL10_DOSES:
        emit("2B", "KZ195", "pY-STAT3", "IL-10 dose", dose,
             f"IL-10 = {dose} ng/mL",
             b.loc[b["stimulation.1.2"] == dose, "pSTAT3"].values)

    # --- KZ216 mixed stimulation (folded into 2A / 2B) ---------------------
    kz216 = _read_design("KZ216", extra=("stimulation.1.0",))
    # WHY this normalisation: the CSV writes the double-stimulation level with an
    # encoding artefact ("IFNG+AF8-IL10") and leaves it blank on some rows.  Both mean
    # IFN-gamma + IL-10; the published-figure convention folds them together before selecting conditions.
    stim0 = kz216["stimulation.1.0"].astype(str)
    stim0 = stim0.where(~stim0.str.lower().isin(["nan", "na", ""]), "IFNG_IL10")
    kz216["stim0"] = stim0.replace({"IFNG+AF8-IL10": "IFNG_IL10"})
    for cond, readout in (("IL10", "pSTAT1"), ("IFNG_IL10", "pSTAT1")):
        emit("2A", "KZ216", "pY-STAT1", "mixed stimulation", cond,
             FIG1_MIXED_LABEL[cond], kz216.loc[kz216["stim0"] == cond, readout].values)
    for cond, readout in (("IFNG", "pSTAT3"), ("IFNG_IL10", "pSTAT3")):
        emit("2B", "KZ216", "pY-STAT3", "mixed stimulation", cond,
             FIG1_MIXED_LABEL[cond], kz216.loc[kz216["stim0"] == cond, readout].values)

    # --- KZ198 cerdulatinib series (panels 2C / 2D) ------------------------
    kz198 = _numeric(_read_design("KZ198"),
                     [config.COL_STIM, *PUBLISHED_DOSE_COLUMNS])
    kz198 = kz198[~kz198[config.COL_WELL].isin(FIG1_KZ198_EXCLUSIONS)]
    sub = kz198[(kz198["inhibitor.1.1"] == 0) & (kz198["inhibitor.1.2"] == 0)
                & (kz198[config.COL_STIM] != 0)]
    for panel, readout, label in (("2C", "pSTAT1", "pY-STAT1"),
                                  ("2D", "pSTAT3", "pY-STAT3")):
        for dose in FIG1_CERDULATINIB_DOSES:
            emit(panel, "KZ198", label, "cerdulatinib dose", dose,
                 f"cerdulatinib = {dose} nM",
                 sub.loc[sub["inhibitor.1.3"] == dose, readout].values)

    return pd.concat(rows, ignore_index=True)


def figure1_histogram_bounds(readout: str) -> dict[str, float]:
    """KZ198 complete/partial bounds on the log10 scale, for the Fig 1 C/D bands.

    Reproduces the published `compute_kz198_hist_bounds`: gate 1 < signal < 1e4, then
    complete = 95th percentile of unstimulated, partial = max(complete, 5th percentile
    of stimulated inhibitor-free) - the same rule as metrics.reference_bounds, and the
    values agree; only the exclusion set differs (see FIG1_KZ198_EXCLUSIONS).
    """
    d = _numeric(_read_design("KZ198"), [config.COL_STIM])
    d = d[~d[config.COL_WELL].isin(FIG1_KZ198_EXCLUSIONS)]
    no_inhib = d[config.COL_INHIB_NAME].astype(str) == "0"

    def gate(v):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        return v[(v > config.SIGNAL_MIN) & (v < config.SIGNAL_MAX)]

    unstim = gate(d.loc[no_inhib & (d[config.COL_STIM] == 0), readout].values)
    sat = gate(d.loc[no_inhib & (d[config.COL_STIM] != 0), readout].values)
    complete = float(np.percentile(unstim, 95))
    partial = float(max(complete, np.percentile(sat, 5)))
    return {"complete": float(np.log10(complete)), "partial": float(np.log10(partial))}


# --- Figures 2 / S1 / S2: per-dose inhibition ------------------------------
def _ungated_logmean(v) -> float:
    """mean(log10(signal)) over every finite positive nucleus.  No gate - see docstring."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.mean(np.log10(v)))


def _gate(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return v[(v > config.SIGNAL_MIN) & (v < config.SIGNAL_MAX)]


def published_inhibition_table(
    exp_ids=config.PANEL_EXPERIMENTS,
    kz223_saturation_substitution: bool = True,
) -> pd.DataFrame:
    """Per-dose inhibition in the published-table long layout, re-derived from raw.

    Returns one row per (experiment, inhibitor, readout, dose, type) with:
        inhibition   the value; `type` is one of LEGACY_TYPES
        n            gated cells behind it
    Dose 0 rows ARE included (the published tables carry them and the published figures plot
    them), unlike perdose.csv which drops them as vehicle controls.

    kz223_saturation_substitution : reproduce the published-figure's relabelling of
        KZ223's baricitinib 0.5 nM wells as the saturated reference.  See the module
        docstring: True reproduces the published files, False is the pipeline
        convention.  Has no effect on any other experiment.
    """
    recs = []
    for exp_id in exp_ids:
        d = _numeric(_read_design(exp_id), [config.COL_STIM, *PUBLISHED_DOSE_COLUMNS])
        d = d[~d[config.COL_WELL].isin(config.ANALYSIS_EXCLUSIONS.get(exp_id, []))]
        code = d[config.COL_INHIB_NAME].astype(str)
        no_inhib = code.isin(config.NO_INHIBITOR_TOKENS)
        stimulated = d[config.COL_STIM] != 0

        for readout in config.READOUTS:
            unstim = d.loc[no_inhib & (d[config.COL_STIM] == 0), readout].values
            if kz223_saturation_substitution and exp_id == "KZ223":
                # KZ223's inhibitor-free saturated wells are unusable; the published-figure convention
                # published-figure convention substitutes the lowest baricitinib dose.
                sat = d.loc[(code == "Bar") & (d["inhibitor.1.1"] == 0.5)
                            & stimulated, readout].values
            else:
                sat = d.loc[no_inhib & stimulated, readout].values
            ug, sg = _gate(unstim), _gate(sat)
            if len(ug) < config.MIN_CELLS or len(sg) < config.MIN_CELLS:
                continue
            complete_bound = float(np.percentile(ug, 95))
            partial_bound = float(max(complete_bound, np.percentile(sg, 5)))
            sat_mean, unstim_mean = float(sg.mean()), float(ug.mean())
            sat_lm = _ungated_logmean(sat)
            unstim_lm = _ungated_logmean(unstim)

            def measures(values) -> dict[str, float]:
                """The four published-figure measures for one dose group's raw intensities."""
                x = _gate(values)
                return {
                    "mean": (sat_mean - float(x.mean())) / (sat_mean - unstim_mean),
                    "log-mean": ((sat_lm - _ungated_logmean(values))
                                 / (sat_lm - unstim_lm)),
                    "complete": float((x < complete_bound).mean()),
                    "partial": float((x < partial_bound).mean()),
                    "n": float(len(x)),
                }

            def emit(dose_col, compound, dose, vals):
                for type_ in LEGACY_TYPES:
                    recs.append({
                        "experiment_id": exp_id,
                        "cell_line": config.CELL[exp_id],
                        "inhibitor": compound,
                        "inhibitor_label": COMPOUND_LABEL[compound],
                        "inhibitor_col": dose_col,
                        "readout": readout,
                        "dose": float(dose),
                        "type": type_,
                        "inhibition": float(vals[type_]),
                        "n": int(vals["n"]),
                    })

            # DOSE-ZERO ROWS.  The published-table layout keys a row by which dose column is
            # nonzero, so the plate's saturated reference group satisfies "all other
            # columns are zero" for ALL THREE columns at once and appears as a dose-0
            # point on every inhibitor's curve.  There are no true vehicle wells for a
            # named inhibitor on these plates (verified: zero rows with a named
            # inhibitor code and dose 0), so this replication is the whole origin of
            # the dose-0 rows in the published tables.
            #
            # The dose-0 row is therefore the reference group normalised against
            # ITSELF: log-mean and mean are 0 by construction, and complete/partial are
            # that group's own state fractions.  It uses `sat` (not the inhibitor-free
            # set) so that KZ223 under the substitution reports its substituted 0.5 nM
            # group here too - which is what the published file shows (KZ223 dose 0 and
            # dose 0.5 carry identical values).
            #
            # NOTE these rows are NOT plotted: every published panel uses a log x-axis, on
            # which dose 0 has no position.  They exist only in the tables.
            if len(sg) >= config.MIN_CELLS:
                zero_vals = measures(sat)
                for dose_col, compound in COLUMN_COMPOUND[exp_id].items():
                    emit(dose_col, compound, 0.0, zero_vals)

            for inhib_code, sub in d[stimulated].groupby(code[stimulated], sort=True):
                if inhib_code in config.NO_INHIBITOR_TOKENS:
                    continue
                dose_col = config.DOSECOL.get(inhib_code)
                if dose_col is None:
                    continue
                compound = COLUMN_COMPOUND[exp_id][dose_col]
                for dose, grp in sub.groupby(dose_col, sort=True):
                    if len(_gate(grp[readout].values)) < config.MIN_CELLS:
                        continue
                    emit(dose_col, compound, dose, measures(grp[readout].values))
    out = pd.DataFrame(recs)
    return out.sort_values(["experiment_id", "readout", "inhibitor_label", "dose",
                            "type"]).reset_index(drop=True)


# --- published-figure 4PL ------------------------------------------------------------
def four_pl(dose, bottom, top, ic50, hill):
    """Same functional form as fitting.four_pl; duplicated so this module stands alone."""
    return bottom + (top - bottom) / (1.0 + (np.asarray(dose, dtype=float) / ic50) ** hill)


def _interp_ic50(dose, y) -> float | None:
    """Log-linear interpolation to inhibition = 0.5, for profiles with < 4 doses.

    The published-figures fall back to this rather than reporting no IC50.  Kept for
    fidelity; on these six plates every profile has >= 10 doses so it never fires.
    """
    order = np.argsort(dose)
    x, yy = np.asarray(dose)[order], np.asarray(y)[order]
    keep = x > 0
    x, yy = x[keep], yy[keep]
    if len(np.unique(x)) < 2:
        return None
    for i in range(len(x) - 1):
        y1, y2 = yy[i], yy[i + 1]
        if (y1 - 0.5) * (y2 - 0.5) <= 0 and y1 != y2:
            t = (0.5 - y1) / (y2 - y1)
            return float(10 ** (np.log10(x[i]) + t * (np.log10(x[i + 1]) - np.log10(x[i]))))
    return None


def published_fit_4pl(dose, y) -> tuple[dict | None, str]:
    """The published fit: data-driven p0, maxfev=10000, interpolation fallback.

    Returns ({bottom, top, ic50, hill}, status) with status in
    {"ok", "interp", "insufficient_points", "invalid_ic50", "fit_failed"}.
    Differs from fitting.fit_4pl in p0 and maxfev - both move the reported IC50, which
    is why this is not routed through fitting.py.
    """
    x = np.asarray(dose, dtype=float)
    yy = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(yy) & (x > 0)
    x, yy = x[keep], yy[keep]
    order = np.argsort(x)
    x, yy = x[order], yy[order]

    if len(np.unique(x)) < 4:
        est = _interp_ic50(x, yy)
        if est:
            return {"bottom": np.nan, "top": np.nan, "ic50": est, "hill": np.nan}, "interp"
        return None, "insufficient_points"

    p0 = [max(0.0, float(np.nanmin(yy))), min(1.0, float(np.nanmax(yy))),
          float(np.median(x)), 1.0]
    bounds = ([0.0, 0.0, float(x.min()) / 10, 0.2],
              [1.2, 1.2, float(x.max()) * 10, 5.0])
    try:
        with warnings.catch_warnings():
            # Overflow inside (x/ic50)**hill on a trial step is normal for the
            # optimiser and does not invalidate the returned optimum.
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(four_pl, x, yy, p0=p0, bounds=bounds, maxfev=10000)
    except Exception as exc:
        est = _interp_ic50(x, yy)
        if est:
            return {"bottom": np.nan, "top": np.nan, "ic50": est, "hill": np.nan}, "interp"
        return None, f"fit_failed: {type(exc).__name__}"
    bottom, top, ic50, hill = (float(v) for v in popt)
    if not np.isfinite(ic50) or ic50 <= 0:
        est = _interp_ic50(x, yy)
        if est:
            return {"bottom": bottom, "top": top, "ic50": est, "hill": hill}, "interp"
        return None, "invalid_ic50"
    return {"bottom": bottom, "top": top, "ic50": ic50, "hill": hill}, "ok"


def published_fit_table(long: pd.DataFrame,
                     group_cols=("experiment_id", "cell_line", "inhibitor_label",
                                 "inhibitor_col", "readout", "type")
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every profile in a published long table -> (fit summary, fitted curves).

    long : output of published_inhibition_table (needs dose, inhibition, and group_cols)
    Returns
        fits   : one row per group with fit_status, n_doses, ic50, bottom, top, hill
        curves : CURVE_POINTS rows per fitted group, log-spaced over the group's own
                 measured dose window (dose > 0), matching the published export
    """
    fit_rows, curve_rows = [], []
    for key, grp in long.groupby(list(group_cols), sort=True):
        meta = dict(zip(group_cols, key))
        g = grp.sort_values("dose")
        params, status = published_fit_4pl(g["dose"].values, g["inhibition"].values)
        positive = g.loc[g["dose"] > 0, "dose"]
        fit_rows.append({**meta, "fit_status": status,
                         "n_doses": int(positive.nunique()),
                         "ic50": params["ic50"] if params else np.nan,
                         "bottom": params["bottom"] if params else np.nan,
                         "top": params["top"] if params else np.nan,
                         "hill": params["hill"] if params else np.nan})
        if params is not None and positive.nunique() >= 2:
            xs = np.logspace(np.log10(positive.min()), np.log10(g["dose"].max()),
                             CURVE_POINTS)
            ys = four_pl(xs, params["bottom"], params["top"], params["ic50"],
                         params["hill"])
            curve_rows.append(pd.DataFrame({**meta, "fit_status": status,
                                            "dose": xs, "inhibition": ys,
                                            "data_type": "fit"}))
    fits = pd.DataFrame(fit_rows)
    curves = (pd.concat(curve_rows, ignore_index=True) if curve_rows
              else pd.DataFrame(columns=[*group_cols, "fit_status", "dose",
                                         "inhibition", "data_type"]))
    return fits, curves


def selectivity_delta(long: pd.DataFrame,
                      types=("log-mean", "complete")) -> pd.DataFrame:
    """Delta-I = inhibition(pSTAT1) - inhibition(pSTAT3), per dose.  Figure S2's input.

    Only rows where BOTH readouts exist at a dose survive, matching the published figure's
    dropna on the pivot.
    """
    sub = long[long["type"].isin(types)]
    wide = sub.pivot_table(
        index=["experiment_id", "cell_line", "inhibitor_label", "inhibitor_col",
               "dose", "type"],
        columns="readout", values="inhibition", aggfunc="first").reset_index()
    wide = wide.dropna(subset=["pSTAT1", "pSTAT3"]).copy()
    wide["delta_inhibition"] = wide["pSTAT1"] - wide["pSTAT3"]
    return wide.rename(columns={"experiment_id": "experiment",
                                "inhibitor_label": "inhibitor",
                                "type": "metric"}).sort_values(
        ["metric", "cell_line", "inhibitor", "dose"]).reset_index(drop=True)
