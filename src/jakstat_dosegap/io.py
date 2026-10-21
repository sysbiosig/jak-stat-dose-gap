"""Raw CellProfiler CSV -> tidy per-nucleus frame.

CONTRACT
--------
load_experiment(exp_id, exclude_wells=..., registry=...) -> pd.DataFrame
    One row per segmented nucleus that survived well exclusion.  Columns:
        experiment  str   e.g. "KZ214"
        cell        str   cell line, "THP1"/"U937" (None for histogram-only plates)
        well        str   e.g. "A01"
        stimulation float ng/mL IFN-alpha; 0 = unstimulated, 10 = saturating
        inhibitor_code str raw token from `inhibitor.1.0` ("0" when none)
        inhibitor   str   published compound name, or None when no inhibitor
        dose        float inhibitor concentration in the plate's own units; NaN when
                          no inhibitor, 0.0 for vehicle wells of a named inhibitor
        pSTAT1      float Alexa488 mean nuclear intensity, COUNTS (already x 65535)
        pSTAT3      float Alexa555 mean nuclear intensity, COUNTS (already x 65535)
    Intensities are NOT gated here - gating is per-readout and belongs to metrics.
    Rows whose intensities are unparseable become NaN rather than being dropped, so
    the two readouts stay row-aligned.

load_registry() -> pd.DataFrame  (metadata/experiment_registry.csv, one row per exp)
registry_exclusions(registry) -> dict[str, list[str]]

INVARIANTS
    * Intensities are returned in counts: raw value x NF (config.NF).
    * `dose` is resolved PER ROW from the inhibitor-specific dose column
      (config.DOSECOL), because one plate carries three inhibitors in three
      different dose columns.
    * Excluded wells are absent from the returned frame, so no statistic computed
      downstream can see them.  Within this function the per-row columns are built
      first and the exclusion filter is applied last; that ordering is safe because
      every column is row-wise independent (no cross-row aggregation happens here),
      so the surviving rows are identical either way.  Any aggregation added to this
      module in future MUST come after the filter.

GOTCHAS THIS MODULE ABSORBS (do not re-handle downstream)
    * The CSVs use bare CR (\\r) line endings, and pandas leaves the CR attached to
      the last field of every record.  Column NAMES and all object-dtype VALUES are
      stripped, otherwise `well.name` values look like "A01\\r" and never match an
      exclusion list.
    * KZ195 and KZ216 carry no inhibitor columns at all; missing design columns are
      filled rather than raising, so the same loader serves all six experiments.
    * Design columns arrive as strings on some plates and numbers on others; the
      numeric ones are coerced with errors="coerce".

PROVENANCE
    Raw files are the archived CellProfiler exports staged under data/raw/, checksums
    in metadata/raw_data_manifest.json.  Exclusion sets come from the per-plate
    plotting scripts as recorded in metadata/experiment_registry.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _strip_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace/CR from column names and every object-dtype value.

    WHY: the raw files use CR line endings, so the final field of each record keeps a
    trailing "\\r".  Which field that is depends on column order, so strip all of them.
    """
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip() if hasattr(df[col], "str") else df[col]
    return df


def load_registry(path: Path | str | None = None) -> pd.DataFrame:
    """Read metadata/experiment_registry.csv, indexed by experiment_id."""
    path = Path(path) if path is not None else config.REGISTRY_CSV
    reg = pd.read_csv(path)
    reg = _strip_frame(reg)
    return reg.set_index("experiment_id", drop=False)


def registry_exclusions(registry: pd.DataFrame | None = None) -> dict[str, list[str]]:
    """experiment_id -> list of excluded well names, parsed from the registry.

    The registry stores exclusions as a single space-separated string ("A01 A02 ...")
    and leaves the field empty when an experiment has none.
    """
    reg = load_registry() if registry is None else registry
    out: dict[str, list[str]] = {}
    for exp_id, row in reg.iterrows():
        raw = row.get("excluded_wells")
        wells = [] if pd.isna(raw) or not str(raw).strip() else str(raw).split()
        out[str(exp_id)] = wells
    return out


def exclusion_conflicts(registry: pd.DataFrame | None = None) -> pd.DataFrame:
    """Audit config.ANALYSIS_EXCLUSIONS against the registry's recorded sets.

    Returns one row per experiment where the two disagree, with the symmetric
    difference.  An empty frame means the registry and the analysis sets agree, which
    is the expected state: the registry was corrected to resolve per-plate
    conflicts by ANALYSIS VALIDITY (keep the set that preserves each plate's reference
    distributions) rather than by set size.  A non-empty frame therefore means the
    registry has drifted and should be re-examined -- see
    `exclusion_conflicts_block_controls` and the ANALYSIS_EXCLUSIONS note in config.py.
    """
    reg = registry_exclusions(registry)
    rows = []
    for exp_id, analysis in config.ANALYSIS_EXCLUSIONS.items():
        r, a = set(reg.get(exp_id, [])), set(analysis)
        if r != a:
            rows.append({
                "experiment": exp_id,
                "n_registry": len(r),
                "n_analysis": len(a),
                "registry_only": " ".join(sorted(r - a)),
                "analysis_only": " ".join(sorted(a - r)),
            })
    return pd.DataFrame(rows)


def reference_wells(exp_id: str) -> tuple[set[str], set[str]]:
    """The (unstimulated, saturated-inhibitor-free) well sets for one experiment.

    These are the two reference distributions every normalization bound depends on:
    the complete bound is the 95th percentile of the unstimulated cells, and the
    partial bound is derived from the 5th percentile of the saturated cells.  Read
    straight from the raw file's own design columns, so it reflects the plate rather
    than any registry claim about it.

    Returns empty sets for plates with no `stimulation.1.1` column.
    """
    d = _read_raw(config.raw_csv(exp_id))
    if config.COL_STIM not in d.columns:
        return set(), set()
    stim = d[config.COL_STIM].astype(str).str.strip()
    unstim = set(d.loc[stim.isin(["0", "0.0"]), config.COL_WELL].astype(str).str.strip())
    if config.COL_INHIB_NAME in d.columns:
        inh = d[config.COL_INHIB_NAME].astype(str).str.strip()
        sat = set(d.loc[(stim == "10") & inh.isin(["0", "0.0"]),
                        config.COL_WELL].astype(str).str.strip())
    else:
        sat = set()
    return unstim, sat


def exclusion_conflicts_block_controls(exp_id: str,
                                      wells: list[str] | None = None) -> bool:
    """True when an exclusion set would leave an experiment without usable controls.

    THE INVARIANT THIS GUARDS: an exclusion set is only usable if at least one
    unstimulated well AND at least one inhibitor-free saturated well survive it.
    Violate that and both normalization bounds become undefined, the experiment silently
    contributes zero profiles, and the published numbers cannot reproduce -- which is
    exactly what the largest-set resolution of KZ198 did (it removed B11+B12, the only
    unstimulated wells, and B09+B10, the only saturated ones).

    `wells` defaults to the registry's set for that experiment.
    """
    if wells is None:
        wells = registry_exclusions().get(exp_id, [])
    dropped = set(wells)
    unstim, sat = reference_wells(exp_id)
    # Histogram-only plates (KZ195/KZ216) carry no inhibitor columns, so they have no
    # saturated reference and never enter the dose-response analysis: only the plates
    # that actually contribute profiles need both distributions.
    if exp_id not in config.PANEL_EXPERIMENTS:
        return False
    return not (unstim - dropped) or not (sat - dropped)


def _read_raw(path: Path) -> pd.DataFrame:
    """Read only the columns the pipeline needs, from a ~100 MB per-nucleus export.

    WHY usecols with a callable: the exports carry several hundred CellProfiler
    measurement columns, and the design columns present differ between plates
    (KZ195/KZ216 have no inhibitor columns).  A predicate takes whatever exists and
    never raises on a missing name, unlike a fixed column list.
    """
    wanted = {config.COL_WELL, config.COL_CELLLINE, config.COL_STIM,
              config.COL_INHIB_NAME, *config.DOSE_COLUMNS, *config.RESP.values()}
    # Column names in the file may carry a trailing CR, so compare on the stripped name.
    d = pd.read_csv(path, usecols=lambda c: str(c).strip() in wanted, low_memory=False)
    return _strip_frame(d)


def _resolve_dose(d: pd.DataFrame) -> pd.Series:
    """Per-row inhibitor dose, taken from the dose column that row's inhibitor uses.

    Returns NaN for rows with no named inhibitor, or whose inhibitor code has no
    known dose column.
    """
    code = d["inhibitor_code"]
    dose = pd.Series(np.nan, index=d.index, dtype=float)
    for inhib_code, col in config.DOSECOL.items():
        if col not in d.columns:
            continue
        mask = code == inhib_code
        if mask.any():
            dose.loc[mask] = pd.to_numeric(d.loc[mask, col], errors="coerce")
    return dose


def load_experiment(
    exp_id: str,
    exclude_wells: list[str] | None = None,
    registry: pd.DataFrame | None = None,
    path: Path | str | None = None,
) -> pd.DataFrame:
    """Load one experiment as a tidy per-nucleus frame.  See module docstring.

    exclude_wells : explicit override.  Default None means "use
                    config.ANALYSIS_EXCLUSIONS for this experiment, falling back to
                    the registry".  Pass [] to load with no exclusions at all.
    """
    path = Path(path) if path is not None else config.raw_csv(exp_id)
    d = _read_raw(path)

    # Intensities: 16-bit rescale back to counts.  Kept as NaN when unparseable so the
    # two readouts remain row-aligned for any per-cell paired analysis.
    out = pd.DataFrame(index=d.index)
    for readout, col in config.RESP.items():
        if col not in d.columns:
            raise KeyError(f"{exp_id}: missing intensity column {col!r} in {path}")
        out[readout] = pd.to_numeric(d[col], errors="coerce") * config.NF

    out.insert(0, "experiment", exp_id)
    out.insert(1, "cell", config.CELL.get(exp_id))
    out.insert(2, "well", d[config.COL_WELL].astype(str) if config.COL_WELL in d
               else pd.Series("", index=d.index))
    out.insert(3, "stimulation", pd.to_numeric(d.get(config.COL_STIM), errors="coerce")
               if config.COL_STIM in d.columns else np.nan)

    # Histogram-only plates (KZ195/KZ216) have no `inhibitor.1.0`: treat every well as
    # inhibitor-free rather than failing, so one loader serves all six experiments.
    if config.COL_INHIB_NAME in d.columns:
        code = d[config.COL_INHIB_NAME].astype(str).str.strip()
    else:
        code = pd.Series("0", index=d.index)
    out.insert(4, "inhibitor_code", code)

    is_inhib = ~code.isin(config.NO_INHIBITOR_TOKENS)
    out.insert(5, "inhibitor", np.where(
        is_inhib, code.map(lambda c: config.LABELMAP.get(c, c)), None))
    out.insert(6, "dose", _resolve_dose(out.assign(**{
        c: d[c] for c in config.DOSE_COLUMNS if c in d.columns})))

    # Default provenance is the dose-response analysis set, NOT the registry set;
    # config.ANALYSIS_EXCLUSIONS documents why they differ for KZ198.  Experiments
    # absent from that map fall back to the registry.
    if exclude_wells is not None:
        excl = list(exclude_wells)
    elif exp_id in config.ANALYSIS_EXCLUSIONS:
        excl = list(config.ANALYSIS_EXCLUSIONS[exp_id])
    else:
        excl = registry_exclusions(registry).get(exp_id, [])
    if excl:
        out = out[~out["well"].isin(excl)]

    return out.reset_index(drop=True)


def load_panel(
    exp_ids: tuple[str, ...] | list[str] = config.PANEL_EXPERIMENTS,
    exclude_wells: dict[str, list[str]] | None = None,
    registry: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Load several experiments -> {experiment_id: per-nucleus frame}.

    exclude_wells : optional {exp_id: [wells]} override, else the default provenance
                    of load_experiment is used for each experiment.
    """
    reg = load_registry() if registry is None else registry
    return {
        e: load_experiment(
            e,
            exclude_wells=None if exclude_wells is None else exclude_wells.get(e),
            registry=reg,
        )
        for e in exp_ids
    }
