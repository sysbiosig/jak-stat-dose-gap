"""The four inhibition measures and the two single-cell state bounds.

CONTRACT
--------
Per-cell gating
    gate(signal) -> np.ndarray
        Keeps 1 < signal < 1e4 (counts) and drops NaN.  Applied to every intensity
        vector before any statistic is taken.  WHY: segmentation artefacts produce
        near-zero and saturated nuclei that would otherwise dominate a log-mean.

References for one (experiment, readout)
    reference_bounds(unstim, saturated) -> Bounds
        complete = 95th percentile of UNSTIMULATED cells.
            A cell below it is indistinguishable from an unstimulated cell, i.e.
            within the non-stimulated range.  95th pct rather than the max because the top tail
            of the unstimulated distribution is segmentation noise.
        partial  = max(complete, 5th percentile of SATURATED cells).
            A cell below it has left the fully-stimulated range, i.e. is at least
            partially inhibited.  The clamp is load-bearing: when the unstimulated and
            saturated distributions overlap, an unclamped 5th percentile would fall
            BELOW the complete bound and "partial" would be the stricter of the two.

The four measures for one dose group
    Population ("average") axes, both normalised so 0 = saturated, 1 = unstimulated:
        mean_inh    = (mean(sat)      - mean(x))      / (mean(sat)      - mean(unstim))
        logmean_inh = (mean(log10 sat)- mean(log10 x))/ (mean(log10 sat)- mean(log10 unstim))
        WHY both: the arithmetic mean is what a lysate/plate-reader assay reports; the
        log-mean is what a geometric-mean readout of a log-normal single-cell
        distribution reports.  The dose gap differs between them, which is the point.
    Single-cell state fractions, both in [0,1]:
        complete = fraction of gated cells with signal < complete bound
        partial  = fraction of gated cells with signal < partial bound
    Measures are NOT clipped: a value slightly outside [0,1] is real noise and
    clipping it here would bias the 4PL fit.

Assembly
    perdose_table(frames) -> pd.DataFrame
        One row per (experiment, cell, inhibitor, readout, dose) with all four
        measures plus n (gated cells).  Dose 0 rows are dropped - they are the vehicle
        controls that define the saturated reference, not dose-response points.
        Groups with fewer than config.MIN_CELLS gated cells are dropped.

INVARIANTS
    * Bounds and both population references are computed ONCE per
      (experiment, readout) from that plate's own controls - never pooled across
      plates, because absolute intensities are not comparable between stainings.
    * The saturated reference uses inhibitor-free, stimulation==10 wells only.
    * log10 is safe because gating already removed signal <= 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


def gate(signal) -> np.ndarray:
    """Apply the per-cell intensity gate; returns a float array of survivors."""
    x = np.asarray(pd.to_numeric(pd.Series(signal), errors="coerce"), dtype=float)
    x = x[np.isfinite(x)]
    return x[(x > config.SIGNAL_MIN) & (x < config.SIGNAL_MAX)]


@dataclass(frozen=True)
class Bounds:
    """Per-(experiment, readout) references.  Intensities in counts.

    complete / partial : state-occupancy thresholds (see module docstring).
    sat_mean / unstim_mean         : arithmetic means of the two references.
    sat_logmean / unstim_logmean   : means of log10 of the two references.
    n_sat / n_unstim               : gated cell counts behind the references.
    """

    complete: float
    partial: float
    sat_mean: float
    unstim_mean: float
    sat_logmean: float
    unstim_logmean: float
    n_sat: int
    n_unstim: int


def reference_bounds(unstim, saturated) -> Bounds:
    """Bounds from one plate's unstimulated and fully stimulated control cells.

    unstim / saturated : raw (ungated) intensity vectors in counts; gating is applied
                         here so callers cannot forget it.
    Raises ValueError when either reference has fewer than config.MIN_CELLS cells.
    """
    u = gate(unstim)
    s = gate(saturated)
    if len(u) < config.MIN_CELLS or len(s) < config.MIN_CELLS:
        raise ValueError(
            f"insufficient control cells: unstim n={len(u)}, saturated n={len(s)} "
            f"(need >= {config.MIN_CELLS} each)")
    complete = float(np.percentile(u, 95))
    partial = float(max(complete, np.percentile(s, 5)))
    return Bounds(
        complete=complete,
        partial=partial,
        sat_mean=float(np.mean(s)),
        unstim_mean=float(np.mean(u)),
        sat_logmean=float(np.mean(np.log10(s))),
        unstim_logmean=float(np.mean(np.log10(u))),
        n_sat=len(s),
        n_unstim=len(u),
    )


def mean_inhibition(signal, bounds: Bounds) -> float:
    """Arithmetic-mean population inhibition; 0 = saturated, 1 = unstimulated."""
    x = gate(signal)
    return float((bounds.sat_mean - np.mean(x)) / (bounds.sat_mean - bounds.unstim_mean))


def logmean_inhibition(signal, bounds: Bounds) -> float:
    """Log-mean population inhibition; 0 = saturated, 1 = unstimulated."""
    x = gate(signal)
    return float((bounds.sat_logmean - np.mean(np.log10(x)))
                 / (bounds.sat_logmean - bounds.unstim_logmean))


def complete_fraction(signal, bounds: Bounds) -> float:
    """Fraction of gated cells below the complete-inhibition bound."""
    return float(np.mean(gate(signal) < bounds.complete))


def partial_fraction(signal, bounds: Bounds) -> float:
    """Fraction of gated cells below the partial-inhibition bound."""
    return float(np.mean(gate(signal) < bounds.partial))


def all_measures(signal, bounds: Bounds) -> dict[str, float]:
    """All four measures plus n, from one dose group's raw intensity vector.

    Gates once and reuses the result, so this is the preferred entry point.
    """
    x = gate(signal)
    return {
        "mean_inh": float((bounds.sat_mean - np.mean(x))
                          / (bounds.sat_mean - bounds.unstim_mean)),
        "logmean_inh": float((bounds.sat_logmean - np.mean(np.log10(x)))
                             / (bounds.sat_logmean - bounds.unstim_logmean)),
        "complete": float(np.mean(x < bounds.complete)),
        "partial": float(np.mean(x < bounds.partial)),
        "n": int(len(x)),
    }


def _controls(d: pd.DataFrame, readout: str) -> tuple[np.ndarray, np.ndarray]:
    """Raw unstimulated and inhibitor-free-saturated intensity vectors for a readout."""
    unstim = d.loc[d["stimulation"] == config.STIM_UNSTIM, readout].values
    no_inhib = d["inhibitor_code"].isin(config.NO_INHIBITOR_TOKENS)
    sat = d.loc[no_inhib & (d["stimulation"] == config.STIM_SATURATING), readout].values
    return unstim, sat


def perdose_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-nucleus frames -> the per-dose measure table.  See module docstring.

    frames : {experiment_id: frame from io.load_experiment}
    Returns columns experiment, cell, inhibitor, readout, dose, mean_inh,
    logmean_inh, complete, partial, n, complete_bound, partial_bound.
    """
    rows = []
    for exp_id, d in frames.items():
        for readout in config.READOUTS:
            unstim, sat = _controls(d, readout)
            try:
                bounds = reference_bounds(unstim, sat)
            except ValueError:
                # No usable controls for this readout on this plate -> the plate cannot
                # define the state bounds at all, so every dose group here is unusable.
                continue
            stim = d[d["stimulation"] == config.STIM_SATURATING]
            for inhib_code, sub in stim.groupby("inhibitor_code", sort=True):
                if inhib_code in config.NO_INHIBITOR_TOKENS:
                    continue
                if inhib_code not in config.DOSECOL:
                    continue  # inhibitor with no dose column on this plate layout
                for dose, g in sub.groupby("dose", sort=True):
                    if dose == 0:
                        continue  # vehicle control, not a dose-response point
                    x = gate(g[readout].values)
                    if len(x) < config.MIN_CELLS:
                        continue
                    rows.append({
                        "experiment": exp_id,
                        "cell": config.CELL.get(exp_id),
                        "inhibitor": config.LABELMAP.get(inhib_code, inhib_code),
                        "readout": readout,
                        "dose": float(dose),
                        **all_measures(g[readout].values, bounds),
                        "complete_bound": bounds.complete,
                        "partial_bound": bounds.partial,
                    })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["experiment", "readout", "inhibitor", "dose"]).reset_index(drop=True)
    return out
