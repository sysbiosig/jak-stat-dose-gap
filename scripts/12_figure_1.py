#!/usr/bin/env python3
"""Figure 1 - response distributions, reference states and inhibitor redistribution.

CONTRACT
--------
Input (read-only):
    data/derived/figure_tables/Figure_1_source_data.csv
        per-nucleus intensities for cytokine-dose and cerdulatinib panels.

Output:
    figures/Figure_1.png

The figure is an explanatory four-row layout:
    A  Compact cytokine-dose distributions for pSTAT1 and pSTAT3, with the combined
       full-stimulation reference shown as the final row.
    B  Non-stimulated and fully stimulated reference distributions used to define the
       complete- and partial-inhibition thresholds.
    C  Cerdulatinib-treated distributions relative to those fixed thresholds.
    D  State occupancy by cerdulatinib dose.

Histogram heights are fractions of cells per bin. Compact stacked histograms use
independent y-axis ranges by row so broad and narrow distributions remain visible.
The state-occupancy bars in row D are absolute fractions on a shared 0-1 scale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from jakstat_dosegap import config, figure_tables  # noqa: E402

TABLES_DIR = REPO / "data" / "derived" / "figure_tables"
FIGURES = REPO / "figures"

READOUTS = ("pSTAT1", "pSTAT3")
READOUT_LABEL = {"pSTAT1": "pSTAT1", "pSTAT3": "pSTAT3"}
PANEL_FROM_READOUT = {"pSTAT1": "2C", "pSTAT3": "2D"}
CERDULATINIB_DOSES = [str(v) for v in figure_tables.FIG1_CERDULATINIB_DOSES]
CYTOKINE_SERIES = {
    "pSTAT1": {
        "panel": "2A",
        "name": "IFN-gamma",
        "unit": "ng/mL",
        "doses": ["0", "0.5", "1", "10"],
    },
    "pSTAT3": {
        "panel": "2B",
        "name": "IL-10",
        "unit": "ng/mL",
        "doses": ["0", "10", "50", "100"],
    },
}

COMPLETE = config.PALETTE["complete"]
PARTIAL = config.PALETTE["partial"]
RESIDUAL = "#707070"
REFERENCE_STIM = "#3d3d3d"
CYTOKINE_COLORS = ("#8c8c8c", "#2c7fb8", "#41ab5d", "#d95f0e")
FULL_STIM_COLOR = "#111111"

XLIM = (0.35, 3.75)
XTICKS = (1, 2, 3)
BINS = np.linspace(XLIM[0], XLIM[1], 130)
FIGSIZE = (10.5, 11.8)
DPI = 300


def load_source() -> pd.DataFrame:
    path = TABLES_DIR / "Figure_1_source_data.csv"
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(REPO)} - run scripts/05_build_figure_tables.py first"
        )
    data = pd.read_csv(path)
    data["condition_value"] = data["condition_value"].astype(str)
    return data


def gate_log(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[(values > config.SIGNAL_MIN) & (values < config.SIGNAL_MAX)]
    return np.log10(values)


def kz198_reference(readout: str) -> tuple[np.ndarray, np.ndarray]:
    raw = figure_tables._numeric(figure_tables._read_design("KZ198"), [config.COL_STIM])
    raw = raw[~raw[config.COL_WELL].isin(figure_tables.FIG1_KZ198_EXCLUSIONS)]
    no_inhibitor = raw[config.COL_INHIB_NAME].astype(str) == "0"
    unstim = raw.loc[no_inhibitor & (raw[config.COL_STIM] == 0), readout].values
    stimulated = raw.loc[no_inhibitor & (raw[config.COL_STIM] != 0), readout].values
    return gate_log(unstim), gate_log(stimulated)


def bounds_for(readout: str) -> dict[str, float]:
    return figure_tables.figure1_histogram_bounds(readout)


def fraction_hist(
    ax,
    values: np.ndarray,
    *,
    color: str,
    alpha: float = 1.0,
    histtype: str = "stepfilled",
    lw: float = 0.8,
) -> None:
    weights = np.full(values.shape, 1.0 / max(len(values), 1))
    ax.hist(values, bins=BINS, weights=weights, color=color, alpha=alpha,
            histtype=histtype, lw=lw)


def format_hist_axis(ax, xlabel: bool = False) -> None:
    ax.set_xlim(*XLIM)
    ax.set_yticks([])
    ax.set_xticks(list(XTICKS))
    if xlabel:
        ax.set_xlabel("log10 nuclear fluorescence intensity")
    else:
        ax.set_xticklabels([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("0.35")
    ax.tick_params(axis="x", length=3, width=0.8, color="0.35")


def cytokine_histograms(data: pd.DataFrame, readout: str) -> list[tuple]:
    series = CYTOKINE_SERIES[readout]
    histograms: list[tuple] = []
    for dose, color in zip(series["doses"], CYTOKINE_COLORS):
        values = data.loc[
            (data["panel"] == series["panel"]) & (data["condition_value"] == dose),
            "intensity_raw",
        ].to_numpy()
        log_values = gate_log(values)
        weights = np.full(log_values.shape, 1.0 / max(len(log_values), 1))
        heights, edges = np.histogram(log_values, bins=BINS, weights=weights)
        label = f"{series['name']} {dose} {series['unit']}"
        histograms.append((label, color, heights, edges, "right"))

    _, full_stim = kz198_reference(readout)
    weights = np.full(full_stim.shape, 1.0 / max(len(full_stim), 1))
    heights, edges = np.histogram(full_stim, bins=BINS, weights=weights)
    histograms.append((
        "full stim.\nIFN-gamma 10 ng/mL\nIL-10 100 ng/mL",
        FULL_STIM_COLOR,
        heights,
        edges,
        "left",
    ))
    return histograms


def draw_cytokine_panel(fig, cell, data: pd.DataFrame, readout: str):
    histograms = cytokine_histograms(data, readout)
    rows = cell.subgridspec(len(histograms), 1, hspace=0.14)
    first_ax = None

    for i, (label, color, heights, edges, label_side) in enumerate(histograms):
        ax = fig.add_subplot(rows[i])
        if first_ax is None:
            first_ax = ax
        ax.stairs(heights, edges, color=color, lw=1.25)
        ax.fill_between(edges[:-1], heights, step="post", color=color, alpha=0.10)
        row_max = float(np.max(heights)) if len(heights) else 0.0
        ax.set_ylim(0, row_max * 1.08 if row_max else 1)
        x_text = 0.02 if label_side == "left" else 0.98
        ha_text = "left" if label_side == "left" else "right"
        y_text = 0.93 if label_side == "left" else 0.76
        ax.text(
            x_text,
            y_text,
            label,
            transform=ax.transAxes,
            color=color,
            fontsize=7.3,
            ha=ha_text,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.70, "edgecolor": "none", "pad": 0.5},
        )
        if i == 0:
            ax.set_title(READOUT_LABEL[readout], fontweight="bold", pad=6)
        if i == len(histograms) // 2:
            ax.set_ylabel("fraction of cells", fontsize=8.5, labelpad=24)
        format_hist_axis(ax, xlabel=False)
        if i != len(histograms) - 1:
            ax.tick_params(axis="x", length=0)

    return first_ax


def draw_reference_panel(ax, readout: str) -> None:
    bounds = bounds_for(readout)
    unstim, stimulated = kz198_reference(readout)
    ax.axvspan(XLIM[0], bounds["complete"], color=COMPLETE, alpha=0.12, lw=0)
    ax.axvspan(bounds["complete"], bounds["partial"], color=PARTIAL, alpha=0.08, lw=0)
    fraction_hist(ax, stimulated, color=REFERENCE_STIM, alpha=0.20, histtype="stepfilled")
    fraction_hist(ax, unstim, color=COMPLETE, alpha=0.25, histtype="stepfilled")
    ax.axvline(bounds["complete"], color=COMPLETE, lw=1.2)
    ax.axvline(bounds["partial"], color=PARTIAL, lw=1.2)
    ax.text(0.02, 0.93, "95th pct\nnon-stim.", transform=ax.transAxes,
            color=COMPLETE, fontsize=8.5, va="top")
    ax.text(0.98, 0.93, "5th pct\nfull stim.", transform=ax.transAxes,
            color=PARTIAL, fontsize=8.5, ha="right", va="top")
    ax.text(0.02, 0.12, "non-stim.\nIFN-gamma 0 ng/mL\nIL-10 0 ng/mL",
            transform=ax.transAxes, color=COMPLETE, fontsize=7.5, va="bottom")
    ax.text(0.98, 0.12, "full stim.\nIFN-gamma 10 ng/mL\nIL-10 100 ng/mL",
            transform=ax.transAxes, color="0.25", fontsize=7.5, ha="right", va="bottom")
    format_hist_axis(ax)


def draw_dose_histograms(fig, cell, data: pd.DataFrame, readout: str) -> None:
    bounds = bounds_for(readout)
    panel = PANEL_FROM_READOUT[readout]
    rows = cell.subgridspec(len(CERDULATINIB_DOSES), 1, hspace=0.18)
    for i, dose in enumerate(CERDULATINIB_DOSES):
        ax = fig.add_subplot(rows[i])
        values = data.loc[
            (data["panel"] == panel) & (data["condition_value"] == dose),
            "intensity_raw",
        ].to_numpy()
        log_values = gate_log(values)
        ax.axvspan(XLIM[0], bounds["complete"], color=COMPLETE, alpha=0.12, lw=0)
        ax.axvspan(bounds["complete"], bounds["partial"], color=PARTIAL, alpha=0.08, lw=0)
        fraction_hist(ax, log_values, color=RESIDUAL, alpha=0.85)
        ax.axvline(bounds["complete"], color=COMPLETE, lw=1.0)
        ax.axvline(bounds["partial"], color=PARTIAL, lw=1.0)
        ax.text(0.98, 0.80, f"{dose} nM", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5)
        format_hist_axis(ax, xlabel=(i == len(CERDULATINIB_DOSES) - 1))


def state_fractions(data: pd.DataFrame, readout: str) -> pd.DataFrame:
    bounds = bounds_for(readout)
    panel = PANEL_FROM_READOUT[readout]
    rows = []
    for dose in CERDULATINIB_DOSES:
        values = data.loc[
            (data["panel"] == panel) & (data["condition_value"] == dose),
            "intensity_raw",
        ].to_numpy()
        log_values = gate_log(values)
        complete = float(np.mean(log_values <= bounds["complete"]))
        partial_total = float(np.mean(log_values <= bounds["partial"]))
        partial_only = max(0.0, partial_total - complete)
        residual = max(0.0, 1.0 - partial_total)
        rows.append({
            "dose": dose,
            "complete": complete,
            "partial_only": partial_only,
            "above_partial": residual,
        })
    return pd.DataFrame(rows)


def draw_state_bars(ax, data: pd.DataFrame, readout: str) -> None:
    frac = state_fractions(data, readout)
    x = np.arange(len(frac))
    ax.bar(x, frac["complete"], color=COMPLETE, label="complete-inhibition state")
    ax.bar(x, frac["partial_only"], bottom=frac["complete"], color=PARTIAL,
           label="partial-only state")
    bottom = frac["complete"] + frac["partial_only"]
    ax.bar(x, frac["above_partial"], bottom=bottom, color=RESIDUAL,
           label="above partial threshold")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(frac["dose"], rotation=0)
    ax.set_xlabel("cerdulatinib (nM)")
    ax.set_ylabel("fraction of cells")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=0.7)


def add_panel_letter(fig, ax, letter: str) -> None:
    box = ax.get_position()
    fig.text(0.025, box.y1 + 0.01, letter, fontsize=15, fontweight="bold",
             va="bottom", ha="left")


def main() -> int:
    data = load_source()
    FIGURES.mkdir(exist_ok=True)

    fig = plt.figure(figsize=FIGSIZE, constrained_layout=False)
    grid = fig.add_gridspec(
        4,
        2,
        height_ratios=[2.75, 1.15, 3.4, 1.4],
        hspace=0.55,
        wspace=0.28,
        left=0.08,
        right=0.985,
        top=0.945,
        bottom=0.13,
    )

    cytokine_axes = []
    reference_axes = []
    inhibitor_axes = []
    occupancy_axes = []
    for j, readout in enumerate(READOUTS):
        cytokine_axes.append(draw_cytokine_panel(fig, grid[0, j], data, readout))

        ax_reference = fig.add_subplot(grid[1, j])
        draw_reference_panel(ax_reference, readout)
        reference_axes.append(ax_reference)

        draw_dose_histograms(fig, grid[2, j], data, readout)
        ax_inhibitor = fig.add_subplot(grid[2, j], frameon=False)
        ax_inhibitor.set_axis_off()
        inhibitor_axes.append(ax_inhibitor)

        ax_occupancy = fig.add_subplot(grid[3, j])
        draw_state_bars(ax_occupancy, data, readout)
        occupancy_axes.append(ax_occupancy)

    add_panel_letter(fig, cytokine_axes[0], "A")
    add_panel_letter(fig, reference_axes[0], "B")
    add_panel_letter(fig, inhibitor_axes[0], "C")
    add_panel_letter(fig, occupancy_axes[0], "D")

    handles, labels = occupancy_axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.55, 0.008),
        fontsize=9,
        title=(
            "Blue = complete-inhibition state; red = partial-only state; gray = above "
            "partial threshold; red + gray = residual pSTAT-positive cells."
        ),
    )
    legend.get_title().set_fontsize(9)

    out = FIGURES / "Figure_1.png"
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
