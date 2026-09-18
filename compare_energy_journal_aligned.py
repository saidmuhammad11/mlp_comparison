#!/usr/bin/env python3
"""
Journal-ready comparison plots and numerical summaries for HistGBDT-PMS.

This revision uses one common analysis pipeline for every reported value:

1. Each log is converted to elapsed time relative to its own first sample.
2. Both policies are interpolated onto the same fixed-duration time grid.
3. The absolute traces, baseline-referenced deltas, table means, percentage
   changes, cache summaries, box plots, and energy values all use those same
   aligned arrays.
4. Means are time-weighted with trapezoidal integration.
5. Energy is integrated from the same aligned power arrays.
6. Consistency checks stop execution when the figure mean delta does not equal
   the difference between the two reported means, or when a nominal 300-s log
   is materially incomplete.

Edit only the CONFIGURATION section for each experiment.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================ CONFIGURATION ============================
BASELINE_FILE = 'HistGBDTdisable_mixed_new2.json'
PROPOSED_FILE = 'HistGBDTenable_mixed_new2.json' 

# WORKLOAD_NAME = "CPU_Intensive"
# WORKLOAD_NAME = "CPU_Intensive"
WORKLOAD_NAME = "Mixed"

BASELINE_LABEL = "Default Windows Medium"
PROPOSED_LABEL = "HistGBDT-PMS"

OUTPUT_DIR = Path("journal_figures")
OUTPUT_PREFIX = "HistGBDT_pms"

# Fixed evaluation interval used by the manuscript.
ANALYSIS_DURATION_S = 300.0

# Both policies are resampled to this common uniform grid. A 1-s grid is much
# finer than the original telemetry interval and keeps all outputs consistent.
COMMON_GRID_STEP_S = 1.0

# A nominal 300-s run may have its last telemetry sample slightly before 300 s.
# The script permits only a small endpoint gap, based on the observed median
# sampling interval. A materially shorter run, such as 277 s, raises an error.
STRICT_DURATION_CHECK = True
ENDPOINT_TOLERANCE_MULTIPLIER = 1.5
MIN_ENDPOINT_TOLERANCE_S = 3.0

# Linear interpolation is used between measured numeric samples. If the final
# sample falls within the permitted endpoint tolerance, its value is held to
# exactly ANALYSIS_DURATION_S by numpy.interp.

# Export both formats. PDF is recommended for LaTeX submission.
VECTOR_FORMATS = ("pdf", "svg")

# Figure preview and saving behavior:
# "ask"    -> show each figure, then ask whether to save it
# "always" -> show each figure and save automatically
# "never"  -> show each figure without saving
SAVE_MODE = "ask"
SHOW_FIGURES = True

# The comprehensive figure contains eight panels and spans both columns.
FULL_WIDTH_IN = 7.16

# Metrics retained in the combined absolute-and-delta figure.
COMBINED_METRICS = [
    ("cpu_power", "CPU power (W)"),
    ("ipc", "IPC"),
    ("cpu_frequency", "CPU freq. (MHz)"),
    ("cpu_usage_overall", "CPU usage (%)"),
]

# General font sizes at the final printed size.
BASE_FONT_SIZE = 10.5
PANEL_LABEL_SIZE = BASE_FONT_SIZE

# Power-state distribution typography.
LEVEL_LABEL_SIZE = 11.2
LEVEL_TICK_SIZE = 10.2
LEVEL_TITLE_SIZE = 11.2
LEVEL_COUNT_SIZE = 10.2
LEVEL_PANEL_SIZE = 11.2

# Cache hit/miss summary typography.
SUMMARY_LABEL_SIZE = 11.8
SUMMARY_TICK_SIZE = 10.8
SUMMARY_LEGEND_SIZE = 11.8
SUMMARY_PANEL_SIZE = 11.8

BASELINE_COLOR = "#4169E1"
PROPOSED_COLOR = "#FF69B4"

STATE_STYLES = {
    "Low": {"color": "#4C78A8", "hatch": "///"},
    "Medium": {"color": "#F2A541", "hatch": "xx"},
    "High": {"color": "#59A14F", "hatch": "..."},
    "Fixed": {"color": "#9D9D9D", "hatch": "\\\\"},
}

CACHE_STYLES = {
    "L2": {"color": "#59A14F"},
    "L3": {"color": "#E15759"},
}

POLICY_STYLES = {
    BASELINE_LABEL: {"hatch": "///"},
    PROPOSED_LABEL: {"hatch": "xx"},
}

MAX_MEMORY_BANDWIDTH = 50.0  # GB/s, used only for workload classification

# =====================================================================


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Return trapezoidal integration with compatibility for older NumPy."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def configure_matplotlib() -> None:
    """Apply a compact, publication-oriented Matplotlib configuration."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": BASE_FONT_SIZE,
            "axes.titlesize": BASE_FONT_SIZE,
            "xtick.labelsize": BASE_FONT_SIZE,
            "ytick.labelsize": BASE_FONT_SIZE,
            "legend.fontsize": BASE_FONT_SIZE,
            "axes.linewidth": 0.8,
            "lines.linewidth": 0.9,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_json_log(filename: str) -> pd.DataFrame:
    """Load a line-delimited JSON log and prepare unique elapsed timestamps."""
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Input log not found: {path.resolve()}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping invalid JSON at line {line_number}: {exc}")
                continue

            entry.setdefault("dvfs_level", "Fixed")
            records.append(entry)

    if not records:
        raise ValueError(f"No valid records were found in {filename}")

    frame = pd.DataFrame(records)
    if "timestamp" not in frame.columns:
        raise KeyError(f"{filename} does not contain a 'timestamp' field")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    duplicate_count = int(frame["timestamp"].duplicated(keep="last").sum())
    if duplicate_count:
        print(
            f"Warning: {filename} contains {duplicate_count} duplicate "
            "timestamp(s); keeping the last record at each timestamp."
        )
        frame = frame.drop_duplicates(subset="timestamp", keep="last")
        frame = frame.reset_index(drop=True)

    frame["elapsed_s"] = (
        frame["timestamp"] - frame["timestamp"].iloc[0]
    ).dt.total_seconds()

    return frame


def estimate_sampling_interval_s(frame: pd.DataFrame) -> float:
    """Estimate the median positive telemetry interval in seconds."""
    elapsed = frame["elapsed_s"].to_numpy(dtype=float)
    differences = np.diff(elapsed)
    positive = differences[np.isfinite(differences) & (differences > 0.0)]
    if positive.size == 0:
        return float("nan")
    return float(np.median(positive))


def endpoint_tolerance_s(frame: pd.DataFrame) -> float:
    """Return the maximum permitted start/end coverage gap."""
    median_interval = estimate_sampling_interval_s(frame)
    if not np.isfinite(median_interval):
        return MIN_ENDPOINT_TOLERANCE_S
    return max(
        MIN_ENDPOINT_TOLERANCE_S,
        ENDPOINT_TOLERANCE_MULTIPLIER * median_interval,
    )


def validate_log_duration(frame: pd.DataFrame, label: str) -> None:
    """Ensure that a log materially covers the configured fixed duration."""
    if frame.empty:
        raise ValueError(f"{label} log is empty")

    observed_end = float(frame["elapsed_s"].iloc[-1])
    tolerance = endpoint_tolerance_s(frame)

    if STRICT_DURATION_CHECK and observed_end < ANALYSIS_DURATION_S - tolerance:
        raise ValueError(
            f"{label} log covers only {observed_end:.2f} s, but the analysis "
            f"requires {ANALYSIS_DURATION_S:.2f} s. The permitted endpoint "
            f"gap is {tolerance:.2f} s. Repeat or repair this run rather than "
            "silently comparing unequal durations."
        )


def build_common_time_grid() -> np.ndarray:
    """Return the common fixed-duration analysis grid in seconds."""
    if ANALYSIS_DURATION_S <= 0.0:
        raise ValueError("ANALYSIS_DURATION_S must be positive")
    if COMMON_GRID_STEP_S <= 0.0:
        raise ValueError("COMMON_GRID_STEP_S must be positive")

    points = int(round(ANALYSIS_DURATION_S / COMMON_GRID_STEP_S))
    return np.linspace(0.0, ANALYSIS_DURATION_S, points + 1)


def numeric_metric_samples(
    frame: pd.DataFrame,
    column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted, unique finite numeric samples for one metric."""
    if column not in frame.columns:
        raise KeyError(f"Missing metric column: {column}")

    values = pd.to_numeric(frame[column], errors="coerce")
    clean = pd.DataFrame(
        {
            "elapsed_s": frame["elapsed_s"],
            "value": values,
        }
    ).dropna()

    clean = clean[np.isfinite(clean["elapsed_s"]) & np.isfinite(clean["value"])]
    clean = clean.sort_values("elapsed_s")
    clean = clean.drop_duplicates(subset="elapsed_s", keep="last")

    if len(clean) < 2:
        raise ValueError(f"Metric '{column}' has fewer than two valid samples")

    return (
        clean["elapsed_s"].to_numpy(dtype=float),
        clean["value"].to_numpy(dtype=float),
    )


def interpolate_numeric_metric(
    frame: pd.DataFrame,
    column: str,
    time_grid_s: np.ndarray,
    label: str,
) -> np.ndarray:
    """Interpolate one numeric metric onto the shared analysis grid."""
    sample_time_s, sample_values = numeric_metric_samples(frame, column)
    tolerance = endpoint_tolerance_s(frame)

    first_time = float(sample_time_s[0])
    last_time = float(sample_time_s[-1])

    if STRICT_DURATION_CHECK and first_time > tolerance:
        raise ValueError(
            f"{label} metric '{column}' begins at {first_time:.2f} s, beyond "
            f"the permitted start gap of {tolerance:.2f} s."
        )

    if STRICT_DURATION_CHECK and last_time < ANALYSIS_DURATION_S - tolerance:
        raise ValueError(
            f"{label} metric '{column}' ends at {last_time:.2f} s, but the "
            f"analysis requires {ANALYSIS_DURATION_S:.2f} s. The permitted "
            f"endpoint gap is {tolerance:.2f} s."
        )

    # numpy.interp performs one-dimensional linear interpolation. Outside the
    # measured domain it holds the endpoint value. Strict duration checks above
    # limit that endpoint hold to a small, controlled interval.
    aligned = np.interp(
        time_grid_s,
        sample_time_s,
        sample_values,
    )

    if not np.all(np.isfinite(aligned)):
        raise ValueError(f"Aligned metric '{column}' contains non-finite values")

    return aligned.astype(float, copy=False)


def align_categorical_state(
    frame: pd.DataFrame,
    time_grid_s: np.ndarray,
) -> np.ndarray:
    """Align categorical controller states by last-observation carry-forward."""
    if "dvfs_level" not in frame.columns:
        return np.full(time_grid_s.shape, "Fixed", dtype=object)

    clean = frame[["elapsed_s", "dvfs_level"]].dropna().copy()
    if clean.empty:
        return np.full(time_grid_s.shape, "Fixed", dtype=object)

    clean = clean.sort_values("elapsed_s")
    clean = clean.drop_duplicates(subset="elapsed_s", keep="last")

    sample_time_s = clean["elapsed_s"].to_numpy(dtype=float)
    sample_states = clean["dvfs_level"].astype(str).to_numpy(dtype=object)

    indices = np.searchsorted(sample_time_s, time_grid_s, side="right") - 1
    indices = np.clip(indices, 0, len(sample_states) - 1)
    return sample_states[indices]


def summarize_aligned_metric(
    time_s: np.ndarray,
    baseline_values: np.ndarray,
    proposed_values: np.ndarray,
) -> dict[str, float]:
    """Calculate time-weighted means, delta, and percentage change."""
    if not (
        len(time_s) == len(baseline_values) == len(proposed_values)
        and len(time_s) >= 2
    ):
        raise ValueError("Aligned time and metric arrays have inconsistent lengths")

    duration_s = float(time_s[-1] - time_s[0])
    if duration_s <= 0.0:
        raise ValueError("Aligned analysis duration must be positive")

    delta_values = baseline_values - proposed_values

    baseline_mean = trapezoid(baseline_values, time_s) / duration_s
    proposed_mean = trapezoid(proposed_values, time_s) / duration_s
    mean_delta = trapezoid(delta_values, time_s) / duration_s

    expected_delta = baseline_mean - proposed_mean
    if not np.isclose(mean_delta, expected_delta, rtol=1e-11, atol=1e-11):
        raise RuntimeError(
            "Mean-delta consistency check failed: mean(baseline - proposed) "
            "does not equal mean(baseline) - mean(proposed)."
        )

    percentage_change = (
        100.0 * (proposed_mean - baseline_mean) / baseline_mean
        if baseline_mean != 0.0
        else float("nan")
    )

    return {
        "baseline_mean": float(baseline_mean),
        "proposed_mean": float(proposed_mean),
        "mean_delta": float(mean_delta),
        "percentage_change": float(percentage_change),
    }


def preview_and_maybe_save(fig: plt.Figure, stem: str) -> None:
    """Display and save a figure according to SAVE_MODE."""
    if SHOW_FIGURES:
        plt.show(block=True)

    normalized_mode = SAVE_MODE.strip().lower()
    if normalized_mode not in {"ask", "always", "never"}:
        raise ValueError('SAVE_MODE must be "ask", "always", or "never".')

    should_save = normalized_mode == "always"

    if normalized_mode == "ask":
        try:
            answer = input(
                f"Save figure '{stem}' as PDF and SVG? [y/N]: "
            ).strip().lower()
            should_save = answer in {"y", "yes"}
        except EOFError:
            should_save = False
            print(
                f"No interactive input was available. Figure '{stem}' "
                "was not saved."
            )

    if should_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for extension in VECTOR_FORMATS:
            output_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{stem}.{extension}"
            fig.savefig(
                output_path,
                format=extension,
                bbox_inches="tight",
                pad_inches=0.02,
                transparent=False,
            )
            print(f"Saved: {output_path}")
    else:
        print(f"Skipped saving: {stem}")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """Place a journal-style panel label at the upper-left of an axis."""
    ax.text(
        -0.10,
        1.025,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def align_ylabels_by_column(
    fig: plt.Figure,
    axes: np.ndarray,
    left_x: float = -0.14,
    right_x: float = -0.13,
) -> None:
    """Align y-axis headings at fixed x coordinates by subplot column."""
    axes_array = np.asarray(axes, dtype=object)
    if axes_array.ndim == 1:
        axes_array = axes_array.reshape(1, -1)

    n_rows, n_cols = axes_array.shape
    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes_array[row, col]
            if not ax.get_visible():
                continue
            ax.yaxis.set_label_coords(left_x if col == 0 else right_x, 0.5)

    for col in range(n_cols):
        visible_column_axes = [
            axes_array[row, col]
            for row in range(n_rows)
            if axes_array[row, col].get_visible()
        ]
        if visible_column_axes:
            fig.align_ylabels(visible_column_axes)


def available_metric_pairs(
    baseline: pd.DataFrame,
    proposed: pd.DataFrame,
) -> list[tuple[str, str]]:
    """Return metrics available in both logs."""
    candidates = [
        ("cpu_power", "CPU power (W)"),
        ("ipc", "IPC"),
        ("cpu_frequency", "CPU freq. (MHz)"),
        ("memory_usage", "Memory usage (%)"),
        ("l2_miss_rate", "L2 miss rate"),
        ("l3_miss_rate", "L3 miss rate"),
        ("cpu_temperature", "CPU temp. (°C)"),
        ("cpu_usage_overall", "CPU usage (%)"),
    ]
    return [
        pair
        for pair in candidates
        if pair[0] in baseline.columns and pair[0] in proposed.columns
    ]


class JournalComparator:
    """Create all figures and tables from one aligned fixed-duration dataset."""

    def __init__(
        self,
        baseline: pd.DataFrame,
        proposed: pd.DataFrame,
        workload_name: str,
    ) -> None:
        self.baseline = baseline
        self.proposed = proposed
        self.workload_name = workload_name
        self.time_s = build_common_time_grid()
        self.time_min = self.time_s / 60.0
        self.metrics = available_metric_pairs(baseline, proposed)
        self._aligned_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._stats_cache: dict[str, dict[str, float]] = {}
        self.workload_type = self.detect_workload_type()

    def aligned_metric(self, column: str) -> tuple[np.ndarray, np.ndarray]:
        """Return cached baseline/proposed arrays on the common time grid."""
        if column not in self._aligned_cache:
            baseline_values = interpolate_numeric_metric(
                self.baseline,
                column,
                self.time_s,
                BASELINE_LABEL,
            )
            proposed_values = interpolate_numeric_metric(
                self.proposed,
                column,
                self.time_s,
                PROPOSED_LABEL,
            )
            self._aligned_cache[column] = (baseline_values, proposed_values)
        return self._aligned_cache[column]

    def metric_statistics(self, column: str) -> dict[str, float]:
        """Return cached statistics for one aligned metric."""
        if column not in self._stats_cache:
            baseline_values, proposed_values = self.aligned_metric(column)
            self._stats_cache[column] = summarize_aligned_metric(
                self.time_s,
                baseline_values,
                proposed_values,
            )
        return self._stats_cache[column]

    def detect_workload_type(self) -> str:
        """Classify the logged workload using the existing heuristic."""
        avg_l3_miss = (
            self.metric_statistics("l3_miss_rate")["proposed_mean"]
            if "l3_miss_rate" in self.proposed.columns
            and "l3_miss_rate" in self.baseline.columns
            else 0.0
        )
        avg_mem_bw = (
            self.metric_statistics("memory_bandwidth")["proposed_mean"]
            if "memory_bandwidth" in self.proposed.columns
            and "memory_bandwidth" in self.baseline.columns
            else 0.0
        )
        if avg_l3_miss > 0.05 and avg_mem_bw > 0.6 * MAX_MEMORY_BANDWIDTH:
            return "memory-intensive"
        return "CPU-intensive"

    def create_combined_absolute_delta_plot(self) -> None:
        """Create the aligned absolute traces and baseline-referenced deltas."""
        metrics = [
            (column, ylabel)
            for column, ylabel in COMBINED_METRICS
            if column in self.baseline.columns and column in self.proposed.columns
        ]
        if not metrics:
            raise ValueError(
                "None of the requested combined metrics are available in both logs."
            )

        rows = len(metrics)
        fig_height = max(6.8, rows * 1.68)
        fig, axes = plt.subplots(
            rows,
            2,
            figsize=(FULL_WIDTH_IN, fig_height),
            sharex="col",
            squeeze=False,
            constrained_layout=False,
        )
        fig.subplots_adjust(
            left=0.10,
            right=0.99,
            bottom=0.10,
            top=0.85,
            hspace=0.62,
            wspace=0.40,
        )

        legend_handles = None
        legend_labels = None

        for row, (column, ylabel) in enumerate(metrics):
            absolute_ax = axes[row, 0]
            delta_ax = axes[row, 1]

            baseline_values, proposed_values = self.aligned_metric(column)
            stats = self.metric_statistics(column)
            delta_values = baseline_values - proposed_values

            baseline_line, = absolute_ax.plot(
                self.time_min,
                baseline_values,
                label=BASELINE_LABEL,
                color=BASELINE_COLOR,
            )
            proposed_line, = absolute_ax.plot(
                self.time_min,
                proposed_values,
                label=PROPOSED_LABEL,
                color=PROPOSED_COLOR,
            )

            absolute_ax.set_ylabel(ylabel, fontsize=BASE_FONT_SIZE, labelpad=2.0)
            absolute_ax.tick_params(
                axis="both",
                which="major",
                labelsize=BASE_FONT_SIZE,
                pad=1.5,
            )
            absolute_ax.grid(True)
            add_panel_label(absolute_ax, f"({chr(ord('a') + 2 * row)})")

            if legend_handles is None:
                legend_handles = [baseline_line, proposed_line]
                legend_labels = [BASELINE_LABEL, PROPOSED_LABEL]

            delta_ax.plot(self.time_min, delta_values, color=BASELINE_COLOR)
            delta_ax.axhline(0.0, color="black", linewidth=0.7)
            delta_ax.set_ylabel(
                rf"$\Delta$ {ylabel}",
                fontsize=BASE_FONT_SIZE,
                labelpad=2.0,
            )
            delta_ax.tick_params(
                axis="both",
                which="major",
                labelsize=BASE_FONT_SIZE,
                pad=1.5,
            )
            delta_ax.grid(True)
            add_panel_label(delta_ax, f"({chr(ord('a') + 2 * row + 1)})")

            finite_delta = delta_values[np.isfinite(delta_values)]
            if finite_delta.size:
                data_min = float(np.min(finite_delta))
                data_max = float(np.max(finite_delta))
                display_min = min(data_min, 0.0)
                display_max = max(data_max, 0.0)
                display_span = display_max - display_min
                if display_span <= 0.0:
                    display_span = max(abs(display_max), 1.0)

                delta_ax.set_ylim(
                    display_min - 0.06 * display_span,
                    display_max + 0.28 * display_span,
                )
                delta_ax.text(
                    float(self.time_min[-1]),
                    display_max + 0.11 * display_span,
                    rf"Mean $\Delta X$ = {stats['mean_delta']:.2f}",
                    ha="right",
                    va="bottom",
                    fontsize=BASE_FONT_SIZE,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.88,
                        "pad": 1.5,
                    },
                )

        axes[0, 0].set_title(
            "Absolute runtime measurements",
            fontsize=BASE_FONT_SIZE,
            y=1.18,
            pad=0,
        )
        axes[0, 1].set_title(
            r"Baseline-referenced difference, $\Delta X$",
            fontsize=BASE_FONT_SIZE,
            y=1.18,
            pad=0,
        )
        axes[-1, 0].set_xlabel("Time (min)", fontsize=BASE_FONT_SIZE)
        axes[-1, 1].set_xlabel("Time (min)", fontsize=BASE_FONT_SIZE)

        align_ylabels_by_column(fig, axes, left_x=-0.15, right_x=-0.16)

        if legend_handles is not None:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                ncol=2,
                frameon=False,
                bbox_to_anchor=(0.5, 0.965),
                borderaxespad=0.0,
                columnspacing=1.15,
                handlelength=1.8,
                handletextpad=0.5,
                fontsize=BASE_FONT_SIZE,
                prop={"size": BASE_FONT_SIZE, "weight": "medium"},
            )

        preview_and_maybe_save(
            fig,
            "combined_absolute_baseline_referenced_delta",
        )
        plt.close(fig)

    def create_box_plot(self) -> None:
        """Create a compact distribution comparison from aligned samples."""
        requested = [
            ("cpu_power", "CPU power (W)"),
            ("ipc", "IPC"),
            ("l2_miss_rate", "L2 miss rate"),
            ("l3_miss_rate", "L3 miss rate"),
        ]
        metrics = [
            pair
            for pair in requested
            if pair[0] in self.baseline.columns and pair[0] in self.proposed.columns
        ]
        if not metrics:
            return

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(FULL_WIDTH_IN, 4.6),
            squeeze=False,
            constrained_layout=False,
        )
        fig.subplots_adjust(
            left=0.10,
            right=0.99,
            bottom=0.15,
            top=0.96,
            hspace=0.42,
            wspace=0.34,
        )

        for index, (column, ylabel) in enumerate(metrics):
            row, col = divmod(index, 2)
            ax = axes[row, col]
            baseline_values, proposed_values = self.aligned_metric(column)
            box = ax.boxplot(
                [baseline_values, proposed_values],
                tick_labels=[BASELINE_LABEL, PROPOSED_LABEL],
                patch_artist=True,
                widths=0.55,
            )
            for patch, color in zip(
                box["boxes"],
                [BASELINE_COLOR, PROPOSED_COLOR],
            ):
                patch.set_facecolor(color)
                patch.set_alpha(0.45)

            ax.set_ylabel(ylabel, labelpad=2.0)
            ax.tick_params(axis="x", rotation=12, pad=2.0)
            ax.tick_params(axis="y", pad=1.5)
            ax.grid(True, axis="y")
            add_panel_label(ax, f"({chr(ord('a') + index)})")

        for index in range(len(metrics), 4):
            row, col = divmod(index, 2)
            axes[row, col].set_visible(False)

        align_ylabels_by_column(fig, axes, left_x=-0.14, right_x=-0.13)
        preview_and_maybe_save(fig, "boxplots")
        plt.close(fig)

    def create_performance_level_plot(self) -> None:
        """Plot time share in each controller state on the common grid."""
        if "dvfs_level" not in self.proposed.columns:
            return

        baseline_states = align_categorical_state(self.baseline, self.time_s)
        proposed_states = align_categorical_state(self.proposed, self.time_s)

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(FULL_WIDTH_IN, 3.85),
            constrained_layout=False,
        )
        fig.subplots_adjust(
            left=0.10,
            right=0.99,
            bottom=0.20,
            top=0.88,
            wspace=0.34,
        )

        self._plot_level_distribution(
            axes[0],
            baseline_states,
            BASELINE_LABEL,
            BASELINE_COLOR,
            "(a)",
        )
        self._plot_level_distribution(
            axes[1],
            proposed_states,
            PROPOSED_LABEL,
            PROPOSED_COLOR,
            "(b)",
        )

        align_ylabels_by_column(
            fig,
            np.asarray(axes).reshape(1, -1),
            left_x=-0.13,
            right_x=-0.13,
        )
        preview_and_maybe_save(fig, "level_distribution")
        plt.close(fig)

    @staticmethod
    def _plot_level_distribution(
        ax: plt.Axes,
        states: np.ndarray,
        label: str,
        color: str,
        panel_label: str,
    ) -> None:
        counts = Counter(states.tolist())
        preferred_order = ["Low", "Medium", "High", "Fixed"]
        categories = sorted(
            counts,
            key=lambda value: (
                preferred_order.index(value)
                if value in preferred_order
                else len(preferred_order)
            ),
        )
        total = max(len(states), 1)
        percentages = [100.0 * counts[category] / total for category in categories]

        bar_colors = [
            STATE_STYLES.get(category, {"color": color})["color"]
            for category in categories
        ]
        bars = ax.bar(
            categories,
            percentages,
            color=bar_colors,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.7,
        )

        for bar, category, percentage in zip(bars, categories, percentages):
            bar.set_hatch(STATE_STYLES.get(category, {}).get("hatch", ""))
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                percentage,
                f"{percentage:.1f}%",
                ha="center",
                va="bottom",
                fontsize=LEVEL_COUNT_SIZE,
            )

        ax.set_ylabel("Analysis-time share (%)", fontsize=LEVEL_LABEL_SIZE, labelpad=4.0)
        ax.set_title(label, fontsize=LEVEL_TITLE_SIZE, pad=6)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=LEVEL_TICK_SIZE,
            pad=2.0,
        )
        ax.grid(True, axis="y")
        ax.text(
            -0.11,
            1.04,
            panel_label,
            transform=ax.transAxes,
            fontsize=LEVEL_PANEL_SIZE,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    def create_cache_plot(self) -> None:
        """Create aligned mean L2/L3 hit-rate and miss-rate summaries."""
        from matplotlib.patches import Patch

        required = {"l2_miss_rate", "l3_miss_rate"}
        if not required.issubset(self.baseline.columns) or not required.issubset(
            self.proposed.columns
        ):
            return

        cache_levels = ["L2", "L3"]
        baseline_miss = [
            self.metric_statistics("l2_miss_rate")["baseline_mean"],
            self.metric_statistics("l3_miss_rate")["baseline_mean"],
        ]
        proposed_miss = [
            self.metric_statistics("l2_miss_rate")["proposed_mean"],
            self.metric_statistics("l3_miss_rate")["proposed_mean"],
        ]
        baseline_hit = [1.0 - value for value in baseline_miss]
        proposed_hit = [1.0 - value for value in proposed_miss]

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(FULL_WIDTH_IN, 3.45),
            constrained_layout=False,
        )
        fig.subplots_adjust(
            left=0.09,
            right=0.99,
            bottom=0.20,
            top=0.68,
            wspace=0.38,
        )

        x = np.arange(len(cache_levels), dtype=float)
        width = 0.34
        panel_data = [
            (axes[0], baseline_hit, proposed_hit, "Mean hit rate", "(a)"),
            (axes[1], baseline_miss, proposed_miss, "Mean miss rate", "(b)"),
        ]

        for ax, baseline_values, proposed_values, ylabel, panel_label in panel_data:
            baseline_bars = ax.bar(
                x - width / 2,
                baseline_values,
                width,
                color=[CACHE_STYLES[level]["color"] for level in cache_levels],
                edgecolor="black",
                linewidth=0.85,
                alpha=0.72,
            )
            proposed_bars = ax.bar(
                x + width / 2,
                proposed_values,
                width,
                color=[CACHE_STYLES[level]["color"] for level in cache_levels],
                edgecolor="black",
                linewidth=0.85,
                alpha=1.0,
            )

            for bar in baseline_bars:
                bar.set_hatch(POLICY_STYLES[BASELINE_LABEL]["hatch"])
            for bar in proposed_bars:
                bar.set_hatch(POLICY_STYLES[PROPOSED_LABEL]["hatch"])

            ax.set_xticks(x, cache_levels)
            ax.set_ylabel(ylabel, fontsize=SUMMARY_LABEL_SIZE, labelpad=4.0)
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=SUMMARY_TICK_SIZE,
                pad=2.0,
            )
            ax.grid(True, axis="y")
            ax.text(
                -0.12,
                1.03,
                panel_label,
                transform=ax.transAxes,
                fontsize=SUMMARY_PANEL_SIZE,
                fontweight="bold",
                va="bottom",
                ha="left",
            )

        axes[0].set_ylim(0.0, 1.0)
        miss_upper = max(max(baseline_miss), max(proposed_miss))
        axes[1].set_ylim(0.0, max(0.10, miss_upper * 1.20))
        axes[0].yaxis.set_label_coords(-0.15, 0.5)
        axes[1].yaxis.set_label_coords(-0.15, 0.5)

        cache_handles = [
            Patch(
                facecolor=CACHE_STYLES["L2"]["color"],
                edgecolor="black",
                label="L2",
            ),
            Patch(
                facecolor=CACHE_STYLES["L3"]["color"],
                edgecolor="black",
                label="L3",
            ),
        ]
        policy_handles = [
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=POLICY_STYLES[BASELINE_LABEL]["hatch"],
                label=BASELINE_LABEL,
            ),
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=POLICY_STYLES[PROPOSED_LABEL]["hatch"],
                label=PROPOSED_LABEL,
            ),
        ]

        cache_legend = fig.legend(
            handles=cache_handles,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.28, 0.985),
            fontsize=SUMMARY_LEGEND_SIZE,
            title="Cache level",
            title_fontsize=SUMMARY_LEGEND_SIZE,
            handlelength=2.0,
            handletextpad=0.45,
            columnspacing=1.1,
        )
        fig.add_artist(cache_legend)
        fig.legend(
            handles=policy_handles,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.73, 0.985),
            fontsize=SUMMARY_LEGEND_SIZE,
            title="Policy",
            title_fontsize=SUMMARY_LEGEND_SIZE,
            handlelength=2.0,
            handletextpad=0.45,
            columnspacing=1.1,
        )

        preview_and_maybe_save(fig, "cache_hit_miss_summary")
        plt.close(fig)

    def export_summary_csv(self) -> None:
        """Write aligned, internally consistent numerical results to CSV."""
        metric_specs = [
            ("Average CPU power", "cpu_power", "W", 1.0),
            ("Average IPC", "ipc", "IPC", 1.0),
            ("Average CPU usage", "cpu_usage_overall", "%", 1.0),
            ("Average L2 miss rate", "l2_miss_rate", "ratio", 1.0),
            ("Average L3 miss rate", "l3_miss_rate", "ratio", 1.0),
            ("Average temperature", "cpu_temperature", "°C", 1.0),
            ("Average frequency", "cpu_frequency", "GHz", 1.0 / 1000.0),
        ]

        rows: list[dict[str, Any]] = []
        for display_name, column, unit, scale in metric_specs:
            if column not in self.baseline.columns or column not in self.proposed.columns:
                continue

            stats = self.metric_statistics(column)
            rows.append(
                {
                    "Metric": display_name,
                    "Unit": unit,
                    BASELINE_LABEL: stats["baseline_mean"] * scale,
                    PROPOSED_LABEL: stats["proposed_mean"] * scale,
                    "Baseline-referenced difference": stats["mean_delta"] * scale,
                    "Difference type": "Time-weighted mean delta",
                    "Change vs. baseline (%)": stats["percentage_change"],
                }
            )

        if "cpu_power" in self.baseline.columns and "cpu_power" in self.proposed.columns:
            baseline_power, proposed_power = self.aligned_metric("cpu_power")
            baseline_energy = trapezoid(baseline_power, self.time_s)
            proposed_energy = trapezoid(proposed_power, self.time_s)
            energy_delta = baseline_energy - proposed_energy
            energy_change = (
                100.0 * (proposed_energy - baseline_energy) / baseline_energy
                if baseline_energy != 0.0
                else float("nan")
            )

            duration_s = float(self.time_s[-1] - self.time_s[0])
            power_stats = self.metric_statistics("cpu_power")
            if not np.isclose(
                baseline_energy / duration_s,
                power_stats["baseline_mean"],
                rtol=1e-10,
                atol=1e-10,
            ):
                raise RuntimeError("Baseline power-energy consistency check failed")
            if not np.isclose(
                proposed_energy / duration_s,
                power_stats["proposed_mean"],
                rtol=1e-10,
                atol=1e-10,
            ):
                raise RuntimeError("Proposed power-energy consistency check failed")

            rows.append(
                {
                    "Metric": "Integrated CPU energy",
                    "Unit": "J",
                    BASELINE_LABEL: baseline_energy,
                    PROPOSED_LABEL: proposed_energy,
                    "Baseline-referenced difference": energy_delta,
                    "Difference type": "Integrated energy difference",
                    "Change vs. baseline (%)": energy_change,
                }
            )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.csv"
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

    def export_analysis_audit(self) -> None:
        """Write metadata that documents coverage and alignment decisions."""
        baseline_interval = estimate_sampling_interval_s(self.baseline)
        proposed_interval = estimate_sampling_interval_s(self.proposed)

        audit = {
            "workload_name": self.workload_name,
            "baseline_file": str(BASELINE_FILE),
            "proposed_file": str(PROPOSED_FILE),
            "baseline_label": BASELINE_LABEL,
            "proposed_label": PROPOSED_LABEL,
            "analysis_duration_s": ANALYSIS_DURATION_S,
            "common_grid_step_s": COMMON_GRID_STEP_S,
            "common_grid_points": int(len(self.time_s)),
            "strict_duration_check": STRICT_DURATION_CHECK,
            "baseline_raw_sample_count": int(len(self.baseline)),
            "proposed_raw_sample_count": int(len(self.proposed)),
            "baseline_observed_duration_s": float(self.baseline["elapsed_s"].iloc[-1]),
            "proposed_observed_duration_s": float(self.proposed["elapsed_s"].iloc[-1]),
            "baseline_median_sampling_interval_s": baseline_interval,
            "proposed_median_sampling_interval_s": proposed_interval,
            "baseline_endpoint_tolerance_s": endpoint_tolerance_s(self.baseline),
            "proposed_endpoint_tolerance_s": endpoint_tolerance_s(self.proposed),
            "numeric_alignment": "linear interpolation to common fixed grid",
            "categorical_alignment": "last observation carried forward",
            "mean_definition": "trapezoidal integral divided by fixed duration",
            "delta_definition": "baseline minus proposed on common grid",
            "energy_definition": "trapezoidal integral of aligned CPU power",
        }

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_analysis_audit.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(audit, handle, indent=2)
        print(f"Saved: {output_path}")

    def print_consistency_report(self) -> None:
        """Print the values that must match the figure and summary table."""
        print("\nAligned consistency report")
        print("=" * 78)
        for column, ylabel in COMBINED_METRICS:
            if column not in self.baseline.columns or column not in self.proposed.columns:
                continue
            stats = self.metric_statistics(column)
            print(
                f"{ylabel:22s} | baseline={stats['baseline_mean']:.8f} | "
                f"proposed={stats['proposed_mean']:.8f} | "
                f"mean delta={stats['mean_delta']:.8f} | "
                f"change={stats['percentage_change']:.4f}%"
            )
        print("=" * 78)


def main() -> None:
    configure_matplotlib()

    baseline = load_json_log(BASELINE_FILE)
    proposed = load_json_log(PROPOSED_FILE)

    validate_log_duration(baseline, BASELINE_LABEL)
    validate_log_duration(proposed, PROPOSED_LABEL)

    print(
        f"Loaded {len(baseline)} baseline samples and "
        f"{len(proposed)} proposed-controller samples."
    )
    print(
        f"Observed durations: {baseline['elapsed_s'].iloc[-1]:.2f} s "
        f"({BASELINE_LABEL}), {proposed['elapsed_s'].iloc[-1]:.2f} s "
        f"({PROPOSED_LABEL})."
    )

    comparator = JournalComparator(
        baseline=baseline,
        proposed=proposed,
        workload_name=WORKLOAD_NAME,
    )

    comparator.print_consistency_report()
    comparator.create_combined_absolute_delta_plot()
    comparator.create_box_plot()
    comparator.create_performance_level_plot()
    comparator.create_cache_plot()
    comparator.export_summary_csv()
    comparator.export_analysis_audit()

    print(
        f"Completed journal-ready exports for {WORKLOAD_NAME} "
        f"({comparator.workload_type})."
    )


if __name__ == "__main__":
    main()
