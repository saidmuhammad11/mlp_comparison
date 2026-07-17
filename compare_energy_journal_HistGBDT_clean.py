#!/usr/bin/env python3
"""
Journal-ready comparison plots for the PMC-assisted power-management study.

Key figure changes:
1. Exports true vector PDF and SVG directly from Matplotlib.
2. Uses compact full-width dimensions suitable for a two-column article.
3. Uses one shared legend and shared time axes.
4. Removes repeated subplot titles and repeated upper x-axis labels.
5. Adds panel labels (a), (b), ...
6. Omits overall figure titles because titles are supplied by LaTeX captions.
7. Uses a white background and tight margins.
8. Writes comparison values to CSV instead of rendering the table as an image.
9. Combines absolute runtime traces with baseline-referenced delta plots.

Edit only the CONFIGURATION section for each experiment.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================ CONFIGURATION ============================

BASELINE_FILE = "histgbdtdisable_mixed1.json"
PROPOSED_FILE = "histgbdtenable_mixed1.json"

# WORKLOAD_NAME = "CPU_Intensive"
# WORKLOAD_NAME = "MEMORY_Intensive"
WORKLOAD_NAME = "Mixed"

BASELINE_LABEL = "Default Windows Medium"
PROPOSED_LABEL = "HistGBDT-RPM"

OUTPUT_DIR = Path("journal_figures")
OUTPUT_PREFIX = "HistGBDT-RPM"

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

# Cache summary is inserted in one manuscript column.
COLUMN_WIDTH_IN = 3.43

# Metrics retained in the combined absolute-and-delta figure.
# Cache behavior is reported in the dedicated cache summary, while memory
# usage and temperature remain available in the numerical summary.
COMBINED_METRICS = [
    ("cpu_power", "CPU power (W)"),
    ("ipc", "IPC"),
    ("cpu_frequency", "CPU freq. (MHz)"),
    ("cpu_usage_overall", "CPU usage (%)"),
]

# General font sizes at the final printed size.
BASE_FONT_SIZE = 10.5
SMALL_FONT_SIZE = BASE_FONT_SIZE
LEGEND_FONT_SIZE = BASE_FONT_SIZE
PANEL_LABEL_SIZE = BASE_FONT_SIZE

# Larger type used only for compact summary figures.
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

# Existing line colors are retained for baseline-versus-HistGBDT-RPM traces.
BASELINE_COLOR = "#4169E1"
PROPOSED_COLOR = "#FF69B4"

# Distinct color and hatch encodings for controller states.
# Hatches ensure that Low, Medium, and High remain distinguishable in grayscale.
STATE_STYLES = {
    "Low": {
        "color": "#4C78A8",
        "hatch": "///",
    },
    "Medium": {
        "color": "#F2A541",
        "hatch": "xx",
    },
    "High": {
        "color": "#59A14F",
        "hatch": "...",
    },
    "Fixed": {
        "color": "#9D9D9D",
        "hatch": "\\\\",
    },
}

# Distinct color and hatch encodings for cache levels.
CACHE_STYLES = {
    "L2": {
        "color": "#59A14F",  # Green
    },
    "L3": {
        "color": "#E15759",  # Red
    },
}

# Policy is encoded by hatch pattern, consistently across L2 and L3.
POLICY_STYLES = {
    BASELINE_LABEL: {
        "hatch": "///",
    },
    PROPOSED_LABEL: {
        "hatch": "xx",
    },
}

MAX_MEMORY_BANDWIDTH = 50.0  # GB/s, used only for workload classification

# =====================================================================


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
    """Load a line-delimited JSON log and return timestamp-sorted samples."""
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Input log not found: {path.resolve()}")

    records: list[dict] = []
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
    return frame.sort_values("timestamp").reset_index(drop=True)


def elapsed_minutes(frame: pd.DataFrame) -> np.ndarray:
    """Return elapsed time relative to the first sample."""
    elapsed = frame["timestamp"] - frame["timestamp"].iloc[0]
    return elapsed.dt.total_seconds().to_numpy() / 60.0


def integrate_energy_from_timestamps(
    frame: pd.DataFrame, power_column: str = "cpu_power"
) -> float:
    """Integrate package power using the measured timestamps."""
    clean = frame[["timestamp", power_column]].dropna()
    if len(clean) < 2:
        return float("nan")

    seconds = (
        clean["timestamp"] - clean["timestamp"].iloc[0]
    ).dt.total_seconds().to_numpy()
    power = clean[power_column].to_numpy(dtype=float)

    # np.trapezoid is preferred in newer NumPy; np.trapz remains widely supported.
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(power, seconds))
    return float(np.trapz(power, seconds))


def preview_and_maybe_save(fig: plt.Figure, stem: str) -> None:
    """
    Display the figure first, then save according to SAVE_MODE.

    With SAVE_MODE="ask", close the displayed figure window and answer the
    terminal prompt. Enter y/yes to save or press Enter to skip saving.
    """
    if SHOW_FIGURES:
        plt.show(block=True)

    normalized_mode = SAVE_MODE.strip().lower()
    if normalized_mode not in {"ask", "always", "never"}:
        raise ValueError(
            'SAVE_MODE must be "ask", "always", or "never".'
        )

    should_save = normalized_mode == "always"

    if normalized_mode == "ask":
        try:
            answer = input(
                f"Save figure '{stem}' as PDF and SVG? [y/N]: "
            ).strip().lower()
            should_save = answer in {"y", "yes"}
        except EOFError:
            # Useful for non-interactive or notebook environments.
            should_save = False
            print(
                f"No interactive input was available. "
                f"Figure '{stem}' was not saved."
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
    """
    Align y-axis headings at fixed x coordinates within each subplot column.

    This prevents different tick-label widths from shifting the y-axis titles.
    """
    axes_array = np.asarray(axes, dtype=object)

    if axes_array.ndim == 1:
        axes_array = axes_array.reshape(1, -1)

    n_rows, n_cols = axes_array.shape

    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes_array[row, col]
            if not ax.get_visible():
                continue

            x_coord = left_x if col == 0 else right_x
            ax.yaxis.set_label_coords(x_coord, 0.5)

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
    def __init__(
        self,
        baseline: pd.DataFrame,
        proposed: pd.DataFrame,
        workload_name: str,
    ) -> None:
        self.baseline = baseline
        self.proposed = proposed
        self.workload_name = workload_name
        self.baseline_time = elapsed_minutes(baseline)
        self.proposed_time = elapsed_minutes(proposed)
        self.metrics = available_metric_pairs(baseline, proposed)
        self.workload_type = self.detect_workload_type()

    def detect_workload_type(self) -> str:
        """Classify the logged workload using the existing heuristic."""
        avg_l3_miss = (
            self.proposed["l3_miss_rate"].mean()
            if "l3_miss_rate" in self.proposed.columns
            else 0.0
        )
        avg_mem_bw = (
            self.proposed["memory_bandwidth"].mean()
            if "memory_bandwidth" in self.proposed.columns
            else 0.0
        )
        if avg_l3_miss > 0.05 and avg_mem_bw > 0.6 * MAX_MEMORY_BANDWIDTH:
            return "memory-intensive"
        return "CPU-intensive"

    def create_combined_absolute_delta_plot(self) -> None:
        """
        Create one journal-ready figure containing absolute and delta traces.

        The left column reports the absolute runtime measurements for the
        baseline and proposed policies. The right column reports the
        baseline-referenced difference

            delta X(t) = X_baseline(t) - X_proposed(t).

        A positive delta means that the proposed policy produces a lower
        measured value than the baseline. Its practical interpretation remains
        metric dependent. For example, a positive power delta indicates lower
        power under the proposed policy, whereas a positive IPC delta indicates
        lower IPC under the proposed policy.
        """
        metrics = [
            (column, ylabel)
            for column, ylabel in COMBINED_METRICS
            if column in self.baseline.columns
            and column in self.proposed.columns
        ]

        if not metrics:
            raise ValueError(
                "None of the requested combined metrics are available "
                "in both input logs."
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

            baseline_line, = absolute_ax.plot(
                self.baseline_time,
                pd.to_numeric(
                    self.baseline[column],
                    errors="coerce",
                ).to_numpy(dtype=float),
                label=BASELINE_LABEL,
                color=BASELINE_COLOR,
            )

            proposed_line, = absolute_ax.plot(
                self.proposed_time,
                pd.to_numeric(
                    self.proposed[column],
                    errors="coerce",
                ).to_numpy(dtype=float),
                label=PROPOSED_LABEL,
                color=PROPOSED_COLOR,
            )

            absolute_ax.set_ylabel(
                ylabel,
                fontsize=BASE_FONT_SIZE,
                labelpad=2.0,
            )
            absolute_ax.tick_params(
                axis="both",
                which="major",
                labelsize=BASE_FONT_SIZE,
                pad=1.5,
            )
            absolute_ax.grid(True)

            absolute_panel_index = 2 * row
            add_panel_label(
                absolute_ax,
                f"({chr(ord('a') + absolute_panel_index)})",
            )

            if legend_handles is None:
                legend_handles = [baseline_line, proposed_line]
                legend_labels = [BASELINE_LABEL, PROPOSED_LABEL]

            baseline_values = pd.to_numeric(
                self.baseline[column],
                errors="coerce",
            ).to_numpy(dtype=float)

            proposed_values = pd.to_numeric(
                self.proposed[column],
                errors="coerce",
            ).to_numpy(dtype=float)

            common_start = max(
                float(np.nanmin(self.baseline_time)),
                float(np.nanmin(self.proposed_time)),
            )
            common_end = min(
                float(np.nanmax(self.baseline_time)),
                float(np.nanmax(self.proposed_time)),
            )

            baseline_mask = (
                (self.baseline_time >= common_start)
                & (self.baseline_time <= common_end)
                & np.isfinite(baseline_values)
            )
            proposed_mask = (
                (self.proposed_time >= common_start)
                & (self.proposed_time <= common_end)
                & np.isfinite(proposed_values)
            )

            delta_time = self.baseline_time[baseline_mask]
            aligned_baseline = baseline_values[baseline_mask]

            if len(delta_time) == 0 or np.count_nonzero(proposed_mask) < 2:
                raise ValueError(
                    f"Insufficient overlapping samples for metric '{column}'."
                )

            aligned_proposed = np.interp(
                delta_time,
                self.proposed_time[proposed_mask],
                proposed_values[proposed_mask],
            )

            delta = aligned_baseline - aligned_proposed

            delta_ax.plot(
                delta_time,
                delta,
                color=BASELINE_COLOR,
            )
            delta_ax.axhline(
                0.0,
                color="black",
                linewidth=0.7,
            )

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

            delta_panel_index = 2 * row + 1
            add_panel_label(
                delta_ax,
                f"({chr(ord('a') + delta_panel_index)})",
            )

            mean_delta = float(np.nanmean(delta))

            # Place the mean-delta label above the highest plotted value.
            # Extra y-axis headroom prevents the annotation from overlapping
            # the runtime-difference trace.
            finite_delta = delta[np.isfinite(delta)]
            if finite_delta.size:
                data_min = float(np.min(finite_delta))
                data_max = float(np.max(finite_delta))

                display_min = min(data_min, 0.0)
                display_max = max(data_max, 0.0)
                display_span = display_max - display_min

                if display_span <= 0.0:
                    display_span = max(abs(display_max), 1.0)

                lower_padding = 0.06 * display_span
                upper_padding = 0.28 * display_span
                annotation_gap = 0.11 * display_span

                delta_ax.set_ylim(
                    display_min - lower_padding,
                    display_max + upper_padding,
                )

                annotation_x = float(np.max(delta_time))
                annotation_y = display_max + annotation_gap

                delta_ax.text(
                    annotation_x,
                    annotation_y,
                    rf"Mean $\Delta X$ = {mean_delta:.2f}",
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

        axes[-1, 0].set_xlabel(
            "Time (min)",
            fontsize=BASE_FONT_SIZE,
        )
        axes[-1, 1].set_xlabel(
            "Time (min)",
            fontsize=BASE_FONT_SIZE,
        )

        align_ylabels_by_column(
            fig,
            axes,
            left_x=-0.15,
            right_x=-0.16,
        )

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
                prop={
                    "size": BASE_FONT_SIZE,
                    "weight": "medium",
                },
            )

        preview_and_maybe_save(
            fig,
            "combined_absolute_baseline_referenced_delta",
        )
        plt.close(fig)

    def create_box_plot(self) -> None:
        """Create a compact two-by-two distribution comparison."""
        requested = [
            ("cpu_power", "CPU power (W)"),
            ("ipc", "IPC"),
            ("l2_miss_rate", "L2 miss rate"),
            ("l3_miss_rate", "L3 miss rate"),
        ]
        metrics = [
            pair
            for pair in requested
            if pair[0] in self.baseline.columns
            and pair[0] in self.proposed.columns
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
            box = ax.boxplot(
                [
                    self.baseline[column].dropna(),
                    self.proposed[column].dropna(),
                ],
                tick_labels=[BASELINE_LABEL, PROPOSED_LABEL],
                patch_artist=True,
                widths=0.55,
            )
            for patch, color in zip(
                box["boxes"], [BASELINE_COLOR, PROPOSED_COLOR]
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

        align_ylabels_by_column(
            fig,
            axes,
            left_x=-0.14,
            right_x=-0.13,
        )

        preview_and_maybe_save(fig, "boxplots")
        plt.close(fig)

    def create_performance_level_plot(self) -> None:
        """Create one two-panel controller-state distribution figure."""
        if "dvfs_level" not in self.proposed.columns:
            return

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
            self.baseline,
            BASELINE_LABEL,
            BASELINE_COLOR,
            "(a)",
        )
        self._plot_level_distribution(
            axes[1],
            self.proposed,
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
        frame: pd.DataFrame,
        label: str,
        color: str,
        panel_label: str,
    ) -> None:
        counts = Counter(frame["dvfs_level"].dropna())
        preferred_order = ["Low", "Medium", "High", "Fixed"]
        categories = sorted(
            counts,
            key=lambda value: (
                preferred_order.index(value)
                if value in preferred_order
                else len(preferred_order)
            ),
        )
        values = [counts[category] for category in categories]

        bar_colors = [
            STATE_STYLES.get(category, {"color": color})["color"]
            for category in categories
        ]
        bars = ax.bar(
            categories,
            values,
            color=bar_colors,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.7,
        )

        for bar, category, value in zip(bars, categories, values):
            style = STATE_STYLES.get(category, {})
            bar.set_hatch(style.get("hatch", ""))

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                str(int(value)),
                ha="center",
                va="bottom",
                fontsize=LEVEL_COUNT_SIZE,
            )

        ax.set_ylabel(
            "Sample count",
            fontsize=LEVEL_LABEL_SIZE,
            labelpad=4.0,
        )
        ax.set_title(
            label,
            fontsize=LEVEL_TITLE_SIZE,
            pad=6,
        )
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
        """
        Create a readable two-column cache summary with parallel panels.

        Panel (a): L2 and L3 mean hit rates.
        Panel (b): L2 and L3 mean miss rates.

        Color identifies cache level:
        - L2: green
        - L3: red

        Hatch identifies policy:
        - Default Windows Medium: diagonal hatch
        - HistGBDT-RPM: cross hatch
        """
        from matplotlib.patches import Patch

        required = {"l2_miss_rate", "l3_miss_rate"}
        if not required.issubset(self.baseline.columns) or not required.issubset(
            self.proposed.columns
        ):
            return

        cache_levels = ["L2", "L3"]

        baseline_miss = [
            float(self.baseline["l2_miss_rate"].mean()),
            float(self.baseline["l3_miss_rate"].mean()),
        ]
        proposed_miss = [
            float(self.proposed["l2_miss_rate"].mean()),
            float(self.proposed["l3_miss_rate"].mean()),
        ]

        baseline_hit = [1.0 - value for value in baseline_miss]
        proposed_hit = [1.0 - value for value in proposed_miss]

        # Parallel panels need the full two-column width to preserve readability.
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
            ax.set_ylabel(
                ylabel,
                fontsize=SUMMARY_LABEL_SIZE,
                labelpad=4.0,
            )
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

        # Keep the two y-axis titles aligned.
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
        """Write numerical results for a native LaTeX table."""
        baseline_energy = integrate_energy_from_timestamps(self.baseline)
        proposed_energy = integrate_energy_from_timestamps(self.proposed)

        metric_specs = [
            ("Average CPU power (W)", "cpu_power"),
            ("Average IPC", "ipc"),
            ("Average CPU usage (%)", "cpu_usage_overall"),
            ("Average L2 miss rate", "l2_miss_rate"),
            ("Average L3 miss rate", "l3_miss_rate"),
            ("Average temperature (°C)", "cpu_temperature"),
            ("Average frequency (GHz)", "cpu_frequency"),
        ]

        rows: list[dict] = []
        for display_name, column in metric_specs:
            if (
                column not in self.baseline.columns
                or column not in self.proposed.columns
            ):
                continue

            baseline_value = float(self.baseline[column].mean())
            proposed_value = float(self.proposed[column].mean())

            if column == "cpu_frequency":
                baseline_value /= 1000.0
                proposed_value /= 1000.0

            signed_change = (
                100.0 * (proposed_value - baseline_value) / baseline_value
                if baseline_value != 0
                else float("nan")
            )
            rows.append(
                {
                    "Metric": display_name,
                    BASELINE_LABEL: baseline_value,
                    PROPOSED_LABEL: proposed_value,
                    "Change vs. baseline (%)": signed_change,
                }
            )

        if np.isfinite(baseline_energy) and np.isfinite(proposed_energy):
            energy_change = (
                100.0
                * (proposed_energy - baseline_energy)
                / baseline_energy
                if baseline_energy != 0
                else float("nan")
            )
            rows.append(
                {
                    "Metric": "Integrated energy (J)",
                    BASELINE_LABEL: baseline_energy,
                    PROPOSED_LABEL: proposed_energy,
                    "Change vs. baseline (%)": energy_change,
                }
            )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.csv"
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Saved: {output_path}")


def main() -> None:
    configure_matplotlib()

    baseline = load_json_log(BASELINE_FILE)
    proposed = load_json_log(PROPOSED_FILE)

    print(
        f"Loaded {len(baseline)} baseline samples and "
        f"{len(proposed)} proposed-controller samples."
    )

    comparator = JournalComparator(
        baseline=baseline,
        proposed=proposed,
        workload_name=WORKLOAD_NAME,
    )

    comparator.create_combined_absolute_delta_plot()
    comparator.create_box_plot()
    comparator.create_performance_level_plot()
    comparator.create_cache_plot()
    comparator.export_summary_csv()

    print(
        f"Completed journal-ready exports for {WORKLOAD_NAME} "
        f"({comparator.workload_type})."
    )


if __name__ == "__main__":
    main()