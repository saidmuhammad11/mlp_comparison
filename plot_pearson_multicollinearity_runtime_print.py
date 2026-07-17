#!/usr/bin/env python3
"""
Generate and DISPLAY Pearson correlation and multicollinearity plots.

Place this file at:
    /home/said/Desktop/mlp_comparison/plot_pearson_multicollinearity_show.py

Expected dataset directory:
    /home/said/Desktop/mlp_comparison/dataset_massive_w4_v2

Run:
    cd /home/said/Desktop/mlp_comparison
    python plot_pearson_multicollinearity_show.py

By default, this script:
    1. Prints the diagnostics in the terminal.
    2. Saves JPEG plots.
    3. Opens the JPEG plots with your default image viewer using xdg-open.

Useful options:
    python plot_pearson_multicollinearity_show.py --matplotlib-show
    python plot_pearson_multicollinearity_show.py --no-open
    python plot_pearson_multicollinearity_show.py --print-full-corr
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DISPLAY_NAMES = {
    "ipc": "IPC",
    "l2_miss_rate": "L2 miss rate",
    "l3_miss_rate": "L3 miss rate",
    "cpu_usage_overall": "CPU usage",
    "cpu_temperature": "CPU temperature",
    "cpu_power": "CPU power",
    "cpu_frequency": "CPU frequency",
    "ipc_change": "IPC change",
    "ipc_avg_8": "Rolling IPC",
    "l3_miss_rate_avg_8": "Rolling L3 miss rate",
    "bw_util": "Bandwidth utilization",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset_massive_w4_v2"))
    parser.add_argument("--out", type=Path, default=Path("plots_pearson_multicollinearity"))
    parser.add_argument("--earlier-max-corr", type=float, default=1.00)
    parser.add_argument("--corr-threshold", type=float, default=0.80)
    parser.add_argument("--vif-threshold", type=float, default=5.0)
    parser.add_argument("--print-full-corr", action="store_true")
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open saved JPEG plots automatically.",
    )
    parser.add_argument(
        "--matplotlib-show",
        action="store_true",
        help="Also show Matplotlib plot windows. Close each window to continue.",
    )
    return parser.parse_args()


def pretty(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def load_data(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with (dataset_dir / "feature_names.json").open("r", encoding="utf-8") as f:
        feature_names = json.load(f)

    X_train_original = np.load(dataset_dir / "X_train_original.npy").astype(float)
    X_train = np.load(dataset_dir / "X_train.npy").astype(float)

    if X_train_original.ndim != 2 or X_train.ndim != 2:
        raise ValueError("X_train_original.npy and X_train.npy must be 2D arrays.")

    if X_train_original.shape[1] != len(feature_names):
        raise ValueError(
            f"X_train_original.npy has {X_train_original.shape[1]} columns, "
            f"but feature_names.json has {len(feature_names)} names."
        )

    if X_train.shape[1] != len(feature_names):
        raise ValueError(
            f"X_train.npy has {X_train.shape[1]} columns, "
            f"but feature_names.json has {len(feature_names)} names."
        )

    return X_train_original, X_train, feature_names


def corr_df(X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(X, columns=feature_names).corr(method="pearson")


def max_abs_off_diagonal(corr: pd.DataFrame) -> float:
    values = corr.to_numpy(copy=True)
    np.fill_diagonal(values, np.nan)
    return float(np.nanmax(np.abs(values)))


def pairwise_corr_table(corr: pd.DataFrame, min_abs: float = 0.25) -> pd.DataFrame:
    names = list(corr.columns)
    rows = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = float(corr.iloc[i, j])
            if abs(r) >= min_abs:
                rows.append(
                    {
                        "feature_1": names[i],
                        "feature_2": names[j],
                        "feature_1_display": pretty(names[i]),
                        "feature_2_display": pretty(names[j]),
                        "pearson_r": r,
                        "abs_pearson_r": abs(r),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "feature_1_display",
                "feature_2_display",
                "pearson_r",
                "abs_pearson_r",
            ]
        )

    return pd.DataFrame(rows).sort_values("abs_pearson_r", ascending=False).reset_index(drop=True)


def vif_table(X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)

    rows = []
    for i, feature in enumerate(feature_names):
        y = Xc[:, i]
        X_other = np.delete(Xc, i, axis=1)

        beta, *_ = np.linalg.lstsq(X_other, y, rcond=None)
        y_hat = X_other @ beta

        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum(y ** 2))

        if ss_tot == 0:
            r2 = np.nan
            vif = np.inf
        else:
            r2 = 1.0 - ss_res / ss_tot
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf

        rows.append(
            {
                "feature": feature,
                "display_name": pretty(feature),
                "R2_vs_other_features": r2,
                "VIF": vif,
            }
        )

    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def save_or_show(fig: plt.Figure, path: Path, matplotlib_show: bool) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", format="jpg")
    print(f"Saved: {path}")

    if matplotlib_show:
        plt.show(block=True)

    plt.close(fig)


def plot_heatmap(corr: pd.DataFrame, title: str, path: Path, matplotlib_show: bool) -> None:
    labels = [pretty(x) for x in corr.columns]

    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson correlation coefficient")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=12)

    fig.tight_layout()
    save_or_show(fig, path, matplotlib_show)


def plot_vif(vif: pd.DataFrame, threshold: float, path: Path, matplotlib_show: bool) -> None:
    plot_df = vif.sort_values("VIF", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["display_name"], plot_df["VIF"])
    ax.axvline(threshold, linestyle="--", linewidth=1)

    # Keep the dashed threshold line and its label inside the axes.
    # The label is rotated vertically beside the threshold line, instead of
    # being written horizontally outside the right side of the plot.
    max_vif = float(plot_df["VIF"].replace([np.inf, -np.inf], np.nan).max())
    if np.isnan(max_vif):
        max_vif = threshold
    x_limit = max(max_vif, threshold) * 1.08
    ax.set_xlim(0, x_limit)

    # Font size control
    xlabel_fontsize = 16
    ylabel_fontsize = 16
    xtick_fontsize = 14
    ytick_fontsize = 14
    title_fontsize = 17
    threshold_label_fontsize = 14

    ax.set_xlabel("Variance inflation factor", fontsize=xlabel_fontsize)
    ax.set_ylabel("Feature", fontsize=ylabel_fontsize)

    ax.tick_params(axis="x", labelsize=xtick_fontsize)
    ax.tick_params(axis="y", labelsize=ytick_fontsize)

    # ax.set_title(
    #     "VIF Diagnostics for 11-Feature Training Set",
    #     fontsize=title_fontsize,
    # )

    ax.annotate(
        f"Common concern threshold: {threshold:g}",
        xy=(threshold, 0.75),
        xycoords=("data", "axes fraction"),
        xytext=(-6, 0),
        textcoords="offset points",
        rotation=90,
        ha="right",
        va="top",
        fontsize=threshold_label_fontsize,
    )

    fig.tight_layout()
    save_or_show(fig, path, matplotlib_show)


def plot_top_pairs(pairs: pd.DataFrame, threshold: float, path: Path, matplotlib_show: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    if pairs.empty:
        ax.text(0.5, 0.5, "No pairwise correlations above |r| >= 0.25", ha="center", va="center")
        ax.axis("off")
    else:
        plot_df = pairs.copy()
        plot_df["pair"] = plot_df["feature_1_display"] + " vs " + plot_df["feature_2_display"]
        plot_df = plot_df.sort_values("abs_pearson_r", ascending=True)

        ax.barh(plot_df["pair"], plot_df["abs_pearson_r"])
        ax.axvline(threshold, linestyle="--", linewidth=1)
        ax.set_xlabel("Absolute Pearson correlation")
        ax.set_title("Largest Pairwise Correlations in 11-Feature Training Set")
        ax.text(
            threshold + 0.005,
            max(len(plot_df) - 1, 0),
            f"High-correlation check: {threshold:.2f}",
            va="center",
            fontsize=14,
        )

        for idx, value in enumerate(plot_df["abs_pearson_r"]):
            ax.text(value + 0.01, idx, f"{value:.2f}", va="center", fontsize=12)

        ax.set_xlim(0, max(0.9, float(plot_df["abs_pearson_r"].max()) + 0.12))

    fig.tight_layout()
    save_or_show(fig, path, matplotlib_show)


def plot_before_after(
    earlier_max_corr: float,
    current_max_corr: float,
    threshold: float,
    path: Path,
    matplotlib_show: bool,
) -> None:
    labels = [
        "Earlier 12-feature set\n(raw bandwidth included)",
        "Current 11-feature set\n(raw bandwidth removed)",
    ]
    values = [earlier_max_corr, current_max_corr]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.axhline(threshold, linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Maximum absolute off-diagonal Pearson correlation")
    ax.set_title("Pearson Redundancy Check Before and After Removing Raw Bandwidth")

    for i, value in enumerate(values):
        ax.text(i, value + 0.03, f"{value:.2f}", ha="center", fontsize=12)

    ax.text(1.02, threshold, f"{threshold:.2f} check line", va="center", fontsize=12)

    fig.tight_layout()
    save_or_show(fig, path, matplotlib_show)


def open_images(paths: list[Path]) -> None:
    print("\nOpening saved JPEG plots with xdg-open...")
    for path in paths:
        try:
            subprocess.Popen(
                ["xdg-open", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"Opened: {path}")
            time.sleep(0.25)
        except FileNotFoundError:
            print("Could not find xdg-open. Open the JPEG files manually from the output folder.")
            return
        except Exception as exc:
            print(f"Could not open {path}: {exc}")


def print_diagnostics(
    pairs: pd.DataFrame,
    vif: pd.DataFrame,
    corr_original: pd.DataFrame,
    corr_balanced: pd.DataFrame,
    corr_threshold: float,
    vif_threshold: float,
    print_full_corr: bool,
) -> None:
    current_max = max_abs_off_diagonal(corr_original)
    balanced_max = max_abs_off_diagonal(corr_balanced)

    print("\n" + "=" * 80)
    print("Runtime Pearson and multicollinearity diagnostics")
    print("=" * 80)

    print("\nTop pairwise Pearson correlations from X_train_original.npy:")
    if pairs.empty:
        print("  No pairwise correlations above |r| >= 0.25.")
    else:
        print(
            pairs[
                ["feature_1_display", "feature_2_display", "pearson_r", "abs_pearson_r"]
            ].to_string(
                index=False,
                formatters={
                    "pearson_r": "{:.4f}".format,
                    "abs_pearson_r": "{:.4f}".format,
                },
            )
        )

    print("\nVIF diagnostics from X_train_original.npy:")
    print(
        vif[["display_name", "feature", "R2_vs_other_features", "VIF"]].to_string(
            index=False,
            formatters={
                "R2_vs_other_features": "{:.4f}".format,
                "VIF": "{:.4f}".format,
            },
        )
    )

    print("\nSummary:")
    print(f"  Max |Pearson r|, X_train_original.npy: {current_max:.4f}")
    print(f"  Max |Pearson r|, X_train.npy:          {balanced_max:.4f}")
    print(f"  Max VIF, X_train_original.npy:         {vif['VIF'].max():.4f}")

    if current_max >= corr_threshold:
        print(f"  Pearson check: possible high pairwise correlation, threshold {corr_threshold:.2f}.")
    else:
        print(f"  Pearson check: no high pairwise correlation above {corr_threshold:.2f}.")

    if vif["VIF"].max() >= vif_threshold:
        print(f"  VIF check: possible multicollinearity concern, threshold {vif_threshold:.2f}.")
    else:
        print(f"  VIF check: no VIF concern above {vif_threshold:.2f}.")

    if print_full_corr:
        print("\nFull Pearson correlation matrix, X_train_original.npy:")
        print(corr_original.round(4).to_string())

        print("\nFull Pearson correlation matrix, X_train.npy:")
        print(corr_balanced.round(4).to_string())

    print("=" * 80 + "\n")


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent

    dataset_dir = args.dataset
    if not dataset_dir.is_absolute():
        dataset_dir = script_dir / dataset_dir

    out_dir = args.out
    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    X_train_original, X_train, feature_names = load_data(dataset_dir)

    corr_original = corr_df(X_train_original, feature_names)
    corr_balanced = corr_df(X_train, feature_names)
    pairs = pairwise_corr_table(corr_original)
    vif = vif_table(X_train_original, feature_names)

    print_diagnostics(
        pairs=pairs,
        vif=vif,
        corr_original=corr_original,
        corr_balanced=corr_balanced,
        corr_threshold=args.corr_threshold,
        vif_threshold=args.vif_threshold,
        print_full_corr=args.print_full_corr,
    )

    corr_original.to_csv(out_dir / "pearson_correlation_matrix_train_11_features.csv")
    corr_balanced.to_csv(out_dir / "pearson_correlation_matrix_balanced_train_11_features.csv")
    pairs.to_csv(out_dir / "top_pairwise_correlations_train_11_features.csv", index=False)
    vif.to_csv(out_dir / "vif_train_11_features.csv", index=False)

    image_paths = [
        out_dir / "pearson_correlation_heatmap_train_11_features.jpg",
        out_dir / "pearson_correlation_heatmap_balanced_train_11_features.jpg",
        out_dir / "vif_diagnostics_train_11_features.jpg",
        out_dir / "top_pairwise_correlations_train_11_features.jpg",
        out_dir / "before_after_pearson_redundancy_summary.jpg",
    ]

    plot_heatmap(
        corr_original,
        "Pearson Correlation Heatmap of Training Features, 11-Feature Set",
        image_paths[0],
        args.matplotlib_show,
    )

    plot_heatmap(
        corr_balanced,
        "Pearson Correlation Heatmap of Balanced Training Features, 11-Feature Set",
        image_paths[1],
        args.matplotlib_show,
    )

    plot_vif(vif, args.vif_threshold, image_paths[2], args.matplotlib_show)
    plot_top_pairs(pairs, args.corr_threshold, image_paths[3], args.matplotlib_show)

    plot_before_after(
        earlier_max_corr=args.earlier_max_corr,
        current_max_corr=max_abs_off_diagonal(corr_original),
        threshold=args.corr_threshold,
        path=image_paths[4],
        matplotlib_show=args.matplotlib_show,
    )

    print("\nDone. Output folder:")
    print(out_dir)

    print("\nJPEG plots:")
    for path in image_paths:
        print(f"  {path}")

    print("\nCSV diagnostics:")
    for path in sorted(out_dir.glob("*.csv")):
        print(f"  {path}")

    if not args.no_open:
        open_images(image_paths)
    else:
        print("\nAutomatic image opening disabled by --no-open.")


if __name__ == "__main__":
    main()