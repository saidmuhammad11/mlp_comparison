#!/usr/bin/env python3
"""
Generate Pearson correlation and multicollinearity plots for the HistGBDT-RPM dataset.

Expected location:
    /home/said/Desktop/mlp_comparison/plot_pearson_multicollinearity.py

Expected dataset directory:
    /home/said/Desktop/mlp_comparison/dataset_massive_w4_v2

Run:
    cd /home/said/Desktop/mlp_comparison
    python plot_pearson_multicollinearity.py

Optional:
    python plot_pearson_multicollinearity.py --dataset dataset_massive_w4_v2 --out plots_pearson_multicollinearity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_DISPLAY_NAMES = {
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
    parser = argparse.ArgumentParser(
        description="Draw Pearson heatmaps, VIF plot, and redundancy summary for the 11-feature dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset_massive_w4_v2"),
        help="Path to dataset directory, relative or absolute. Default: dataset_massive_w4_v2",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("plots_pearson_multicollinearity"),
        help="Output directory for JPEG and CSV files. Default: plots_pearson_multicollinearity",
    )
    parser.add_argument(
        "--earlier-max-corr",
        type=float,
        default=1.00,
        help="Earlier 12-feature max off-diagonal Pearson correlation. Default: 1.00",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.80,
        help="High-correlation reference line. Default: 0.80",
    )
    parser.add_argument(
        "--vif-threshold",
        type=float,
        default=5.0,
        help="VIF concern reference line. Default: 5.0",
    )
    return parser.parse_args()


def load_feature_names(dataset_dir: Path) -> list[str]:
    feature_path = dataset_dir / "feature_names.json"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature_names.json at: {feature_path}")

    with feature_path.open("r", encoding="utf-8") as f:
        names = json.load(f)

    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        raise ValueError("feature_names.json must contain a JSON list of strings.")

    return names


def load_matrix(path: Path, expected_cols: int, label: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} at: {path}")

    X = np.load(path)

    if X.ndim != 2:
        raise ValueError(f"{label} must be a 2D array. Found shape: {X.shape}")

    if X.shape[1] != expected_cols:
        raise ValueError(
            f"{label} has {X.shape[1]} columns, but feature_names.json has {expected_cols} names."
        )

    return X.astype(float)


def pretty_names(feature_names: list[str]) -> list[str]:
    return [DEFAULT_DISPLAY_NAMES.get(name, name.replace("_", " ").title()) for name in feature_names]


def compute_corr_df(X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(X, columns=feature_names).corr(method="pearson")


def compute_vif(X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """
    Compute VIF without requiring statsmodels.

    VIF_i = 1 / (1 - R_i^2), where R_i^2 comes from regressing feature i
    on all other features.
    """
    X = np.asarray(X, dtype=float)
    X_centered = X - X.mean(axis=0)

    rows = []
    for i, name in enumerate(feature_names):
        y = X_centered[:, i]
        X_others = np.delete(X_centered, i, axis=1)

        beta, *_ = np.linalg.lstsq(X_others, y, rcond=None)
        y_hat = X_others @ beta

        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum(y**2))

        if ss_tot == 0:
            r2 = np.nan
            vif = np.inf
        else:
            r2 = 1.0 - (ss_res / ss_tot)
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf

        rows.append(
            {
                "feature": name,
                "display_name": DEFAULT_DISPLAY_NAMES.get(name, name.replace("_", " ").title()),
                "R2_vs_other_features": r2,
                "VIF": vif,
            }
        )

    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def pairwise_summary(corr_df: pd.DataFrame, min_abs: float = 0.25) -> pd.DataFrame:
    names = list(corr_df.columns)
    rows = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = float(corr_df.iloc[i, j])
            if abs(r) >= min_abs:
                rows.append(
                    {
                        "feature_1": names[i],
                        "feature_2": names[j],
                        "feature_1_display": DEFAULT_DISPLAY_NAMES.get(
                            names[i], names[i].replace("_", " ").title()
                        ),
                        "feature_2_display": DEFAULT_DISPLAY_NAMES.get(
                            names[j], names[j].replace("_", " ").title()
                        ),
                        "pearson_r": r,
                        "abs_pearson_r": abs(r),
                    }
                )

    return (
        pd.DataFrame(rows)
        .sort_values("abs_pearson_r", ascending=False)
        .reset_index(drop=True)
    )


def save_jpeg(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=300, bbox_inches="tight", format="jpg")
    plt.close(fig)


def plot_heatmap(
    corr_df: pd.DataFrame,
    display_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 10))

    im = ax.imshow(corr_df.values, vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson correlation coefficient")

    ax.set_xticks(np.arange(len(display_names)))
    ax.set_yticks(np.arange(len(display_names)))
    ax.set_xticklabels(display_names, rotation=45, ha="right")
    ax.set_yticklabels(display_names)
    ax.set_title(title)

    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            ax.text(
                j,
                i,
                f"{corr_df.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.tight_layout()
    save_jpeg(fig, output_path)


def plot_vif(vif_df: pd.DataFrame, vif_threshold: float, output_path: Path) -> None:
    plot_df = vif_df.sort_values("VIF", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["display_name"], plot_df["VIF"])
    ax.axvline(vif_threshold, linestyle="--", linewidth=1)
    ax.set_xlabel("Variance inflation factor")
    ax.set_title("VIF Diagnostics for 11-Feature Training Set")
    ax.text(
        vif_threshold + 0.05,
        max(len(plot_df) - 1, 0),
        f"Common concern threshold: {vif_threshold:g}",
        va="center",
        fontsize=9,
    )

    fig.tight_layout()
    save_jpeg(fig, output_path)


def plot_top_pairwise_correlations(
    pair_df: pd.DataFrame,
    corr_threshold: float,
    output_path: Path,
) -> None:
    if pair_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(
            0.5,
            0.5,
            "No pairwise correlations found above the selected minimum.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        fig.tight_layout()
        save_jpeg(fig, output_path)
        return

    plot_df = pair_df.copy()
    plot_df["pair"] = plot_df["feature_1_display"] + " vs " + plot_df["feature_2_display"]
    plot_df = plot_df.sort_values("abs_pearson_r", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(plot_df["pair"], plot_df["abs_pearson_r"])
    ax.axvline(corr_threshold, linestyle="--", linewidth=1)
    ax.set_xlabel("Absolute Pearson correlation")
    ax.set_title("Largest Pairwise Correlations in 11-Feature Training Set")

    y_for_label = max(len(plot_df) - 1, 0)
    ax.text(
        corr_threshold + 0.005,
        y_for_label,
        f"High-correlation check: {corr_threshold:.2f}",
        va="center",
        fontsize=9,
    )

    for idx, value in enumerate(plot_df["abs_pearson_r"]):
        ax.text(value + 0.01, idx, f"{value:.2f}", va="center", fontsize=9)

    ax.set_xlim(0, max(0.9, float(plot_df["abs_pearson_r"].max()) + 0.12))
    fig.tight_layout()
    save_jpeg(fig, output_path)


def plot_before_after_summary(
    earlier_max_corr: float,
    current_max_corr: float,
    corr_threshold: float,
    output_path: Path,
) -> None:
    labels = [
        "Earlier 12-feature set\n(raw bandwidth included)",
        "Current 11-feature set\n(raw bandwidth removed)",
    ]
    values = [earlier_max_corr, current_max_corr]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.axhline(corr_threshold, linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Maximum absolute off-diagonal Pearson correlation")
    ax.set_title("Pearson Redundancy Check Before and After Removing Raw Bandwidth")

    for i, value in enumerate(values):
        ax.text(i, value + 0.03, f"{value:.2f}", ha="center", fontsize=11)

    ax.text(1.02, corr_threshold, f"{corr_threshold:.2f} check line", va="center", fontsize=9)

    fig.tight_layout()
    save_jpeg(fig, output_path)


def max_off_diagonal_abs(corr_df: pd.DataFrame) -> float:
    values = corr_df.to_numpy(copy=True)
    np.fill_diagonal(values, np.nan)
    return float(np.nanmax(np.abs(values)))


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    dataset_dir = args.dataset
    out_dir = args.out

    if not dataset_dir.is_absolute():
        dataset_dir = script_dir / dataset_dir

    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    feature_names = load_feature_names(dataset_dir)
    labels = pretty_names(feature_names)

    X_train_original = load_matrix(
        dataset_dir / "X_train_original.npy",
        expected_cols=len(feature_names),
        label="X_train_original.npy",
    )
    X_train = load_matrix(
        dataset_dir / "X_train.npy",
        expected_cols=len(feature_names),
        label="X_train.npy",
    )

    corr_original = compute_corr_df(X_train_original, feature_names)
    corr_balanced = compute_corr_df(X_train, feature_names)
    vif_df = compute_vif(X_train_original, feature_names)
    pair_df = pairwise_summary(corr_original, min_abs=0.25)

    corr_original.to_csv(out_dir / "pearson_correlation_matrix_train_11_features.csv")
    corr_balanced.to_csv(out_dir / "pearson_correlation_matrix_balanced_train_11_features.csv")
    vif_df.to_csv(out_dir / "vif_train_11_features.csv", index=False)
    pair_df.to_csv(out_dir / "top_pairwise_correlations_train_11_features.csv", index=False)

    plot_heatmap(
        corr_original,
        labels,
        "Pearson Correlation Heatmap of Training Features, 11-Feature Set",
        out_dir / "pearson_correlation_heatmap_train_11_features.jpg",
    )

    plot_heatmap(
        corr_balanced,
        labels,
        "Pearson Correlation Heatmap of Balanced Training Features, 11-Feature Set",
        out_dir / "pearson_correlation_heatmap_balanced_train_11_features.jpg",
    )

    plot_vif(
        vif_df,
        args.vif_threshold,
        out_dir / "vif_diagnostics_train_11_features.jpg",
    )

    plot_top_pairwise_correlations(
        pair_df,
        args.corr_threshold,
        out_dir / "top_pairwise_correlations_train_11_features.jpg",
    )

    current_max_corr = max_off_diagonal_abs(corr_original)
    plot_before_after_summary(
        earlier_max_corr=args.earlier_max_corr,
        current_max_corr=current_max_corr,
        corr_threshold=args.corr_threshold,
        output_path=out_dir / "before_after_pearson_redundancy_summary.jpg",
    )

    print("\nDone. Files written to:")
    print(out_dir)
    print("\nJPEG plots:")
    for path in sorted(out_dir.glob("*.jpg")):
        print(f"  {path}")

    print("\nCSV diagnostics:")
    for path in sorted(out_dir.glob("*.csv")):
        print(f"  {path}")

    print("\nMain diagnostics from X_train_original.npy:")
    print(f"  Samples: {X_train_original.shape[0]}")
    print(f"  Features: {X_train_original.shape[1]}")
    print(f"  Max absolute off-diagonal Pearson correlation: {current_max_corr:.4f}")
    print(f"  Max VIF: {vif_df['VIF'].max():.4f}")

    if current_max_corr >= args.corr_threshold:
        print("  Pearson check: possible high pairwise correlation.")
    else:
        print("  Pearson check: no high pairwise correlation above the selected threshold.")

    if vif_df["VIF"].max() >= args.vif_threshold:
        print("  VIF check: possible multicollinearity concern.")
    else:
        print("  VIF check: no VIF concern above the selected threshold.")


if __name__ == "__main__":
    main()
