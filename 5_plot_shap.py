#!/usr/bin/env python3
"""
Generate SHAP plots for the 11-feature HistGBDT-RPM model.

Place this file at:
    /home/said/Desktop/mlp_comparison/plot_shap_results_11_features.py

Expected default paths:
    Dataset:
        /home/said/Desktop/mlp_comparison/dataset_massive_w4_v2
    Model:
        /home/said/Desktop/mlp_comparison/models_massive_w4_v2_unleashed/histgbdt.joblib

Run:
    cd /home/said/Desktop/mlp_comparison
    python plot_shap_results_11_features.py

This script:
    1. Loads feature_names.json from the dataset folder.
    2. Loads X_test.npy.
    3. Loads the trained HistGBDT model.
    4. Computes SHAP values on a test subset.
    5. Saves SHAP summary, SHAP bar, and SHAP waterfall plots as JPEG.
    6. Opens the JPEG plots automatically using xdg-open.

Useful options:
    python plot_shap_results_11_features.py --class-index 2
    python plot_shap_results_11_features.py --sample-size 500
    python plot_shap_results_11_features.py --sample-index 0
    python plot_shap_results_11_features.py --no-open
    python plot_shap_results_11_features.py --matplotlib-show
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import shap
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

CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SHAP plots for the 11-feature HistGBDT-RPM model."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset_massive_w4_v2"),
        help="Dataset directory. Default: dataset_massive_w4_v2",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models_massive_w4_v2_unleashed/histgbdt.joblib"),
        help="Model path. Default: models_massive_w4_v2_unleashed/histgbdt.joblib",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("plots_shap_11_features"),
        help="Output directory. Default: plots_shap_11_features",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of test samples used for SHAP. Default: 500",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index within the sampled test subset for the waterfall plot. Default: 0",
    )
    parser.add_argument(
        "--class-index",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Class to explain for multiclass SHAP. 0=Low, 1=Medium, 2=High. Default: 2",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open saved JPEG plots automatically.",
    )
    parser.add_argument(
        "--matplotlib-show",
        action="store_true",
        help="Also show Matplotlib windows. Close each window to continue.",
    )
    return parser.parse_args()


def pretty_feature_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def resolve_path(path: Path, script_dir: Path) -> Path:
    return path if path.is_absolute() else script_dir / path


def configure_fonts() -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )


def load_feature_names(dataset_dir: Path) -> list[str]:
    feature_path = dataset_dir / "feature_names.json"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature_names.json: {feature_path}")

    with feature_path.open("r", encoding="utf-8") as f:
        feature_names = json.load(f)

    if not isinstance(feature_names, list) or not all(isinstance(x, str) for x in feature_names):
        raise ValueError("feature_names.json must contain a JSON list of strings.")

    return feature_names


def load_test_data(dataset_dir: Path, n_features: int) -> np.ndarray:
    data_path = dataset_dir / "X_test.npy"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing X_test.npy: {data_path}")

    X_test = np.load(data_path).astype(float)

    if X_test.ndim != 2:
        raise ValueError(f"X_test.npy must be a 2D array. Found shape: {X_test.shape}")

    if X_test.shape[1] != n_features:
        raise ValueError(
            f"X_test.npy has {X_test.shape[1]} columns, but feature_names.json has {n_features} names."
        )

    return X_test


def make_explainer(model, X_background: np.ndarray):
    try:
        return shap.TreeExplainer(model)
    except Exception as exc:
        print(f"TreeExplainer failed, falling back to shap.Explainer. Reason: {exc}")
        return shap.Explainer(model, X_background)


def select_class_explanation(shap_values, class_index: int, feature_names: list[str]) -> shap.Explanation:
    display_feature_names = [pretty_feature_name(x) for x in feature_names]

    if isinstance(shap_values, list):
        selected = shap_values[class_index]
        if isinstance(selected, shap.Explanation):
            selected.feature_names = display_feature_names
            return selected
        values = np.asarray(selected)
        return shap.Explanation(values=values, feature_names=display_feature_names)

    if isinstance(shap_values, shap.Explanation):
        values = shap_values.values
        if values.ndim == 3:
            selected = shap_values[:, :, class_index]
            selected.feature_names = display_feature_names
            return selected
        shap_values.feature_names = display_feature_names
        return shap_values

    values = np.asarray(shap_values)
    if values.ndim == 3:
        values = values[:, :, class_index]

    return shap.Explanation(values=values, feature_names=display_feature_names)


def save_show_and_close(output_path: Path, matplotlib_show: bool) -> None:
    fig = plt.gcf()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", format="jpg")
    print(f"Saved: {output_path}")

    if matplotlib_show:
        plt.show(block=True)

    plt.close(fig)


def plot_summary_beeswarm(
    explanation: shap.Explanation,
    X_sample: np.ndarray,
    display_feature_names: list[str],
    class_label: str,
    output_path: Path,
    matplotlib_show: bool,
) -> None:
    print("Generating SHAP summary beeswarm plot...")

    plt.figure(figsize=(9.5, 8.5))
    shap.summary_plot(
        explanation.values,
        X_sample,
        feature_names=display_feature_names,
        show=False,
        max_display=len(display_feature_names),
    )

    ax = plt.gca()
    ax.set_xlabel("SHAP value, impact on model output", fontsize=16)
    # ax.set_title(f"SHAP Summary Plot, Class: {class_label}", fontsize=18)

    for label in ax.get_yticklabels():
        label.set_fontsize(16)
    ax.tick_params(axis="x", labelsize=14)

    fig = plt.gcf()
    if len(fig.axes) > 1:
        cbar = fig.axes[-1]
        cbar.tick_params(labelsize=14)
        cbar.set_ylabel("Feature value", fontsize=16)

    plt.subplots_adjust(left=0.32, right=0.95, top=0.90, bottom=0.14)
    save_show_and_close(output_path, matplotlib_show)


def plot_mean_abs_bar(
    explanation: shap.Explanation,
    display_feature_names: list[str],
    class_label: str,
    output_path: Path,
    matplotlib_show: bool,
) -> None:
    print("Generating SHAP mean absolute importance plot...")

    mean_abs = np.mean(np.abs(explanation.values), axis=0)
    order = np.argsort(mean_abs)

    ordered_names = [display_feature_names[i] for i in order]
    ordered_values = mean_abs[order]

    plt.figure(figsize=(9.5, 7.0))
    ax = plt.gca()

    bars = ax.barh(ordered_names, ordered_values)

    ax.set_xlabel("Mean absolute SHAP value", fontsize=16)
    # ax.set_title(f"SHAP Feature Importance, Class: {class_label}", fontsize=18)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    max_value = float(np.max(ordered_values)) if len(ordered_values) else 1.0
    if max_value <= 0:
        max_value = 1.0

    # Give enough right-side space so the IPC label stays inside the axes.
    ax.set_xlim(0, max_value * 1.14)

    outside_offset = max_value * 0.012
    ipc_inside_offset = max_value * 0.006

    # Only the top feature, usually IPC, gets a special label placement:
    # near the end of its bar and slightly above the bar.
    top_feature_index = int(np.argmax(ordered_values))

    for i, (bar, value) in enumerate(zip(bars, ordered_values)):
        bar_center_y = bar.get_y() + bar.get_height() / 2.0

        if i == top_feature_index:
            x = value + ipc_inside_offset
            y = bar.get_y() + bar.get_height() * 0.78

            ax.text(
                x,
                y,
                f"{value:.4f}",
                ha="left",
                va="bottom",
                fontsize=11,
                clip_on=True,
            )
        else:
            ax.text(
                value + outside_offset,
                bar_center_y,
                f"{value:.4f}",
                ha="left",
                va="center",
                fontsize=11,
                clip_on=True,
            )

    plt.subplots_adjust(left=0.32, right=0.95, top=0.90, bottom=0.14)
    save_show_and_close(output_path, matplotlib_show)


def plot_waterfall(
    explanation: shap.Explanation,
    sample_index: int,
    class_label: str,
    output_path: Path,
    matplotlib_show: bool,
) -> None:
    print(f"Generating SHAP waterfall plot for sampled row #{sample_index}...")

    if sample_index < 0 or sample_index >= explanation.shape[0]:
        raise IndexError(
            f"sample-index {sample_index} is out of range for explanation with {explanation.shape[0]} rows."
        )

    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(
        explanation[sample_index],
        show=False,
        max_display=min(12, explanation.shape[1]),
    )

    ax = plt.gca()
    # ax.set_title(f"SHAP Waterfall Plot, Class: {class_label}, Sample: {sample_index}", fontsize=18)

    for label in ax.get_yticklabels():
        label.set_fontsize(15)

    for text in ax.texts:
        text.set_fontsize(14)

    ax.tick_params(axis="x", labelsize=13)

    plt.subplots_adjust(left=0.36, right=0.95, top=0.90, bottom=0.12)
    save_show_and_close(output_path, matplotlib_show)


def open_images(paths: list[Path]) -> None:
    print("\nOpening saved SHAP JPEG plots with xdg-open...")
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


def main() -> None:
    args = parse_args()
    configure_fonts()

    script_dir = Path(__file__).resolve().parent
    dataset_dir = resolve_path(args.dataset, script_dir)
    model_path = resolve_path(args.model, script_dir)
    out_dir = resolve_path(args.out, script_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    feature_names = load_feature_names(dataset_dir)
    display_feature_names = [pretty_feature_name(x) for x in feature_names]

    X_test = load_test_data(dataset_dir, n_features=len(feature_names))
    sample_size = min(args.sample_size, X_test.shape[0])
    X_sample = X_test[:sample_size]

    print("\nLoading model and data...")
    print(f"Model:   {model_path}")
    print(f"Dataset: {dataset_dir}")
    print(f"X_test shape: {X_test.shape}")
    print(f"SHAP sample shape: {X_sample.shape}")
    print(f"Feature count: {len(feature_names)}")
    print(f"Feature order: {feature_names}")

    model = joblib.load(model_path)
    if hasattr(model, "classes_"):
        print(f"Model classes_: {list(model.classes_)}")

    class_label = CLASS_NAMES.get(args.class_index, f"Class {args.class_index}")
    print(f"Explaining class index {args.class_index}: {class_label}")

    print("\nCalculating SHAP values...")
    explainer = make_explainer(model, X_sample)
    shap_values = explainer(X_sample)

    class_explanation = select_class_explanation(
        shap_values=shap_values,
        class_index=args.class_index,
        feature_names=feature_names,
    )

    print(f"Selected SHAP explanation shape: {class_explanation.shape}")

    class_explanation.data = X_sample
    class_explanation.feature_names = display_feature_names

    summary_path = out_dir / f"shap_summary_beeswarm_class_{args.class_index}_{class_label.lower()}.jpg"
    bar_path = out_dir / f"shap_mean_abs_importance_class_{args.class_index}_{class_label.lower()}.jpg"
    waterfall_path = out_dir / f"shap_waterfall_sample_{args.sample_index}_class_{args.class_index}_{class_label.lower()}.jpg"

    plot_summary_beeswarm(
        explanation=class_explanation,
        X_sample=X_sample,
        display_feature_names=display_feature_names,
        class_label=class_label,
        output_path=summary_path,
        matplotlib_show=args.matplotlib_show,
    )

    plot_mean_abs_bar(
        explanation=class_explanation,
        display_feature_names=display_feature_names,
        class_label=class_label,
        output_path=bar_path,
        matplotlib_show=args.matplotlib_show,
    )

    plot_waterfall(
        explanation=class_explanation,
        sample_index=args.sample_index,
        class_label=class_label,
        output_path=waterfall_path,
        matplotlib_show=args.matplotlib_show,
    )

    image_paths = [summary_path, bar_path, waterfall_path]

    print("\nDone. SHAP plots saved in:")
    print(out_dir)

    print("\nJPEG plots:")
    for path in image_paths:
        print(f"  {path}")

    if not args.no_open:
        open_images(image_paths)
    else:
        print("\nAutomatic image opening disabled by --no-open.")


if __name__ == "__main__":
    main()