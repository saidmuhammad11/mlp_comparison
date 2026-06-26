#!/usr/bin/env python3
"""
train_export_histgbdt.py

This is the missing piece that creates the .joblib your runtime expects.

It trains a HistGradientBoostingClassifier on:
  dataset/X_train.npy, dataset/y_train.npy
and evaluates on:
  dataset/X_test.npy, dataset/y_test.npy

Then exports:
  models/histgbdt.joblib

Also writes:
  models/histgbdt_meta.json
  results/histgbdt_test_metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

FEATURE_ORDER = [
    "ipc",
    "l2_miss_rate",
    "l3_miss_rate",
    "memory_bandwidth",
    "cpu_usage_overall",
    "cpu_temperature",
    "cpu_power",
    "cpu_frequency",
    "ipc_change",
    "ipc_avg_5",
    "l3_miss_rate_avg_5",
    "bw_util",
]

def cv_search(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Small hyperparameter search (macro-F1), because HGBDT also overfits if you let it.
    Keep this small so it runs fast on Windows.
    """
    grid = []
    for max_depth in [None, 3, 5, 8]:
        for learning_rate in [0.05, 0.1, 0.2]:
            for max_leaf_nodes in [15, 31, 63]:
                grid.append(dict(max_depth=max_depth,
                                 learning_rate=learning_rate,
                                 max_leaf_nodes=max_leaf_nodes))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best = {"score": -1.0, "params": None}

    for params in grid:
        scores = []
        for tr, va in skf.split(X, y):
            clf = HistGradientBoostingClassifier(
                random_state=42,
                max_iter=300,
                l2_regularization=0.0,
                **params,
            )
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[va])
            scores.append(f1_score(y[va], pred, average="macro"))
        score = float(np.mean(scores))
        if score > best["score"]:
            best = {"score": score, "params": params}
    return best

def main():
    X_train = np.load(DATASET / "X_train.npy")
    y_train = np.load(DATASET / "y_train.npy")
    X_test  = np.load(DATASET / "X_test.npy")
    y_test  = np.load(DATASET / "y_test.npy")

    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2D, got {X_train.shape}")

    if X_train.shape[1] != len(FEATURE_ORDER):
        print(f"⚠️ Warning: X has {X_train.shape[1]} features but FEATURE_ORDER has {len(FEATURE_ORDER)}.")
        print("   If your dataset feature order differs, update FEATURE_ORDER in BOTH runtime and this script.")

    best = cv_search(X_train, y_train)
    print("Best CV (macro-F1):", best["score"])
    print("Best params:", best["params"])

    clf = HistGradientBoostingClassifier(
        random_state=42,
        max_iter=300,
        l2_regularization=0.0,
        **best["params"],
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted")),
    }
    cm = confusion_matrix(y_test, pred).tolist()

    model_path = MODELS / "histgbdt.joblib"
    dump(clf, model_path)
    print("✅ Saved model:", model_path)

    meta = {
        "model_type": "HistGradientBoostingClassifier",
        "model_path": str(model_path.relative_to(ROOT)),
        "feature_order": FEATURE_ORDER,
        "n_features": int(X_train.shape[1]),
        "classes": [int(x) for x in getattr(clf, "classes_", np.unique(y_train)).tolist()],
        "training_cv": best,
        "scaler_stats_path": "dataset/scaler_stats.npz",
    }
    with open(MODELS / "histgbdt_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(RESULTS / "histgbdt_test_metrics.json", "w") as f:
        json.dump({"metrics": metrics, "confusion_matrix": cm}, f, indent=2)

    print("✅ Saved meta   :", MODELS / "histgbdt_meta.json")
    print("✅ Saved metrics:", RESULTS / "histgbdt_test_metrics.json")

if __name__ == "__main__":
    main()
