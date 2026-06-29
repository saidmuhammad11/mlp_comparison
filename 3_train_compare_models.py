#!/usr/bin/env python3
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

def calculate_switch_metrics(y_pred):
    """Calculates switch rate and average run length for sequence predictions."""
    if len(y_pred) <= 1:
        return 0.0, 1.0
    switches = np.sum(y_pred[:-1] != y_pred[1:])
    switch_rate = switches / (len(y_pred) - 1)
    avg_run_length = len(y_pred) / (switches + 1) if switches > 0 else len(y_pred)
    return switch_rate, avg_run_length

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", required=True, help="Path to built dataset directory")
    parser.add_argument("--outdir", required=True, help="Directory to save CSV results")
    parser.add_argument("--modeldir", required=True, help="Directory to save trained models")
    parser.add_argument("--use_smote", action="store_true", help="Apply SMOTE to baseline models")
    parser.add_argument("--histgbdt_balance", type=str, default="none", 
                        choices=["none", "smote", "sample_weight", "class_weight"],
                        help="Balancing strategy specifically for HistGBDT")
    args = parser.parse_args()

    # Setup directories
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    datadir = Path(args.datadir)
    
    # Load data
    print("[INFO] Loading dataset...")
    X_train = np.load(datadir / "X_train.npy").astype(float)
    X_test = np.load(datadir / "X_test.npy").astype(float)
    y_train = np.load(datadir / "y_train.npy")
    y_test = np.load(datadir / "y_test.npy")
    
    # Load feature names if available
    feature_names = []
    if (datadir / "feature_names.json").exists():
        with open(datadir / "feature_names.json", "r") as f:
            feature_names = json.load(f)
    
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test : {X_test.shape}")
    if feature_names:
        print(f"  feature names: {feature_names}")

    # Map labels to numeric if they are strings
    classes = ["Low", "Medium", "High"]
    if y_train.dtype.kind in {'U', 'S', 'O'}:
        label_map = {"Low": 0, "Medium": 1, "High": 2}
        y_train = np.array([label_map[y] for y in y_train])
        y_test = np.array([label_map[y] for y in y_test])

    # Print distribution
    unique_tr, counts_tr = np.unique(y_train, return_counts=True)
    unique_te, counts_te = np.unique(y_test, return_counts=True)
    
    tr_dist = ", ".join([f"{classes[k]}={v} ({v/len(y_train)*100:.1f}%)" for k, v in zip(unique_tr, counts_tr)])
    te_dist = ", ".join([f"{classes[k]}={v} ({v/len(y_test)*100:.1f}%)" for k, v in zip(unique_te, counts_te)])
    print("[INFO] Class distribution")
    print(f"  train: {tr_dist}")
    print(f"  test: {te_dist}")

    # Baseline SMOTE
    X_fit, y_fit = X_train, y_train
    if args.use_smote:
        if not HAS_SMOTE:
            print("[WARN] imblearn not installed. Cannot use SMOTE.")
        else:
            print("[INFO] Applying SMOTE to baseline models...")
            smote = SMOTE(random_state=42)
            X_fit, y_fit = smote.fit_resample(X_train, y_train)

    # Define baseline models
    models = {
        "LogReg": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "LinearSVM": LinearSVC(max_iter=2000, random_state=42, class_weight="balanced", dual=False),
        "kNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    }

    results = []
    per_class_metrics = []
    test_predictions = pd.DataFrame()

    # Train and evaluate baselines
    for name, mdl in models.items():
        print(f"[INFO] Training {name}...")
        
        # Training
        mdl.fit(X_fit, y_fit)
        
        # Latency check (100 samples)
        sample_size = min(100, len(X_test))
        start_time = time.perf_counter()
        mdl.predict(X_test[:sample_size])
        end_time = time.perf_counter()
        lat_ms = ((end_time - start_time) / sample_size) * 1000

        # Full Test Predictions
        y_pred = mdl.predict(X_test)
        test_predictions[name] = y_pred

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        mac_f1 = f1_score(y_test, y_pred, average='macro')
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        sw_rate, avg_run = calculate_switch_metrics(y_pred)
        
        print(f"[OK] {name}: acc={acc:.3f} macroF1={mac_f1:.3f} balancedAcc={bal_acc:.3f} switch={sw_rate:.3f} avgRun={avg_run:.3f} lat(ms)={lat_ms:.4f}")

        results.append({
            "model": name,
            "accuracy": acc,
            "macro_f1": mac_f1,
            "balanced_acc": bal_acc,
            "switch_rate": sw_rate,
            "avg_run_length": avg_run,
            "latency_ms_per_sample": lat_ms
        })

        # Per Class Metrics
        prec, rec, f1_val, supp = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2])
        for i, class_name in enumerate(classes):
            print(f"      {class_name}: P={prec[i]:.3f} R={rec[i]:.3f} F1={f1_val[i]:.3f} N={supp[i]}")
            per_class_metrics.append({
                "model": name,
                "class": class_name,
                "precision": prec[i],
                "recall": rec[i],
                "f1": f1_val[i],
                "support": supp[i]
            })

    # ==========================================
    # HISTGBDT - THE UNLEASHED MODEL
    # ==========================================
    print("[INFO] Training HistGBDT...")
    print(f"[INFO] HistGBDT balancing: {args.histgbdt_balance} (Unleashed Hyperparameters)")

    # 1. Initialize Unleashed Parameters
    histgbdt = HistGradientBoostingClassifier(
        max_iter=300,          # More trees to map complex boundaries
        learning_rate=0.05,    # Slower, more careful learning
        min_samples_leaf=2,    # Allowed to isolate sharp, transient CPU spikes
        l2_regularization=0.0, # Removed the mathematical straitjacket
        random_state=42
    )

    # 2. Data prep based on strategy
    X_hgbdt, y_hgbdt = X_train, y_train
    sample_weights = None

    if args.histgbdt_balance == "smote" and HAS_SMOTE:
        smote = SMOTE(random_state=42)
        X_hgbdt, y_hgbdt = smote.fit_resample(X_train, y_train)
    elif args.histgbdt_balance == "sample_weight":
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # 3. Fit Model
    if sample_weights is not None:
        histgbdt.fit(X_hgbdt, y_hgbdt, sample_weight=sample_weights)
    else:
        histgbdt.fit(X_hgbdt, y_hgbdt)

    # 4. Latency Check
    sample_size = min(100, len(X_test))
    start_time = time.perf_counter()
    histgbdt.predict(X_test[:sample_size])
    end_time = time.perf_counter()
    lat_ms = ((end_time - start_time) / sample_size) * 1000

    # 5. Full Predictions
    y_pred = histgbdt.predict(X_test)
    test_predictions["HistGBDT"] = y_pred

    # 6. Metrics
    acc = accuracy_score(y_test, y_pred)
    mac_f1 = f1_score(y_test, y_pred, average='macro')
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    sw_rate, avg_run = calculate_switch_metrics(y_pred)
    
    print(f"[OK] HistGBDT: acc={acc:.3f} macroF1={mac_f1:.3f} balancedAcc={bal_acc:.3f} switch={sw_rate:.3f} avgRun={avg_run:.3f} lat(ms)={lat_ms:.4f}")

    results.append({
        "model": "HistGBDT",
        "accuracy": acc,
        "macro_f1": mac_f1,
        "balanced_acc": bal_acc,
        "switch_rate": sw_rate,
        "avg_run_length": avg_run,
        "latency_ms_per_sample": lat_ms
    })

    prec, rec, f1_val, supp = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2])
    for i, class_name in enumerate(classes):
        print(f"      {class_name}: P={prec[i]:.3f} R={rec[i]:.3f} F1={f1_val[i]:.3f} N={supp[i]}")
        per_class_metrics.append({
            "model": "HistGBDT",
            "class": class_name,
            "precision": prec[i],
            "recall": rec[i],
            "f1": f1_val[i],
            "support": supp[i]
        })

    # Save models and configurations
    joblib.dump(histgbdt, Path(args.modeldir) / "histgbdt.joblib")
    print(f"[OK] Saved HistGBDT model to {Path(args.modeldir) / 'histgbdt.joblib'}")

    if feature_names:
        with open(Path(args.modeldir) / "feature_names.json", "w") as f:
            json.dump(feature_names, f)
        with open(Path(args.outdir) / "feature_names.json", "w") as f:
            json.dump(feature_names, f)

    run_config = vars(args)
    with open(Path(args.outdir) / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    with open(Path(args.modeldir) / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # Save CSVs
    pd.DataFrame(results).to_csv(Path(args.outdir) / "model_comparison.csv", index=False)
    pd.DataFrame(per_class_metrics).to_csv(Path(args.outdir) / "per_class_metrics.csv", index=False)
    test_predictions.to_csv(Path(args.outdir) / "test_predictions.csv", index=False)

    # Save Confusion Matrices
    for name in test_predictions.columns:
        cm = confusion_matrix(y_test, test_predictions[name], labels=[0, 1, 2])
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        cm_df.to_csv(Path(args.outdir) / f"confusion_{name}.csv")

    print("[DONE] Wrote comparison files and saved models.")

if __name__ == "__main__":
    main()