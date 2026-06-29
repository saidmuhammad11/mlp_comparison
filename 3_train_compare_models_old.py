import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier



CLASS_NAMES = ["Low", "Medium", "High"]


def load_feature_names(datadir: Path, n_features: int):
    feature_path = datadir / "feature_names.json"

    if feature_path.exists():
        with feature_path.open("r", encoding="utf-8") as f:
            feature_names = json.load(f)

        if len(feature_names) != n_features:
            raise SystemExit(
                f"Feature-name mismatch: X has {n_features} columns, "
                f"but feature_names.json has {len(feature_names)} names."
            )

        return feature_names

    return [f"feature_{i}" for i in range(n_features)]


def decision_stability(meta_test: pd.DataFrame, y_pred: np.ndarray):
    df = meta_test.copy()
    df["y_pred"] = y_pred

    switch_counts = 0
    trans_counts = 0
    run_lengths = []

    for _, g in df.groupby("source_file"):
        yp = g["y_pred"].to_numpy()

        if len(yp) <= 1:
            continue

        switch_counts += int(np.sum(yp[1:] != yp[:-1]))
        trans_counts += len(yp) - 1

        run = 1
        for i in range(1, len(yp)):
            if yp[i] == yp[i - 1]:
                run += 1
            else:
                run_lengths.append(run)
                run = 1

        run_lengths.append(run)

    switch_rate = switch_counts / trans_counts if trans_counts > 0 else 0.0
    avg_run = float(np.mean(run_lengths)) if run_lengths else 0.0

    return switch_rate, avg_run


def latency_ms_per_sample(predict_fn, X: np.ndarray, repeats: int = 20, max_samples: int = 500):
    if len(X) == 0:
        return 0.0

    Xs = X[: min(len(X), max_samples)]

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = predict_fn(Xs)
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / (repeats * len(Xs))


def smote_multiclass(X, y, k=5, seed=42):
    """
    Minimal multiclass SMOTE, train only.
    No imblearn dependency.
    """
    rng = np.random.RandomState(seed)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    classes, counts = np.unique(y, return_counts=True)

    if len(classes) < 2:
        return X, y

    max_count = int(counts.max())

    X_out = [X]
    y_out = [y]

    for cls, cnt in zip(classes, counts):
        cnt = int(cnt)

        if cnt >= max_count:
            continue

        Xc = X[y == cls]

        if len(Xc) < 2:
            continue

        k_eff = min(k, len(Xc) - 1)

        if k_eff < 1:
            continue

        nnm = NearestNeighbors(n_neighbors=k_eff + 1)
        nnm.fit(Xc)

        neigh = nnm.kneighbors(Xc, return_distance=False)[:, 1:]

        n_gen = max_count - cnt
        synth = np.empty((n_gen, X.shape[1]), dtype=np.float32)

        for i in range(n_gen):
            a = rng.randint(0, len(Xc))
            b = neigh[a][rng.randint(0, k_eff)]
            lam = rng.rand()
            synth[i] = Xc[a] + lam * (Xc[b] - Xc[a])

        X_out.append(synth)
        y_out.append(np.full(n_gen, cls, dtype=np.int64))

    return np.vstack(X_out), np.concatenate(y_out)


def balanced_sample_weights(y):
    """
    Create class-balanced sample weights.

    Weight formula:
        n_samples / (n_classes * class_count)

    This keeps the original telemetry samples and gives more weight to rare classes.
    """
    y = np.asarray(y, dtype=np.int64)

    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)

    class_to_weight = {
        cls: n_samples / (n_classes * count)
        for cls, count in zip(classes, counts)
    }

    return np.array([class_to_weight[v] for v in y], dtype=np.float64)


def per_class_metrics(name, y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )

    rows = []

    for i, cls_name in enumerate(CLASS_NAMES):
        rows.append(
            {
                "model": name,
                "class": cls_name,
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": int(support[i]),
            }
        )

    return rows


def eval_model(name, model, X_test, y_test, meta_test, latency_repeats=20, latency_samples=500):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    bacc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    swr, avg_run = decision_stability(meta_test, y_pred)

    lat = latency_ms_per_sample(
        model.predict,
        X_test,
        repeats=latency_repeats,
        max_samples=latency_samples,
    )

    return {
        "model": name,
        "accuracy": acc,
        "macro_f1": f1m,
        "balanced_acc": bacc,
        "switch_rate": swr,
        "avg_run_length": avg_run,
        "latency_ms_per_sample": lat,
    }, cm, y_pred


def print_class_counts(y_train, y_test):
    print("[INFO] Class distribution")

    for label, y in [("train", y_train), ("test", y_test)]:
        counts = np.bincount(y.astype(int), minlength=3)
        total = counts.sum()

        parts = []
        for i, name in enumerate(CLASS_NAMES):
            pct = 100.0 * counts[i] / total if total else 0.0
            parts.append(f"{name}={counts[i]} ({pct:.1f}%)")

        print(f"  {label}: " + ", ".join(parts))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--datadir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--modeldir", required=True)

    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--use_smote",
        action="store_true",
        help="Legacy option. Applies SMOTE to all models. Prefer --histgbdt_balance sample_weight for HistGBDT.",
    )

    ap.add_argument(
        "--histgbdt_balance",
        default="none",
        choices=["none", "sample_weight", "smote"],
        help="Balancing method used only for HistGBDT.",
    )

    ap.add_argument("--latency_samples", type=int, default=500)
    ap.add_argument("--latency_repeats", type=int, default=None)

    args = ap.parse_args()

    datadir = Path(args.datadir)
    outdir = Path(args.outdir)
    modeldir = Path(args.modeldir)

    outdir.mkdir(parents=True, exist_ok=True)
    modeldir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(datadir / "X_train.npy")
    y_train = np.load(datadir / "y_train.npy")
    X_test = np.load(datadir / "X_test.npy")
    y_test = np.load(datadir / "y_test.npy")

    meta_test = pd.read_csv(datadir / "meta_test.csv")

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise SystemExit("X_train and X_test must be 2-D arrays.")

    if X_train.shape[1] != X_test.shape[1]:
        raise SystemExit(
            f"Feature mismatch: X_train has {X_train.shape[1]} features, "
            f"but X_test has {X_test.shape[1]} features."
        )

    n_features = X_train.shape[1]
    feature_names = load_feature_names(datadir, n_features)

    print("[INFO] Loaded dataset")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test : {X_test.shape}")
    print(f"  input dimension: {n_features}")
    print(f"  feature names: {feature_names}")

    if n_features != 11:
        print(
            f"[WARN] Expected 11 features for the updated paper setup, "
            f"but got {n_features}."
        )

    print_class_counts(y_train, y_test)

    if args.use_smote and args.histgbdt_balance != "none":
        print(
            "[WARN] Both --use_smote and --histgbdt_balance were set. "
            "The HistGBDT-specific setting will override the legacy SMOTE setting for HistGBDT."
        )

    if args.use_smote:
        X_train_global, y_train_global = smote_multiclass(
            X_train,
            y_train,
            seed=args.seed,
        )
        cw_bal = None
        cw_bal_sub = None
        print("[INFO] Applied SMOTE to non-HistGBDT models.")
    else:
        X_train_global = X_train
        y_train_global = y_train
        cw_bal = "balanced"
        cw_bal_sub = "balanced_subsample"

    models = [
        (
            "LogReg",
            LogisticRegression(
                max_iter=2500,
                class_weight=cw_bal,
            ),
        ),
        (
            "LinearSVM",
            LinearSVC(
                class_weight=cw_bal,
                max_iter=5000,
                random_state=args.seed,
            ),
        ),
        (
            "kNN",
            KNeighborsClassifier(
                n_neighbors=11,
            ),
        ),
        (
            "DecisionTree",
            DecisionTreeClassifier(
                max_depth=10,
                random_state=args.seed,
                class_weight=cw_bal,
            ),
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=200 if not args.quick else 80,
                random_state=args.seed,
                n_jobs=-1,
                class_weight=cw_bal_sub,
            ),
        ),
        (
            "HistGBDT",
            HistGradientBoostingClassifier(
                learning_rate=0.1,
                max_depth=6,
                max_iter=250 if not args.quick else 120,
                random_state=args.seed,
            ),
        ),
    ]

    latency_repeats = (
        args.latency_repeats
        if args.latency_repeats is not None
        else (8 if args.quick else 20)
    )

    results = []
    cms = {}
    predictions = {}
    class_rows = {}

    for name, mdl in models:
        print(f"[INFO] Training {name}...")

        fit_kwargs = {}

        if name == "HistGBDT":
            if args.histgbdt_balance == "sample_weight":
                X_fit = X_train
                y_fit = y_train
                fit_kwargs["sample_weight"] = balanced_sample_weights(y_fit)
                print("[INFO] HistGBDT balancing: sample_weight")

            elif args.histgbdt_balance == "smote":
                X_fit, y_fit = smote_multiclass(
                    X_train,
                    y_train,
                    seed=args.seed,
                )
                print("[INFO] HistGBDT balancing: smote")

            elif args.use_smote:
                X_fit = X_train_global
                y_fit = y_train_global
                print("[INFO] HistGBDT balancing: legacy global smote")

            else:
                X_fit = X_train
                y_fit = y_train
                print("[INFO] HistGBDT balancing: none")

        else:
            X_fit = X_train_global
            y_fit = y_train_global

        mdl.fit(X_fit, y_fit, **fit_kwargs)

        r, cm, y_pred = eval_model(
            name,
            mdl,
            X_test,
            y_test,
            meta_test,
            latency_repeats=latency_repeats,
            latency_samples=args.latency_samples,
        )

        results.append(r)
        cms[name] = cm
        predictions[name] = y_pred
        class_rows[name] = per_class_metrics(name, y_test, y_pred)

        print(
            f"[OK] {name}: "
            f"acc={r['accuracy']:.3f} "
            f"macroF1={r['macro_f1']:.3f} "
            f"balancedAcc={r['balanced_acc']:.3f} "
            f"switch={r['switch_rate']:.3f} "
            f"avgRun={r['avg_run_length']:.3f} "
            f"lat(ms)={r['latency_ms_per_sample']:.4f}"
        )

        for row in class_rows[name]:
            print(
                f"      {row['class']}: "
                f"P={row['precision']:.3f} "
                f"R={row['recall']:.3f} "
                f"F1={row['f1']:.3f} "
                f"N={row['support']}"
            )

        if name == "HistGBDT":
            histgbdt_path = modeldir / "histgbdt.joblib"
            joblib.dump(mdl, histgbdt_path)
            print(f"[OK] Saved HistGBDT model to {histgbdt_path}")

    df_results = pd.DataFrame(results).sort_values(
        ["macro_f1", "balanced_acc"],
        ascending=False,
    )

    df_results.to_csv(outdir / "model_comparison.csv", index=False)

    all_class_rows = []
    for rows in class_rows.values():
        all_class_rows.extend(rows)

    pd.DataFrame(all_class_rows).to_csv(
        outdir / "per_class_metrics.csv",
        index=False,
    )

    for name, cm in cms.items():
        pd.DataFrame(
            cm,
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        ).to_csv(outdir / f"confusion_{name}.csv")

    pred_df = meta_test.copy()
    pred_df["y_true"] = y_test

    for name, y_pred_model in predictions.items():
        pred_df[f"pred_{name}"] = y_pred_model

    pred_df.to_csv(outdir / "test_predictions.csv", index=False)

    with open(outdir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    with open(modeldir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    run_config = {
        "n_features": int(n_features),
        "feature_names": feature_names,
        "quick": bool(args.quick),
        "use_smote": bool(args.use_smote),
        "histgbdt_balance": args.histgbdt_balance,
        "seed": int(args.seed),
    }

    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    with open(modeldir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print("[DONE] Wrote:")
    print(f"  {outdir / 'model_comparison.csv'}")
    print(f"  {outdir / 'per_class_metrics.csv'}")
    print(f"  {outdir / 'test_predictions.csv'}")
    print(f"  {outdir / 'confusion_*.csv'}")
    print(f"  {outdir / 'feature_names.json'}")
    print(f"  {outdir / 'run_config.json'}")

    print("[DONE] Saved models:")
    print(f"  {modeldir / 'histgbdt.joblib'}")
    print(f"  {modeldir / 'feature_names.json'}")
    print(f"  {modeldir / 'run_config.json'}")


if __name__ == "__main__":
    main()