import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

import torch
import torch.nn as nn
import torch.optim as optim

CLASS_NAMES = ["Low", "Medium", "High"]


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
        trans_counts += (len(yp) - 1)

        run = 1
        for i in range(1, len(yp)):
            if yp[i] == yp[i - 1]:
                run += 1
            else:
                run_lengths.append(run)
                run = 1
        run_lengths.append(run)

    switch_rate = (switch_counts / trans_counts) if trans_counts > 0 else 0.0
    avg_run = float(np.mean(run_lengths)) if run_lengths else 0.0
    return switch_rate, avg_run


def latency_ms_per_sample(predict_fn, X: np.ndarray, repeats: int = 20, max_samples: int = 500) -> float:
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
    Minimal multiclass SMOTE (train only), no imblearn dependency.
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
        neigh = nnm.kneighbors(Xc, return_distance=False)[:, 1:]  # drop self

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


def eval_model(name, model, X_test, y_test, meta_test, latency_repeats=20, latency_samples=500):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    bacc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    swr, avg_run = decision_stability(meta_test, y_pred)
    lat = latency_ms_per_sample(model.predict, X_test, repeats=latency_repeats, max_samples=latency_samples)

    return {
        "model": name,
        "accuracy": acc,
        "macro_f1": f1m,
        "balanced_acc": bacc,
        "switch_rate": swr,
        "avg_run_length": avg_run,
        "latency_ms_per_sample": lat,
    }, cm


class TorchMLP(nn.Module):
    def __init__(self, in_dim=12, h1=64, h2=64, h3=32, out_dim=3, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h3, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TorchWrapper:
    def __init__(self, model: nn.Module, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def predict(self, X: np.ndarray):
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.model(x)
            return torch.argmax(logits, dim=1).cpu().numpy()


def train_torch_mlp(X_train, y_train, X_val, y_val,
                    seed=42, epochs=1500, lr=1e-3, dropout=0.10,
                    batch_size=128, weight_decay=1e-4, quick=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cpu"
    mdl = TorchMLP(dropout=dropout).to(device)
    opt = optim.Adam(mdl.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    if quick:
        epochs = min(epochs, 50)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)

    ds = torch.utils.data.TensorDataset(Xtr, ytr)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_state = None
    best_f1 = -1.0

    for ep in range(1, epochs + 1):
        mdl.train()
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(mdl(xb), yb)
            loss.backward()
            opt.step()

        mdl.eval()
        with torch.no_grad():
            pv = torch.argmax(mdl(Xv), dim=1).cpu().numpy()
        f1m = f1_score(y_val, pv, average="macro")

        if f1m > best_f1:
            best_f1 = f1m
            best_state = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}

        if ep in (1, epochs) or (ep % 100 == 0):
            print(f"    [TorchMLP] epoch {ep}/{epochs} val_macroF1={f1m:.4f} best={best_f1:.4f}")

    if best_state is not None:
        mdl.load_state_dict(best_state)
    mdl.eval()
    return mdl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--modeldir", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--use_smote", action="store_true", help="Apply internal SMOTE to TRAIN only")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--mlp_epochs", type=int, default=1500)
    ap.add_argument("--mlp_lr", type=float, default=1e-3)
    ap.add_argument("--mlp_dropout", type=float, default=0.10)
    ap.add_argument("--mlp_batch", type=int, default=128)
    ap.add_argument("--mlp_wd", type=float, default=1e-4)

    ap.add_argument("--latency_samples", type=int, default=500)
    ap.add_argument("--latency_repeats", type=int, default=None)
    args = ap.parse_args()

    datadir = Path(args.datadir)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    modeldir = Path(args.modeldir); modeldir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(datadir / "X_train.npy")
    y_train = np.load(datadir / "y_train.npy")
    X_test  = np.load(datadir / "X_test.npy")
    y_test  = np.load(datadir / "y_test.npy")
    meta_test = pd.read_csv(datadir / "meta_test.csv")

    rng = np.random.RandomState(args.seed)
    idx = np.arange(len(X_train))
    rng.shuffle(idx)
    cut = int(0.85 * len(idx))
    tr_idx, va_idx = idx[:cut], idx[cut:]
    Xtr, ytr = X_train[tr_idx], y_train[tr_idx]
    Xva, yva = X_train[va_idx], y_train[va_idx]

    if args.use_smote:
        X_train_fit, y_train_fit = smote_multiclass(X_train, y_train, seed=args.seed)
        Xtr_fit, ytr_fit = smote_multiclass(Xtr, ytr, seed=args.seed)
        cw_bal = None
        cw_bal_sub = None
    else:
        X_train_fit, y_train_fit = X_train, y_train
        Xtr_fit, ytr_fit = Xtr, ytr
        cw_bal = "balanced"
        cw_bal_sub = "balanced_subsample"

    models = [
        ("LogReg", LogisticRegression(max_iter=2500, class_weight=cw_bal)),
        ("LinearSVM", LinearSVC(class_weight=cw_bal)),
        ("kNN", KNeighborsClassifier(n_neighbors=11)),
        ("DecisionTree", DecisionTreeClassifier(max_depth=10, random_state=args.seed, class_weight=cw_bal)),
        ("RandomForest", RandomForestClassifier(
            n_estimators=200 if not args.quick else 80,
            random_state=args.seed,
            n_jobs=-1,
            class_weight=cw_bal_sub
        )),
        ("HistGBDT", HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_depth=6,
            max_iter=250 if not args.quick else 120,
            random_state=args.seed
        )),
    ]

    latency_repeats = args.latency_repeats if args.latency_repeats is not None else (8 if args.quick else 20)
    results = []
    cms = {}

    for name, mdl in models:
        mdl.fit(X_train_fit, y_train_fit)
        r, cm = eval_model(name, mdl, X_test, y_test, meta_test,
                           latency_repeats=latency_repeats, latency_samples=args.latency_samples)
        results.append(r)
        cms[name] = cm
        print(f"[OK] {name}: acc={r['accuracy']:.3f} macroF1={r['macro_f1']:.3f} switch={r['switch_rate']:.3f} lat(ms)={r['latency_ms_per_sample']:.4f}")

    print(f"[INFO] Training TorchMLP epochs={args.mlp_epochs} batch={args.mlp_batch} lr={args.mlp_lr} dropout={args.mlp_dropout} wd={args.mlp_wd}")
    torch_mlp = train_torch_mlp(
        Xtr_fit, ytr_fit, Xva, yva,
        seed=args.seed, epochs=args.mlp_epochs, lr=args.mlp_lr, dropout=args.mlp_dropout,
        batch_size=args.mlp_batch, weight_decay=args.mlp_wd, quick=args.quick
    )
    torch_path = modeldir / "torch_mlp.pth"
    torch.save(torch_mlp.state_dict(), torch_path)
    wrapper = TorchWrapper(torch_mlp, device="cpu")

    r, cm = eval_model("TorchMLP", wrapper, X_test, y_test, meta_test,
                       latency_repeats=latency_repeats, latency_samples=args.latency_samples)
    results.append(r)
    cms["TorchMLP"] = cm
    print(f"[OK] TorchMLP: acc={r['accuracy']:.3f} macroF1={r['macro_f1']:.3f} switch={r['switch_rate']:.3f} lat(ms)={r['latency_ms_per_sample']:.4f}")

    df = pd.DataFrame(results).sort_values(["macro_f1", "balanced_acc"], ascending=False)
    df.to_csv(outdir / "model_comparison.csv", index=False)

    for name, cm in cms.items():
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(outdir / f"confusion_{name}.csv")

    print(f"[DONE] Wrote:\n  {outdir / 'model_comparison.csv'}\n  confusion_*.csv\n  saved model: {torch_path}")


if __name__ == "__main__":
    main()
