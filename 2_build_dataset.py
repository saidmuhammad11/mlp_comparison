
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

LOW_NAMES = {"low", "powersaver", "power saver", "power_saver"}
MED_NAMES = {"medium", "balanced"}
HIGH_NAMES = {"high", "highperformance", "high performance", "high_performance"}

def normalize_level(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().lower()
    if s in LOW_NAMES: return 0
    if s in MED_NAMES: return 1
    if s in HIGH_NAMES: return 2
    try:
        iv = int(float(s))
        if iv in (0,1,2): return iv
    except Exception:
        pass
    return None

def safe_div(a, b, eps=1e-9):
    return a / (b + eps)

def rolling_mean(arr, n=5):
    out = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        s = max(0, i-n+1)
        out[i] = np.mean(arr[s:i+1])
    return out

def decide_optimal_level_from_features(ipc, mr_l3, bw_util, cpu_freq, cpu_power):
    ipc=float(ipc); mr_l3=float(mr_l3); bw_util=float(bw_util)
    cpu_freq=float(cpu_freq); cpu_power=float(cpu_power)
    ee = (ipc * cpu_freq) / (cpu_power + 1e-5)

    if (mr_l3 > 0.25) or (bw_util > 0.75): return 0
    if (ipc < 0.7) and (mr_l3 > 0.15): return 0

    if (ipc > 1.7) and (mr_l3 < 0.08): return 2
    if (ee > 0.08) and (mr_l3 < 0.12) and (ipc > 1.4): return 2

    if (0.9 <= ipc <= 1.5) and (0.05 <= mr_l3 <= 0.20): return 1
    if 0.02 <= ee <= 0.06: return 1

    if ipc >= 1.2: return 2
    if ipc <= 0.9: return 0
    return 1

def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def build_features(df: pd.DataFrame, N=5):
    """
    12-D features:
    [ IPC, MR_L2, MR_L3, BW, u, tau, P, f, dIPC, IPC_avg, MR_L3_avg, BW_util ]
    """
    col_ipc = first_existing(df, ["ipc", "IPC", "instructions_per_cycle"])
    col_l2_hit  = first_existing(df, ["l2_cache_hits","L2_hit","l2_hit","L2_HIT","l2_hits"])
    col_l2_miss = first_existing(df, ["l2_cache_misses","L2_miss","l2_miss","L2_MISS","l2_misses"])
    col_l3_hit  = first_existing(df, ["l3_cache_hits","L3_hit","l3_hit","LLC_hit","llc_hit","L3_HIT"])
    col_l3_miss = first_existing(df, ["l3_cache_misses","L3_miss","l3_miss","LLC_miss","llc_miss","L3_MISS"])
    col_bw  = first_existing(df, ["memory_bandwidth","BW","bw","mem_bw","dram_bw","total_bw"])
    col_u   = first_existing(df, ["cpu_usage_overall","cpu_usage","CPU_usage","u","util","cpu_util","cpu_utilization"])
    col_tau = first_existing(df, ["cpu_temperature","temp","temperature","cpu_temp","tau"])
    col_p   = first_existing(df, ["cpu_power","power","pkg_power","package_power","P"])
    col_f   = first_existing(df, ["cpu_frequency","freq","frequency","cpu_freq","f","effective_frequency"])

    missing = [name for name, col in [
        ("IPC", col_ipc), ("L2_hit", col_l2_hit), ("L2_miss", col_l2_miss),
        ("L3_hit", col_l3_hit), ("L3_miss", col_l3_miss), ("BW", col_bw),
        ("CPU_usage", col_u), ("temp", col_tau), ("power", col_p), ("freq", col_f)
    ] if col is None]
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {missing}")

    if "timestamp" in df.columns:
        df = df.sort_values(["source_file","timestamp"], kind="mergesort").reset_index(drop=True)
    else:
        df = df.sort_values(["source_file"], kind="mergesort").reset_index(drop=True)

    l2_hit = df[col_l2_hit].astype(float).to_numpy()
    l2_miss = df[col_l2_miss].astype(float).to_numpy()
    l3_hit = df[col_l3_hit].astype(float).to_numpy()
    l3_miss = df[col_l3_miss].astype(float).to_numpy()

    mr_l2 = safe_div(l2_miss, l2_hit + l2_miss)
    mr_l3 = safe_div(l3_miss, l3_hit + l3_miss)

    ipc = df[col_ipc].astype(float).to_numpy()
    bw = df[col_bw].astype(float).to_numpy()
    u = df[col_u].astype(float).to_numpy()
    tau = df[col_tau].astype(float).to_numpy()
    p = df[col_p].astype(float).to_numpy()
    f = df[col_f].astype(float).to_numpy()

    dipc = np.zeros_like(ipc, dtype=float)
    ipc_avg = np.zeros_like(ipc, dtype=float)
    mr3_avg = np.zeros_like(ipc, dtype=float)
    bw_util = np.zeros_like(ipc, dtype=float)

    for _, idxs in df.groupby("source_file").groups.items():
        idxs = np.array(list(idxs))
        ipc_f = ipc[idxs]
        mr3_f = mr_l3[idxs]
        bw_f = bw[idxs]

        dipc_f = np.zeros_like(ipc_f, dtype=float)
        dipc_f[1:] = ipc_f[1:] - ipc_f[:-1]

        ipc_avg_f = rolling_mean(ipc_f, n=N)
        mr3_avg_f = rolling_mean(mr3_f, n=N)

        bw_max = 1e-3
        bw_util_f = np.zeros_like(bw_f, dtype=float)
        for i in range(len(bw_f)):
            bw_max = max(bw_max, bw_f[i])
            bw_util_f[i] = np.clip(bw_f[i] / bw_max, 0.0, 1.5)

        dipc[idxs] = dipc_f
        ipc_avg[idxs] = ipc_avg_f
        mr3_avg[idxs] = mr3_avg_f
        bw_util[idxs] = bw_util_f

    X = np.stack([ipc, mr_l2, mr_l3, bw, u, tau, p, f, dipc, ipc_avg, mr3_avg, bw_util], axis=1)

    meta = df[["source_file"]].copy()
    if "timestamp" in df.columns:
        meta["timestamp"] = df["timestamp"]
    return X, meta, df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV from 1_prepare_data.py")
    ap.add_argument("--outdir", required=True, help="Output dataset dir")
    ap.add_argument("--train_files", default="", help="Comma-separated source_file(s) for train (or single-file split)")
    ap.add_argument("--test_files", default="", help="Comma-separated source_file(s) for test")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--label_col", default="dvfs_level")
    ap.add_argument("--label_rule", default="from_col", choices=["from_col","optimal_rule"])
    ap.add_argument("--test_ratio", type=float, default=0.30, help="If only --train_files given, split that file chronologically with this test ratio.")
    args = ap.parse_args()

    df0 = pd.read_csv(args.input)
    X, meta, df = build_features(df0, N=args.window)

    # labels
    if args.label_rule == "from_col":
        if args.label_col not in df.columns:
            raise SystemExit(f"Label column '{args.label_col}' not found in {args.input}")
        y_raw = df[args.label_col].apply(normalize_level).to_numpy()
        keep = np.array([v is not None for v in y_raw])
        X = X[keep]
        meta = meta.loc[keep].reset_index(drop=True)
        y = np.array([v for v in y_raw[keep] if v is not None], dtype=int)
    else:
        y = np.array([decide_optimal_level_from_features(r[0], r[2], r[11], r[7], r[6]) for r in X], dtype=int)

    train_files = [s.strip() for s in args.train_files.split(",") if s.strip()]
    test_files  = [s.strip() for s in args.test_files.split(",") if s.strip()]

    # split modes
    if train_files and not test_files:
        # filter to specified file(s), then chronological split
        subset = meta["source_file"].isin(train_files).to_numpy()
        Xs, ys = X[subset], y[subset]
        metas = meta.loc[subset].reset_index(drop=True)

        if len(Xs) < 10:
            raise SystemExit("Too few samples after filtering train_files; check source_file names.")

        if "timestamp" in metas.columns:
            t = pd.to_datetime(metas["timestamp"], errors="coerce")
            order = np.argsort(t.to_numpy())
            Xs, ys, metas = Xs[order], ys[order], metas.iloc[order].reset_index(drop=True)

        cut = int((1.0 - args.test_ratio) * len(Xs))
        X_train, y_train = Xs[:cut], ys[:cut]
        X_test,  y_test  = Xs[cut:], ys[cut:]
        meta_train = metas.iloc[:cut].reset_index(drop=True)
        meta_test  = metas.iloc[cut:].reset_index(drop=True)

    elif train_files and test_files:
        train_mask = meta["source_file"].isin(train_files).to_numpy()
        test_mask  = meta["source_file"].isin(test_files).to_numpy()
        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]
        meta_train = meta.loc[train_mask].reset_index(drop=True)
        meta_test  = meta.loc[test_mask].reset_index(drop=True)
    else:
        # fallback random split
        rng = np.random.RandomState(42)
        idx = np.arange(len(X))
        rng.shuffle(idx)
        cut = int(0.8 * len(X))
        tr, te = idx[:cut], idx[cut:]
        X_train, y_train = X[tr], y[tr]
        X_test,  y_test  = X[te], y[te]
        meta_train = meta.iloc[tr].reset_index(drop=True)
        meta_test  = meta.iloc[te].reset_index(drop=True)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir/"X_train.npy", X_train_s)
    np.save(outdir/"y_train.npy", y_train)
    np.save(outdir/"X_test.npy",  X_test_s)
    np.save(outdir/"y_test.npy",  y_test)
    meta_train.to_csv(outdir/"meta_train.csv", index=False)
    meta_test.to_csv(outdir/"meta_test.csv", index=False)
    np.savez(outdir/"scaler_stats.npz", mean_=scaler.mean_, scale_=scaler.scale_, var_=scaler.var_)

    print("[OK] Dataset built:")
    print(f"  train: {len(X_train)} samples from {sorted(set(meta_train['source_file']))}")
    print(f"  test : {len(X_test)} samples from {sorted(set(meta_test['source_file']))}")
    print(f"  saved to {outdir}")

if __name__ == "__main__":
    main()
