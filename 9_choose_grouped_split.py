import argparse
import random
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--label_col", default="raw_label")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target_test_frac", type=float, default=0.25)
    ap.add_argument("--min_test_class", type=int, default=40)
    ap.add_argument("--min_train_class", type=int, default=80)
    ap.add_argument("--trials", type=int, default=80000)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    labels = ["Low", "Medium", "High"]
    df = df[df[args.label_col].astype(str).isin(labels)].copy()

    ct = pd.crosstab(df["source_file"], df[args.label_col]).reindex(columns=labels, fill_value=0)
    ct["total"] = ct.sum(axis=1)

    files = list(ct.index)
    total_n = int(ct["total"].sum())

    if len(files) < 4:
        raise SystemExit("Need at least 4 source files for grouped split.")

    rng = random.Random(42)

    best = None

    for _ in range(args.trials):
        shuffled = files[:]
        rng.shuffle(shuffled)

        test_files = []
        test_n = 0

        target_n = args.target_test_frac * total_n

        for f in shuffled:
            # avoid making test too large
            if test_n < target_n or len(test_files) < 3:
                test_files.append(f)
                test_n += int(ct.loc[f, "total"])

        test_files = sorted(set(test_files))
        train_files = [f for f in files if f not in test_files]

        if not train_files or not test_files:
            continue

        test_counts = ct.loc[test_files, labels].sum()
        train_counts = ct.loc[train_files, labels].sum()

        if (test_counts < args.min_test_class).any():
            continue
        if (train_counts < args.min_train_class).any():
            continue

        test_frac = test_counts / test_counts.sum()
        train_frac = train_counts / train_counts.sum()

        # Prefer a balanced test set and a not-too-skewed train set
        test_balance = (test_frac - 1 / 3).abs().sum()
        train_balance = (train_frac - 1 / 3).abs().sum()
        size_penalty = abs((test_counts.sum() / total_n) - args.target_test_frac)

        score = test_balance + 0.35 * train_balance + size_penalty

        if best is None or score < best[0]:
            best = (score, train_files, test_files, train_counts, test_counts)

    if best is None:
        print("No split found.")
        print("Try lowering --min_test_class or --min_train_class.")
        raise SystemExit(1)

    score, train_files, test_files, train_counts, test_counts = best

    print("Selected split")
    print("score:", score)

    print("\nTRAIN files:")
    for f in train_files:
        print(" ", f)

    print("\nTEST files:")
    for f in test_files:
        print(" ", f)

    print("\nTRAIN counts:")
    print(train_counts)
    print("train total:", int(train_counts.sum()))

    print("\nTEST counts:")
    print(test_counts)
    print("test total:", int(test_counts.sum()))

    out = Path(args.out)
    with out.open("w") as f:
        f.write('TRAIN_FILES="' + ",".join(train_files) + '"\n')
        f.write('TEST_FILES="' + ",".join(test_files) + '"\n')

    print("\nWrote:", out)


if __name__ == "__main__":
    main()

