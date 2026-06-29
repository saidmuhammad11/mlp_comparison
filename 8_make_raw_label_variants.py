from pathlib import Path
import pandas as pd

IN = Path("dataset_sw_adaptive/label_csvs/prepared_raw_label.csv")
OUTDIR = Path("dataset_sw_adaptive/raw_label_variants")
OUTDIR.mkdir(parents=True, exist_ok=True)

LABEL = "raw_label"
VALID = ["Low", "Medium", "High"]

df = pd.read_csv(IN)
df = df[df[LABEL].astype(str).isin(VALID)].copy()

labels = ["Low", "Medium", "High"]
ct = pd.crosstab(df["source_file"], df[LABEL]).reindex(columns=labels, fill_value=0)
ct["total"] = ct.sum(axis=1)
ct["medium_ratio"] = ct["Medium"] / ct["total"]
ct["high_ratio"] = ct["High"] / ct["total"]

# Medium-heavy files: dominated by Medium and weak High coverage
medium_heavy = ct[
    (ct["medium_ratio"] >= 0.65) &
    (ct["high_ratio"] <= 0.15) &
    (ct["total"] >= 50)
].index.tolist()

print("Raw-label per-file distribution:")
print(ct.sort_values(["medium_ratio", "total"], ascending=[False, False]).to_string())

print("\nMedium-heavy files selected for removal/capping:")
for f in medium_heavy:
    print(" ", f)

ct.to_csv(OUTDIR / "raw_label_source_stats.csv")

# Variant 1: all
df.to_csv(OUTDIR / "prepared_raw_label_all.csv", index=False)

# Variant 2: remove Medium-heavy files
df_no = df[~df["source_file"].isin(medium_heavy)].copy()
df_no.to_csv(OUTDIR / "prepared_raw_label_no_medium_heavy.csv", index=False)

# Variant 3: cap Medium-heavy files
parts = []

for (src, label), g in df.groupby(["source_file", LABEL]):
    if src in medium_heavy:
        if label == "Medium":
            g = g.sample(n=min(len(g), 180), random_state=42)
        elif label == "Low":
            g = g.sample(n=min(len(g), 180), random_state=42)
        else:
            # keep all High
            g = g
    parts.append(g)

df_cap = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
df_cap.to_csv(OUTDIR / "prepared_raw_label_medium_heavy_capped.csv", index=False)

print("\nSaved variants:")
for name, d in [
    ("all", df),
    ("no_medium_heavy", df_no),
    ("medium_heavy_capped", df_cap),
]:
    print("\n==", name)
    print("rows:", len(d))
    print(d[LABEL].value_counts())
