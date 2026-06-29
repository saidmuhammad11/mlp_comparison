import json
from pathlib import Path
import pandas as pd


RAW_DIR = Path("raw_logs/sw_adaptive")
CSV_IN = Path("dataset_sw_adaptive/prepared_full.csv")
CSV_OUT = Path("dataset_sw_adaptive/prepared_full_with_labels.csv")

LABEL_COLS = ["raw_label", "majority_label", "final_label"]


def read_records(path):
    text = path.read_text(errors="ignore").strip()
    if not text:
        return []

    # JSON list or dict
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass

    # JSONL fallback
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    return records


# Build lookup: source_file + timestamp -> labels
lookup = {}

for path in sorted(RAW_DIR.glob("*.json")):
    records = read_records(path)

    for r in records:
        ts = r.get("timestamp")
        if not ts:
            continue

        key = (path.name, str(ts))

        lookup[key] = {
            "raw_label": r.get("raw_label", ""),
            "majority_label": r.get("majority_label", ""),
            "final_label": r.get("final_label", ""),
        }

print("Label lookup records:", len(lookup))

df = pd.read_csv(CSV_IN)

for col in LABEL_COLS:
    values = []
    for _, row in df.iterrows():
        key = (row["source_file"], str(row["timestamp"]))
        values.append(lookup.get(key, {}).get(col, ""))
    df[col] = values

df.to_csv(CSV_OUT, index=False)

print("Saved:", CSV_OUT)
print("Rows:", len(df))

for col in ["dvfs_level", "raw_label", "majority_label", "final_label"]:
    print("\n==", col)
    if col not in df.columns:
        print("MISSING")
        continue

    valid = df[col].notna() & df[col].astype(str).isin(["Low", "Medium", "High"])
    print("valid rows:", valid.sum())
    print(df.loc[valid, col].value_counts())
