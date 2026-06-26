#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterator, Any, List


def iter_json_records(path: Path) -> Iterator[Dict[str, Any]]:
    """
    Robust reader:
      - JSON Lines: one JSON object per line
      - JSON array
      - Single JSON object
      - Concatenated JSON objects (rare but possible)
    """
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return

    # Fast path: JSON array
    if text[0] == "[":
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                for obj in arr:
                    if isinstance(obj, dict):
                        yield obj
            return
        except json.JSONDecodeError:
            # fall through to streaming decode
            pass

    # Try single JSON object
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            yield obj
            return
        if isinstance(obj, list):
            for x in obj:
                if isinstance(x, dict):
                    yield x
            return
    except json.JSONDecodeError:
        pass

    # JSON Lines fallback (most common for logs)
    records: List[Dict[str, Any]] = []
    ok_lines = 0
    bad_lines = 0
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # allow trailing commas if someone wrote pseudo-jsonl
        if s.endswith(","):
            s = s[:-1]
        # skip array brackets if present
        if s in ("[", "]"):
            continue
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                ok_lines += 1
                yield obj
            else:
                bad_lines += 1
        except json.JSONDecodeError:
            bad_lines += 1

    # If nothing worked, try sequential raw_decode (concatenated JSON objects)
    if ok_lines == 0:
        dec = json.JSONDecoder()
        i = 0
        n = len(text)
        while i < n:
            # skip whitespace and commas
            while i < n and text[i] in " \t\r\n,":
                i += 1
            if i >= n:
                break
            try:
                obj, j = dec.raw_decode(text, i)
            except json.JSONDecodeError:
                break
            i = j
            if isinstance(obj, dict):
                yield obj


def flatten_record(rec: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    """
    Keep only the fields we care about. If a field is missing, it stays blank.
    """
    out = {
        "source_file": source_file,
        "timestamp": rec.get("timestamp", ""),
        "ipc": rec.get("ipc", None),

        "l2_cache_hits": rec.get("l2_cache_hits", None),
        "l2_cache_misses": rec.get("l2_cache_misses", None),
        "l3_cache_hits": rec.get("l3_cache_hits", None),
        "l3_cache_misses": rec.get("l3_cache_misses", None),

        "l2_miss_rate": rec.get("l2_miss_rate", None),
        "l3_miss_rate": rec.get("l3_miss_rate", None),

        "memory_bandwidth": rec.get("memory_bandwidth", None),
        "cpu_usage_overall": rec.get("cpu_usage_overall", None),
        "cpu_temperature": rec.get("cpu_temperature", None),
        "cpu_power": rec.get("cpu_power", None),
        "cpu_frequency": rec.get("cpu_frequency", None),

        "dvfs_level": rec.get("dvfs_level", ""),
        "dvfs_applied": rec.get("dvfs_applied", None),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input folder containing .json logs (raw_logs)")
    ap.add_argument("--output", required=True, help="Output CSV path (e.g., dataset/prepared.csv)")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted([p for p in in_dir.rglob("*.json") if p.is_file()])
    if not json_files:
        raise SystemExit(f"No .json files found under: {in_dir}")

    fieldnames = [
        "source_file", "timestamp", "ipc",
        "l2_cache_hits", "l2_cache_misses", "l3_cache_hits", "l3_cache_misses",
        "l2_miss_rate", "l3_miss_rate",
        "memory_bandwidth", "cpu_usage_overall", "cpu_temperature", "cpu_power", "cpu_frequency",
        "dvfs_level", "dvfs_applied"
    ]

    n_rows = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for fp in json_files:
            for rec in iter_json_records(fp):
                row = flatten_record(rec, source_file=fp.name)
                w.writerow(row)
                n_rows += 1

    print(f"[OK] Wrote {n_rows} rows to {out_path}")


if __name__ == "__main__":
    main()