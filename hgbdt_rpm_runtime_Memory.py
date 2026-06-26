#!/usr/bin/env python3
"""
Serverless-style Runtime Power Management (RPM) via Memory-Tier Selection

Objective:
- Choose memory tier m ∈ {128,256,512,1024,2048} MB per invocation
- Use ONLY serverless-visible signals at runtime:
  duration_ms, cpu_time_ms, rss_mb, peak_rss_mb, io bytes, cold_start, etc.
- Predict latency and energy for each candidate tier using deployable models:
  f_T(x,m) -> latency_ms
  f_E(x,m) -> energy_joules
- Decision rule:
  - pick lowest predicted energy among tiers satisfying SLO
  - else pick tier with lowest predicted latency
- Log one JSON record per invocation (never only averages)

Notes:
- This script assumes you run the workload inside Docker with memory limits.
- It does not use PMCs at runtime.
- If you have a Windows lab mode (Intel PCM) for energy labeling, do that in
  the data-collection harness, not here.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib  # models saved via joblib


# ----------------------------
# Configuration / Defaults
# ----------------------------

DEFAULT_MEMORY_TIERS_MB = [128, 256, 512, 1024, 2048]

# Serverless-visible feature order expected by both models
FEATURE_ORDER = [
    "cpu_time_ms",
    "rss_mb",
    "peak_rss_mb",
    "io_read_bytes",
    "io_write_bytes",
    "cold_start",
    "concurrency",
    "queue_delay_ms",
    "mem_limit_mb",  # candidate tier injected per evaluation
]

STOP_REQUESTED = False


def _handle_sigint(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)


# ----------------------------
# Utilities: measurement
# ----------------------------

def now_ms() -> int:
    return int(time.time() * 1000)


def read_proc_io_bytes_linux(pid: int) -> Tuple[int, int]:
    """
    Linux-only best-effort: /proc/<pid>/io provides read_bytes and write_bytes.
    In containers, this refers to the process view. If unavailable, returns (0,0).
    """
    path = Path(f"/proc/{pid}/io")
    if not path.exists():
        return (0, 0)

    read_b = 0
    write_b = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    read_b = int(line.split(":")[1].strip())
                elif line.startswith("write_bytes:"):
                    write_b = int(line.split(":")[1].strip())
    except Exception:
        return (0, 0)
    return (read_b, write_b)


def read_proc_rss_peak_linux(pid: int) -> Tuple[float, float]:
    """
    Best-effort RSS and peak RSS (MB).
    - RSS from /proc/<pid>/status VmRSS
    - Peak from /proc/<pid>/status VmHWM (high water mark)
    If not available, returns (0,0).
    """
    path = Path(f"/proc/{pid}/status")
    if not path.exists():
        return (0.0, 0.0)

    rss_kb = 0
    peak_kb = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                elif line.startswith("VmHWM:"):
                    peak_kb = int(line.split()[1])
    except Exception:
        return (0.0, 0.0)

    return (rss_kb / 1024.0, peak_kb / 1024.0)


def cpu_time_ms_from_resource(child_rusage) -> int:
    """
    Convert child process rusage utime+stime to ms.
    Works on Linux/macOS. On Windows, this path may differ.
    """
    ut = getattr(child_rusage, "ru_utime", 0.0)
    st = getattr(child_rusage, "ru_stime", 0.0)
    return int((ut + st) * 1000.0)


# ----------------------------
# Data record
# ----------------------------

@dataclass
class InvocationRecord:
    platform: str
    workload: str
    run_id: str
    invocation_id: str
    timestamp_start_ms: int
    timestamp_end_ms: int

    mem_limit_mb: int
    cold_start: int
    concurrency: int
    queue_delay_ms: int

    duration_ms: int
    cpu_time_ms: int
    rss_mb: float
    peak_rss_mb: float
    io_read_bytes: int
    io_write_bytes: int

    # Labels (optional; often empty at deployment)
    energy_joules: Optional[float] = None
    edp: Optional[float] = None

    # Predictions (for debugging / analysis)
    pred_energy_joules: Optional[float] = None
    pred_latency_ms: Optional[float] = None


# ----------------------------
# Models
# ----------------------------

class Predictor:
    def __init__(self, model_path: Path, feature_order: List[str]):
        self.model_path = model_path
        self.feature_order = feature_order
        self.model = joblib.load(model_path)

    def predict_one(self, features: Dict[str, Any]) -> float:
        x = []
        for k in self.feature_order:
            if k not in features:
                raise KeyError(f"Missing feature '{k}' for model input")
            x.append(float(features[k]))
        # shape: (1, n_features)
        return float(self.model.predict([x])[0])


# ----------------------------
# Controller logic
# ----------------------------

def select_memory_tier(
    base_features: Dict[str, Any],
    tiers_mb: List[int],
    slo_ms: int,
    pred_latency: Predictor,
    pred_energy: Predictor,
) -> Tuple[int, Dict[int, Dict[str, float]]]:
    """
    For each tier m:
      - inject mem_limit_mb=m
      - predict latency and energy
    Choose:
      - min energy among tiers with latency <= SLO
      - else min latency
    Returns chosen tier and per-tier predictions.
    """
    per_tier: Dict[int, Dict[str, float]] = {}
    feasible: List[int] = []

    for m in tiers_mb:
        feats = dict(base_features)
        feats["mem_limit_mb"] = m
        t_hat = pred_latency.predict_one(feats)
        e_hat = pred_energy.predict_one(feats)
        per_tier[m] = {"latency_ms": t_hat, "energy_j": e_hat}
        if t_hat <= float(slo_ms):
            feasible.append(m)

    if feasible:
        chosen = min(feasible, key=lambda mm: per_tier[mm]["energy_j"])
    else:
        chosen = min(tiers_mb, key=lambda mm: per_tier[mm]["latency_ms"])

    return chosen, per_tier


# ----------------------------
# Workload runner (Docker)
# ----------------------------

def run_workload_in_docker(
    image: str,
    command: List[str],
    mem_limit_mb: int,
    cold_start: int,
    extra_docker_args: List[str],
) -> Tuple[int, int, float, float, int, int]:
    """
    Runs a workload in Docker with a given memory limit.
    Returns:
      duration_ms, cpu_time_ms, rss_mb, peak_rss_mb, io_read_bytes, io_write_bytes

    Cold start handling:
      - cold_start=1: we do not reuse container (docker run creates a new container anyway)
      - warm start: in true serverless you'd reuse a container; Docker run always creates new unless you implement reuse.
        For simplicity here: treat cold_start flag as metadata unless you build reuse mode.
    """
    # docker run --rm --memory 512m <image> <command...>
    docker_cmd = ["docker", "run", "--rm", f"--memory={mem_limit_mb}m"]
    docker_cmd += extra_docker_args
    docker_cmd.append(image)
    docker_cmd += command

    t0 = now_ms()

    # We want IO/RSS; easiest is to measure inside container (future improvement).
    # For now: measure only duration here. RSS/IO will be 0 unless we instrument inside the container.
    # We'll still keep the fields to match schema.

    # Run process
    try:
        # capture output for debugging (optional)
        proc = subprocess.run(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        rc = proc.returncode
    except FileNotFoundError:
        raise RuntimeError("Docker not found. Install Docker or adjust runner.")
    except Exception as e:
        raise RuntimeError(f"Failed to run docker workload: {e}")

    t1 = now_ms()
    duration = t1 - t0

    # Placeholder metrics (0). Best practice: instrument inside container and report.
    cpu_time_ms = 0
    rss_mb = 0.0
    peak_rss_mb = 0.0
    io_r = 0
    io_w = 0

    if rc != 0:
        # You can log stderr for debugging
        sys.stderr.write(proc.stderr[:2000] + "\n")

    return duration, cpu_time_ms, rss_mb, peak_rss_mb, io_r, io_w


# ----------------------------
# Main loop
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="windows_or_linux", help="Label for platform field")
    ap.add_argument("--run-id", default=str(uuid.uuid4())[:8], help="Run/session id")
    ap.add_argument("--workload", required=True, help="Workload name label")

    ap.add_argument("--tiers-mb", default=",".join(map(str, DEFAULT_MEMORY_TIERS_MB)),
                    help="Comma-separated memory tiers in MB")

    ap.add_argument("--slo-ms", type=int, default=200, help="Latency SLO (ms)")

    ap.add_argument("--model-latency", required=True, help="Path to latency regressor (joblib)")
    ap.add_argument("--model-energy", required=True, help="Path to energy regressor (joblib)")

    ap.add_argument("--log-jsonl", required=True, help="Output JSONL log file path")

    # Runtime-visible inputs (from serverless environment / harness)
    ap.add_argument("--cold-start", type=int, default=0, help="1=cold, 0=warm")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--queue-delay-ms", type=int, default=0)

    # Workload execution config
    ap.add_argument("--docker-image", required=True, help="Docker image containing workload")
    ap.add_argument("--docker-cmd", required=True,
                    help='Command inside container, e.g. "python run_workload.py --name=W --input=I"')
    ap.add_argument("--docker-extra-args", default="", help='Extra docker args, e.g. "--cpus=1"')

    # Optional: simple stabilization to avoid thrashing tiers
    ap.add_argument("--hysteresis", type=int, default=0,
                    help="If >0, limit tier changes to at most this many steps per invocation")

    args = ap.parse_args()

    tiers_mb = [int(x.strip()) for x in args.tiers_mb.split(",") if x.strip()]
    tiers_mb = sorted(set(tiers_mb))

    pred_T = Predictor(Path(args.model_latency), FEATURE_ORDER)
    pred_E = Predictor(Path(args.model_energy), FEATURE_ORDER)

    log_path = Path(args.log_jsonl)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    docker_cmd = args.docker_cmd.strip().split()
    docker_extra_args = args.docker_extra_args.strip().split() if args.docker_extra_args.strip() else []

    prev_tier: Optional[int] = None

    with log_path.open("a", encoding="utf-8") as logf:
        inv_idx = 0
        while not STOP_REQUESTED:
            inv_idx += 1
            invocation_id = f"{args.run_id}-{inv_idx:06d}"

            # Base features available BEFORE running (serverless-visible / known config)
            # NOTE: cpu_time/rss/io are usually known only AFTER execution unless you have
            # prior history or partial-progress features. For now, we treat them as
            # last-observed values (requires you to feed them in / update from last record).
            #
            # Minimal deployable approach:
            # - For first invocation: set unknowns to 0 or historical defaults
            # - After each invocation: update base_features from measured results
            #
            # Here we implement that "update from last invocation" loop.
            if inv_idx == 1:
                base_features = {
                    "cpu_time_ms": 0,
                    "rss_mb": 0,
                    "peak_rss_mb": 0,
                    "io_read_bytes": 0,
                    "io_write_bytes": 0,
                    "cold_start": int(args.cold_start),
                    "concurrency": int(args.concurrency),
                    "queue_delay_ms": int(args.queue_delay_ms),
                    "mem_limit_mb": tiers_mb[0],  # placeholder; overridden per tier evaluation
                }
            else:
                # base_features has already been updated at end of loop
                base_features["cold_start"] = int(args.cold_start)
                base_features["concurrency"] = int(args.concurrency)
                base_features["queue_delay_ms"] = int(args.queue_delay_ms)

            # Decide tier
            chosen, per_tier = select_memory_tier(
                base_features=base_features,
                tiers_mb=tiers_mb,
                slo_ms=args.slo_ms,
                pred_latency=pred_T,
                pred_energy=pred_E,
            )

            # Optional hysteresis: prevent big jumps
            if prev_tier is not None and args.hysteresis > 0:
                # move at most +/- hysteresis steps in sorted tier list
                idx_prev = tiers_mb.index(prev_tier)
                idx_new = tiers_mb.index(chosen)
                if abs(idx_new - idx_prev) > args.hysteresis:
                    idx_new = idx_prev + args.hysteresis * (1 if idx_new > idx_prev else -1)
                    chosen = tiers_mb[idx_new]

            # Execute workload at chosen tier
            ts0 = now_ms()
            duration_ms, cpu_time_ms, rss_mb, peak_rss_mb, io_r, io_w = run_workload_in_docker(
                image=args.docker_image,
                command=docker_cmd,
                mem_limit_mb=chosen,
                cold_start=int(args.cold_start),
                extra_docker_args=docker_extra_args,
            )
            ts1 = now_ms()

            # Store predictions for the chosen tier (for debugging)
            pred_latency_ms = per_tier[chosen]["latency_ms"]
            pred_energy_j = per_tier[chosen]["energy_j"]

            record = InvocationRecord(
                platform=args.platform,
                workload=args.workload,
                run_id=args.run_id,
                invocation_id=invocation_id,
                timestamp_start_ms=ts0,
                timestamp_end_ms=ts1,
                mem_limit_mb=chosen,
                cold_start=int(args.cold_start),
                concurrency=int(args.concurrency),
                queue_delay_ms=int(args.queue_delay_ms),
                duration_ms=int(duration_ms),
                cpu_time_ms=int(cpu_time_ms),
                rss_mb=float(rss_mb),
                peak_rss_mb=float(peak_rss_mb),
                io_read_bytes=int(io_r),
                io_write_bytes=int(io_w),
                pred_energy_joules=float(pred_energy_j),
                pred_latency_ms=float(pred_latency_ms),
            )

            logf.write(json.dumps(asdict(record)) + "\n")
            logf.flush()

            # Update base features from this just-finished invocation (online adaptation)
            base_features = {
                "cpu_time_ms": int(cpu_time_ms),
                "rss_mb": float(rss_mb),
                "peak_rss_mb": float(peak_rss_mb),
                "io_read_bytes": int(io_r),
                "io_write_bytes": int(io_w),
                "cold_start": int(args.cold_start),
                "concurrency": int(args.concurrency),
                "queue_delay_ms": int(args.queue_delay_ms),
                "mem_limit_mb": chosen,  # last used (not critical)
            }
            prev_tier = chosen

            # Optional: break after one invocation if you want single-shot mode
            # (you can add --max-invocations)
            # time.sleep(0.01)

    print(f"Stopped. Logs saved to: {log_path}")


if __name__ == "__main__":
    main()