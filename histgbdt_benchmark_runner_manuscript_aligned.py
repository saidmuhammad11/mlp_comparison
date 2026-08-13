#!/usr/bin/env python3
"""
HistGBDT Benchmark Runner for Windows
=====================================

Purpose
-------
Run real benchmark workloads under either:
  1) the trained HistGBDT-RPM controller, or
  2) a fixed Windows Balanced ("Medium") baseline,

while using the SAME telemetry/feature pipeline for both modes.

This version is intended for publishable benchmark experiments. It fixes the
main problems in the previous benchmark script:

- Uses the actual trained HistGBDT .joblib model (no threshold substitute).
- Uses the same feature order/scaler as the trained model.
- Validates the model/scaler feature count before running.
- Keeps a handle to the benchmark process and supports run-to-completion.
- Applies OMP/MKL/OpenBLAS/NumExpr thread limits consistently.
- Uses the original Low/Medium/High Windows power-plan mapping only.
  It DOES NOT modify PROCTHROTTLEMIN/MAX.
- Avoids redundant powercfg calls when the predicted level has not changed.
- Uses the same telemetry collection path for HistGBDT and baseline.
- Never silently generates simulated measurements. Measurement failure aborts
  the experiment unless --allow-simulated is explicitly supplied for debugging.
- Uses the manuscript-aligned 11-feature HistGBDT input with 8-sample
  rolling IPC/L3 histories (no extra bw_util model feature).
- Saves:
    * JSONL time-series log
    * benchmark stdout/stderr
    * summary JSON containing runtime, integrated package energy, averages,
      state counts, switch count, inference time, monitoring overhead, and
      fixed-work evidence (workload hash/signature and parsed benchmark
      completion/verification metadata when printed by the benchmark).
- No live plotting during the benchmark, to avoid adding avoidable experimental
  overhead. Plot the JSONL after the experiment.

Recommended use for equal-work benchmark validation
---------------------------------------------------
Run the SAME benchmark / input / problem size / thread count to completion:

    python histgbdt_benchmark_runner.py ^
        --workload NPB3.0-omp-C/workload_cpu1_nas_ep.py ^
        --controller balanced ^
        --cores 4 ^
        --run-mode completion

    python histgbdt_benchmark_runner.py ^
        --workload NPB3.0-omp-C/workload_cpu1_nas_ep.py ^
        --controller hgbdt ^
        --cores 4 ^
        --run-mode completion

For custom continuously-running workloads, use:
    --run-mode duration --duration 300

IMPORTANT
---------
The FEATURE_ORDER below MUST match the model used to produce histgbdt.joblib.
This script fails loudly if model.n_features_in_ or scaler dimensions disagree.
Do not delete/add/reorder features without retraining the model.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import psutil


# ---------------------------------------------------------------------------
# Global stop handling
# ---------------------------------------------------------------------------
STOP_REQUESTED = False


def handle_ctrl_c(sig, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\nCtrl+C detected - requesting a clean shutdown...")


signal.signal(signal.SIGINT, handle_ctrl_c)


ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL_FILE = ROOT / "models" / "histgbdt.joblib"
DEFAULT_SCALER_FILE = ROOT / "dataset" / "scaler_stats.npz"

# Keep this identical to the manuscript/runtime/training pipeline.
# HistGBDT-RPM uses an 8-sample bounded history for IPC and L3 miss rate.
N_ROLLING_SAMPLES = 8

# Keep the same plan definitions used by the original HistGBDT experiments.
POWER_PLANS = {
    "High": "27ad8305-4092-41f2-a01b-5a6003fb5077",
    "Medium": "381b4222-f694-41f0-9685-ff5bb260df2e",
    "Low": "b2524225-86dc-424d-ba5b-e78d683c5d3a",
}

PCM_CANDIDATES = [
    ROOT / "pcm.exe",
    Path(r"C:\Users\saidm\OneDrive\Desktop\Programming\pcm\build\bin\Release\pcm.exe"),
    Path(r"C:\Users\saidm\Desktop\Programming\pcm\build\bin\Release\pcm.exe"),
    Path(r"C:\Users\saidm\pcm\build\bin\Release\pcm.exe"),
]

PCM_POWER_CANDIDATES = [
    ROOT / "pcm-power.exe",
    Path(r"C:\Users\saidm\OneDrive\Desktop\Programming\pcm\build\bin\Release\pcm-power.exe"),
    Path(r"C:\Users\saidm\Desktop\Programming\pcm\build\bin\Release\pcm-power.exe"),
    Path(r"C:\Users\saidm\pcm\build\bin\Release\pcm-power.exe"),
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def resolve_executable(explicit: Optional[str], candidates: List[Path], command_name: str) -> Optional[str]:
    """Resolve an executable from an explicit path, known paths, or PATH."""
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return str(p.resolve())
        found = shutil.which(explicit)
        if found:
            return found
        return None

    for p in candidates:
        if p.exists():
            return str(p.resolve())

    return shutil.which(command_name)


def load_scaler_stats(path: Optional[Path]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load scaler mean/scale from an .npz file."""
    if path is None or not path.exists():
        return None, None

    stats = np.load(path)

    if "mean" in stats:
        mean = stats["mean"].astype(np.float32)
    elif "mean_" in stats:
        mean = stats["mean_"].astype(np.float32)
    else:
        raise KeyError(f"No 'mean' or 'mean_' key found in scaler file. Keys: {list(stats.keys())}")

    if "scale" in stats:
        scale = stats["scale"].astype(np.float32)
    elif "scale_" in stats:
        scale = stats["scale_"].astype(np.float32)
    else:
        raise KeyError(f"No 'scale' or 'scale_' key found in scaler file. Keys: {list(stats.keys())}")

    scale = np.where(scale == 0, 1.0, scale)
    return mean, scale


def trapezoid_integral(times_s: np.ndarray, values: np.ndarray, duration_s: float) -> float:
    """
    Integrate a sampled signal over [0, duration_s].

    The first/last measured values are extended to the exact benchmark
    boundaries. This is still an estimate, but it is preferable to mean*nominal
    duration when sampling intervals are not perfectly uniform.
    """
    if len(times_s) == 0 or duration_s <= 0:
        return 0.0

    order = np.argsort(times_s)
    t = np.asarray(times_s[order], dtype=float)
    y = np.asarray(values[order], dtype=float)

    # Keep only samples in/near the benchmark interval.
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if len(t) == 0:
        return 0.0

    # Clamp timestamps to benchmark boundaries.
    t = np.clip(t, 0.0, duration_s)

    # Collapse duplicate timestamps if any.
    unique_t = []
    unique_y = []
    for ti, yi in zip(t, y):
        if unique_t and abs(ti - unique_t[-1]) < 1e-12:
            unique_y[-1] = yi
        else:
            unique_t.append(float(ti))
            unique_y.append(float(yi))

    t = np.asarray(unique_t, dtype=float)
    y = np.asarray(unique_y, dtype=float)

    if len(t) == 1:
        return float(y[0] * duration_s)

    if t[0] > 0.0:
        t = np.insert(t, 0, 0.0)
        y = np.insert(y, 0, y[0])

    if t[-1] < duration_s:
        t = np.append(t, duration_s)
        y = np.append(y, y[-1])

    # np.trapezoid exists in newer NumPy; np.trapz keeps compatibility.
    return float(np.trapz(y, t))


def time_weighted_average(times_s: np.ndarray, values: np.ndarray, duration_s: float) -> float:
    if duration_s <= 0:
        return float("nan")
    return trapezoid_integral(times_s, values, duration_s) / duration_s


def sha256_file(path: Path) -> Optional[str]:
    """Return a SHA-256 digest for a workload file, if it is readable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def parse_benchmark_work_metadata(stdout_path: Path) -> Dict:
    """Extract conservative fixed-work/verification evidence from benchmark stdout.

    The parser intentionally records only fields that are explicitly printed by
    the benchmark/wrapper.  Missing fields remain absent; nothing is inferred.
    It recognizes common NPB output plus optional burst-wrapper counters.
    """
    metadata: Dict[str, object] = {}
    try:
        text = stdout_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return metadata

    patterns = {
        "benchmark_class": r"(?im)^\s*Class\s*[:=]\s*(.+?)\s*$",
        "problem_size": r"(?im)^\s*Size\s*[:=]\s*(.+?)\s*$",
        "iterations": r"(?im)^\s*Iterations\s*[:=]\s*(.+?)\s*$",
        "reported_threads": r"(?im)^\s*(?:Total\s+threads|Threads)\s*[:=]\s*(.+?)\s*$",
        "verification": r"(?im)^\s*Verification\s*[:=]\s*(.+?)\s*$",
        "requested_invocations": r"(?im)^\s*Requested\s+invocations\s*[:=]\s*(\d+)\s*$",
        "completed_invocations": r"(?im)^\s*Completed\s+invocations\s*[:=]\s*(\d+)\s*$",
        "inter_invocation_delay_s": r"(?im)^\s*Inter[- ]invocation\s+delay(?:\s*\(s\))?\s*[:=]\s*([0-9.]+)\s*$",
        "random_numbers_generated": r"(?im)^\s*Number\s+of\s+random\s+numbers\s+generated\s*[:=]\s*(.+?)\s*$",
        "array_size": r"(?im)^\s*(?:Array\s+size|ARRAY_SIZE)\s*[:=]\s*(.+?)\s*$",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            value = m.group(1).strip()
            if key in {"requested_invocations", "completed_invocations"}:
                try:
                    metadata[key] = int(value)
                    continue
                except ValueError:
                    pass
            if key == "inter_invocation_delay_s":
                try:
                    metadata[key] = float(value)
                    continue
                except ValueError:
                    pass
            metadata[key] = value

    metadata["completion_banner_detected"] = bool(
        re.search(r"(?i)benchmark\s+completed|completed\s+successfully", text)
    )
    verification = str(metadata.get("verification", "")).lower()
    if verification:
        metadata["verification_successful"] = (
            "successful" in verification or verification.strip() in {"success", "passed", "pass"}
        )

    req = metadata.get("requested_invocations")
    done = metadata.get("completed_invocations")
    if isinstance(req, int) and isinstance(done, int):
        metadata["invocation_count_matches"] = (req == done)

    return metadata


def fixed_work_signature(workload: Path, workload_args: List[str], cores: int) -> Dict:
    """Build controller-independent identifiers for pairing equal-work runs."""
    workload_hash = sha256_file(workload)
    canonical = {
        "workload_name": workload.name,
        "workload_sha256": workload_hash,
        "workload_args": list(workload_args),
        "thread_count": int(cores),
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    canonical["signature_sha256"] = hashlib.sha256(payload).hexdigest()
    return canonical


# ---------------------------------------------------------------------------
# HistGBDT model wrapper
# ---------------------------------------------------------------------------
class HistGBDTModel:
    """
    Wrapper around the trained HistGradientBoostingClassifier.

    The deployed benchmark controller is intentionally aligned with the
    manuscript: exactly 11 inputs, in the same order as Eq. (x_k), with
    8-sample rolling IPC and L3-miss-rate histories.  A model/scaler exported
    with any other dimensionality is rejected rather than silently changing
    the controller evaluated in the paper.
    """

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
        "ipc_avg_8",
        "l3_miss_rate_avg_8",
    ]

    def __init__(
        self,
        model_path: Path,
        scaler_path: Optional[Path],
        enable_proba_hysteresis: bool = False,
        proba_margin: float = 0.15,
    ):
        if not model_path.exists():
            raise FileNotFoundError(f"HistGBDT model not found: {model_path}")

        self.model = joblib.load(model_path)
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.scaler_mean, self.scaler_scale = load_scaler_stats(scaler_path)

        self.enable_proba_hysteresis = enable_proba_hysteresis
        self.proba_margin = float(proba_margin)
        self.last_level = "Medium"

        self.classes = list(getattr(self.model, "classes_", [0, 1, 2]))
        self.class_to_level = self._build_class_mapping(self.classes)

        self._validate_feature_dimensions()

        print(f"Loaded HistGBDT model: {model_path}")
        if scaler_path and self.scaler_mean is not None:
            print(f"Loaded scaler stats:   {scaler_path}")
        else:
            print("WARNING: no scaler stats loaded.")

        print(f"Model classes:         {self.classes}")
        print(f"Feature count:         {len(self.FEATURE_ORDER)}")

    @staticmethod
    def _build_class_mapping(classes) -> Dict:
        classes_list = list(classes)

        if all(isinstance(c, str) for c in classes_list):
            mapping = {}
            for c in classes_list:
                norm = c.strip().lower()
                if norm == "low":
                    mapping[c] = "Low"
                elif norm == "medium":
                    mapping[c] = "Medium"
                elif norm == "high":
                    mapping[c] = "High"
                else:
                    raise ValueError(f"Unexpected string class label: {c!r}")
            return mapping

        try:
            class_ints = [int(c) for c in classes_list]
        except Exception as exc:
            raise ValueError(f"Unsupported HistGBDT classes_: {classes_list}") from exc

        if set(class_ints) == {0, 1, 2}:
            return {0: "Low", 1: "Medium", 2: "High"}

        if len(class_ints) != 3:
            raise ValueError(
                f"Expected exactly 3 classes for Low/Medium/High, got: {classes_list}"
            )

        # Fallback for a nonstandard numeric encoding.
        ordered = sorted(class_ints)
        return {ordered[0]: "Low", ordered[1]: "Medium", ordered[2]: "High"}

    def _validate_feature_dimensions(self):
        expected = len(self.FEATURE_ORDER)

        model_n = getattr(self.model, "n_features_in_", None)
        if model_n is not None and int(model_n) != expected:
            raise ValueError(
                f"MODEL FEATURE MISMATCH: model expects {model_n} features, "
                f"but FEATURE_ORDER defines {expected}: {self.FEATURE_ORDER}"
            )

        if self.scaler_mean is not None and len(self.scaler_mean) != expected:
            raise ValueError(
                f"SCALER FEATURE MISMATCH: scaler mean has {len(self.scaler_mean)} "
                f"values, expected {expected}."
            )

        if self.scaler_scale is not None and len(self.scaler_scale) != expected:
            raise ValueError(
                f"SCALER FEATURE MISMATCH: scaler scale has {len(self.scaler_scale)} "
                f"values, expected {expected}."
            )

    def _prepare_vector(self, features: Dict[str, float]) -> np.ndarray:
        missing = [name for name in self.FEATURE_ORDER if name not in features]
        if missing:
            raise KeyError(f"Missing HistGBDT input features: {missing}")

        x = np.asarray(
            [float(features[name]) for name in self.FEATURE_ORDER],
            dtype=np.float32,
        )

        if not np.all(np.isfinite(x)):
            bad = {
                name: float(value)
                for name, value in zip(self.FEATURE_ORDER, x)
                if not np.isfinite(value)
            }
            raise ValueError(f"Non-finite model input(s): {bad}")

        if self.scaler_mean is not None and self.scaler_scale is not None:
            x = (x - self.scaler_mean) / self.scaler_scale

        return x.reshape(1, -1)

    def predict_level(self, features: Dict[str, float]) -> Tuple[str, float, Optional[float]]:
        """
        Return:
            level
            inference_time_ms
            probability_margin (None if predict_proba is not used)
        """
        X = self._prepare_vector(features)

        t0 = time.perf_counter()
        probability_margin = None

        if self.enable_proba_hysteresis and hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            sorted_idx = np.argsort(proba)[::-1]

            top_idx = int(sorted_idx[0])
            second_idx = int(sorted_idx[1])
            probability_margin = float(proba[top_idx] - proba[second_idx])

            # IMPORTANT: predict_proba columns correspond to model.classes_.
            predicted_class = self.classes[top_idx]
            try:
                class_key = int(predicted_class)
            except Exception:
                class_key = predicted_class

            new_level = self.class_to_level[class_key]

            if new_level != self.last_level and probability_margin < self.proba_margin:
                level = self.last_level
            else:
                level = new_level
                self.last_level = level
        else:
            predicted_class = self.model.predict(X)[0]
            try:
                class_key = int(predicted_class)
            except Exception:
                class_key = predicted_class

            level = self.class_to_level[class_key]
            self.last_level = level

        inference_ms = (time.perf_counter() - t0) * 1000.0
        return level, inference_ms, probability_margin


# ---------------------------------------------------------------------------
# Benchmark / controller runner
# ---------------------------------------------------------------------------
class HistGBDTBenchmarkRunner:
    def __init__(
        self,
        model: HistGBDTModel,
        controller: str,
        pcm_path: str,
        pcm_power_path: str,
        output_dir: Path,
        sample_sleep_s: float = 3.0,
        allow_simulated: bool = False,
    ):
        if controller not in {"hgbdt", "balanced"}:
            raise ValueError("controller must be 'hgbdt' or 'balanced'")

        self.model = model
        self.controller = controller
        self.pcm_path = pcm_path
        self.pcm_power_path = pcm_power_path
        self.output_dir = output_dir
        self.sample_sleep_s = float(sample_sleep_s)
        self.allow_simulated = allow_simulated

        self.data_buffer: List[Dict] = []
        self.active_level: Optional[str] = None
        self.switch_count = 0
        self.powercfg_time_s = 0.0
        self.benchmark_process: Optional[subprocess.Popen] = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------- power plan control --------------------------
    def set_power_plan(
        self,
        level: str,
        force: bool = False,
        record_experiment: bool = True,
    ) -> Tuple[bool, float]:
        if level not in POWER_PLANS:
            raise ValueError(f"Invalid level: {level}")

        # Avoid redundant powercfg calls. A decision that keeps the same state
        # should not incur an artificial actuation overhead.
        if not force and level == self.active_level:
            return True, 0.0

        guid = POWER_PLANS[level]
        t0 = time.perf_counter()

        try:
            result = subprocess.run(
                ["powercfg", "-setactive", guid],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to activate Windows power plan {level}: {exc}") from exc

        elapsed = time.perf_counter() - t0
        previous = self.active_level
        self.active_level = level
        if record_experiment:
            self.powercfg_time_s += elapsed
            if previous is not None and previous != level:
                self.switch_count += 1

        return True, elapsed

    # -------------------------- telemetry --------------------------
    def _simulate_pcm_data(self) -> Dict[str, float]:
        return {
            "ipc": float(np.random.uniform(0.5, 2.0)),
            "l2_cache_hits": int(np.random.randint(50000, 80000)),
            "l2_cache_misses": int(np.random.randint(1000, 5000)),
            "l3_cache_hits": int(np.random.randint(30000, 60000)),
            "l3_cache_misses": int(np.random.randint(500, 3000)),
            "memory_bandwidth": float(np.random.uniform(10000, 25000)),
        }

    def _simulate_power_data(self) -> Dict[str, float]:
        return {
            "cpu_power": float(np.random.uniform(25, 65)),
            "gpu_power": float(np.random.uniform(5, 20)),
            "cpu_temperature": float(np.random.uniform(50, 85)),
            "cpu_frequency": float(np.random.uniform(1800, 3600)),
        }

    def read_pcm_data(self) -> Dict[str, float]:
        """
        Preserve the parser used by the working original runtime.

        If your pcm.exe output format changes, update ONLY this parser after
        confirming the exact column layout. This function does not silently
        invent values in publication mode.
        """
        try:
            result = subprocess.run(
                [self.pcm_path, "1", "-nc", "-ns"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"pcm.exe returned {result.returncode}: {result.stderr.strip()}"
                )

            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if len(lines) < 2:
                raise RuntimeError("Unexpected pcm.exe output: fewer than two non-empty lines.")

            data_line = lines[-1]
            parts = [x for x in data_line.split(" ") if x]

            if len(parts) < 8:
                raise RuntimeError(
                    f"Unexpected pcm.exe data row ({len(parts)} tokens): {data_line}"
                )

            ipc = float(parts[1])
            mem_bw = float(parts[2])
            l2_hits = int(parts[4])
            l2_misses = int(parts[5])
            l3_hits = int(parts[6])
            l3_misses = int(parts[7])

            return {
                "ipc": ipc,
                "l2_cache_hits": l2_hits,
                "l2_cache_misses": l2_misses,
                "l3_cache_hits": l3_hits,
                "l3_cache_misses": l3_misses,
                "memory_bandwidth": mem_bw,
            }

        except Exception:
            if self.allow_simulated:
                print("WARNING: PCM read failed; using simulated PCM data because --allow-simulated was set.")
                return self._simulate_pcm_data()
            raise

    def read_power_data(self) -> Dict[str, float]:
        # Use a unique temporary path to avoid stale/colliding pcm-power files.
        fd, temp_name = tempfile.mkstemp(
            prefix="pcm_power_",
            suffix=".csv",
            dir=str(self.output_dir),
        )
        os.close(fd)

        temp_path = Path(temp_name)
        try:
            # pcm-power may expect to create/overwrite the file.
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass

            result = subprocess.run(
                [self.pcm_power_path, "-duration", "1", "-file", str(temp_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"pcm-power.exe returned {result.returncode}: {result.stderr.strip()}"
                )

            if not temp_path.exists():
                raise RuntimeError("pcm-power.exe did not create the expected CSV file.")

            lines = [line.strip() for line in temp_path.read_text(errors="replace").splitlines() if line.strip()]
            if len(lines) < 2:
                raise RuntimeError("Unexpected pcm-power CSV: fewer than two non-empty lines.")

            values = lines[-1].split(",")

            if len(values) < 5:
                raise RuntimeError(f"Unexpected pcm-power CSV row: {lines[-1]}")

            return {
                "cpu_power": float(values[1]) if values[1] else 0.0,
                "gpu_power": float(values[2]) if values[2] else 0.0,
                "cpu_temperature": float(values[3]) if values[3] else 0.0,
                "cpu_frequency": float(values[4]) if values[4] else 0.0,
            }

        except Exception:
            if self.allow_simulated:
                print("WARNING: power read failed; using simulated power data because --allow-simulated was set.")
                return self._simulate_power_data()
            raise
        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass

    # -------------------------- feature engineering --------------------------
    def feature_engineer(self, data: Dict) -> Dict:
        history = self.data_buffer + [data]
        df = pd.DataFrame(history)

        data["ipc_change"] = (
            float(df["ipc"].diff().iloc[-1]) if len(df) > 1 else 0.0
        )
        data["ipc_avg_8"] = float(
            df["ipc"]
            .rolling(window=N_ROLLING_SAMPLES, min_periods=1)
            .mean()
            .iloc[-1]
        )
        data["l3_miss_rate_avg_8"] = float(
            df["l3_miss_rate"]
            .rolling(window=N_ROLLING_SAMPLES, min_periods=1)
            .mean()
            .iloc[-1]
        )

        # Memory bandwidth itself is one of the 11 manuscript features.
        # Do not add a derived bw_util input: that would change the deployed
        # model from the 11-dimensional controller described in the paper.
        return data

    def collect_sample(self, benchmark_start_perf: float) -> Dict:
        sample_collect_start = time.perf_counter()

        pcm_t0 = time.perf_counter()
        pcm = self.read_pcm_data()
        pcm_read_s = time.perf_counter() - pcm_t0

        power_t0 = time.perf_counter()
        power = self.read_power_data()
        power_read_s = time.perf_counter() - power_t0

        psutil_t0 = time.perf_counter()
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().percent
        psutil_read_s = time.perf_counter() - psutil_t0

        total_l2 = pcm["l2_cache_hits"] + pcm["l2_cache_misses"]
        total_l3 = pcm["l3_cache_hits"] + pcm["l3_cache_misses"]

        l2_miss_rate = pcm["l2_cache_misses"] / total_l2 if total_l2 > 0 else 0.0
        l3_miss_rate = pcm["l3_cache_misses"] / total_l3 if total_l3 > 0 else 0.0

        sample_perf = time.perf_counter()

        data = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_s": float(sample_perf - benchmark_start_perf),
            "cpu_usage_overall": float(cpu_usage),
            "memory_usage": float(memory_usage),
            "ipc": float(pcm["ipc"]),
            "l2_miss_rate": round(float(l2_miss_rate), 4),
            "l3_miss_rate": round(float(l3_miss_rate), 4),
            "memory_bandwidth": float(pcm["memory_bandwidth"]),
            "cpu_power": round(float(power["cpu_power"]), 3),
            "cpu_temperature": round(float(power["cpu_temperature"]), 3),
            "cpu_frequency": round(float(power["cpu_frequency"]), 3),
            "pcm_read_s": float(pcm_read_s),
            "power_read_s": float(power_read_s),
            "psutil_read_s": float(psutil_read_s),
        }

        data = self.feature_engineer(data)

        inference_ms = 0.0
        probability_margin = None

        if self.controller == "hgbdt":
            predicted_level, inference_ms, probability_margin = self.model.predict_level(data)
            applied, actuation_s = self.set_power_plan(predicted_level)
            level = predicted_level
        else:
            # Baseline plan is set once before launch. Do not repeatedly call powercfg.
            level = "Medium"
            applied = True
            actuation_s = 0.0

        data["controller"] = self.controller
        data["dvfs_level"] = level
        data["dvfs_applied"] = bool(applied)
        data["inference_ms"] = float(inference_ms)
        data["probability_margin"] = (
            None if probability_margin is None else float(probability_margin)
        )
        data["actuation_s"] = float(actuation_s)
        data["sample_collection_s"] = float(time.perf_counter() - sample_collect_start)

        return data

    # -------------------------- benchmark process --------------------------
    def build_benchmark_command(
        self,
        workload: Path,
        workload_args: List[str],
    ) -> List[str]:
        suffix = workload.suffix.lower()

        if suffix == ".py":
            return [sys.executable, str(workload)] + workload_args

        # Executable or command-like file.
        return [str(workload)] + workload_args

    def start_benchmark(
        self,
        workload: Path,
        workload_args: List[str],
        cores: int,
        workdir: Optional[Path],
    ) -> Tuple[subprocess.Popen, Path, Path]:
        if not workload.exists():
            raise FileNotFoundError(f"Benchmark/workload not found: {workload}")

        cwd = (workdir or workload.parent).resolve()
        if not cwd.exists():
            raise FileNotFoundError(f"Benchmark working directory not found: {cwd}")

        env = os.environ.copy()
        thread_count = str(cores)

        # Keep common CPU-library thread counts controlled and reproducible.
        env["OMP_NUM_THREADS"] = thread_count
        env["OMP_DYNAMIC"] = "FALSE"
        env["MKL_NUM_THREADS"] = thread_count
        env["MKL_DYNAMIC"] = "FALSE"
        env["OPENBLAS_NUM_THREADS"] = thread_count
        env["NUMEXPR_NUM_THREADS"] = thread_count

        stdout_path = self.output_dir / "benchmark_stdout.txt"
        stderr_path = self.output_dir / "benchmark_stderr.txt"

        stdout_f = open(stdout_path, "w", buffering=1)
        stderr_f = open(stderr_path, "w", buffering=1)

        cmd = self.build_benchmark_command(workload, workload_args)

        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        print("Benchmark command:")
        print("  " + " ".join(f'"{x}"' if " " in x else x for x in cmd))
        print(f"Working directory: {cwd}")
        print(f"Thread count:      {cores}")

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=stdout_f,
                stderr=stderr_f,
                creationflags=creationflags,
            )
        except Exception:
            stdout_f.close()
            stderr_f.close()
            raise

        # Attach file handles so they stay alive and can be closed later.
        proc._benchmark_stdout_handle = stdout_f  # type: ignore[attr-defined]
        proc._benchmark_stderr_handle = stderr_f  # type: ignore[attr-defined]

        self.benchmark_process = proc
        return proc, stdout_path, stderr_path

    def stop_benchmark(self):
        proc = self.benchmark_process
        if proc is None or proc.poll() is not None:
            return

        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        finally:
            self.benchmark_process = None

    @staticmethod
    def close_benchmark_streams(proc: subprocess.Popen):
        for attr in ("_benchmark_stdout_handle", "_benchmark_stderr_handle"):
            handle = getattr(proc, attr, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass

    # -------------------------- experiment --------------------------
    def run(
        self,
        workload: Path,
        workload_args: List[str],
        cores: int,
        run_mode: str,
        duration_s: Optional[float],
        settle_s: float,
        workdir: Optional[Path],
    ) -> Dict:
        global STOP_REQUESTED

        if run_mode not in {"completion", "duration"}:
            raise ValueError("run_mode must be 'completion' or 'duration'")

        if run_mode == "duration" and (duration_s is None or duration_s <= 0):
            raise ValueError("--duration must be > 0 in duration mode")

        # Start from a known state for BOTH benchmark modes.
        self.set_power_plan("Medium", force=True, record_experiment=False)

        if settle_s > 0:
            print(f"Settling in Medium plan for {settle_s:.1f} s...")
            time.sleep(settle_s)

        # Prime psutil so the first subsequent reading is meaningful.
        psutil.cpu_percent(interval=None)

        log_path = self.output_dir / "monitoring.jsonl"

        proc, stdout_path, stderr_path = self.start_benchmark(
            workload=workload,
            workload_args=workload_args,
            cores=cores,
            workdir=workdir,
        )

        benchmark_start_perf = time.perf_counter()
        benchmark_start_wall = datetime.now()

        print(f"Benchmark PID:     {proc.pid}")
        print(f"Controller:        {self.controller}")
        print(f"Run mode:          {run_mode}")
        if run_mode == "duration":
            print(f"Target duration:   {duration_s:.3f} s")
        print(f"Sampling sleep:    {self.sample_sleep_s:.3f} s")
        print("-" * 72)

        exit_reason = "completed"
        iteration = 0

        try:
            with open(log_path, "w", buffering=1) as log_f:
                while not STOP_REQUESTED:
                    now = time.perf_counter()
                    elapsed = now - benchmark_start_perf

                    # Completion mode: stop when benchmark exits.
                    if run_mode == "completion" and proc.poll() is not None:
                        exit_reason = "benchmark_completed"
                        break

                    # Duration mode: stop at requested wall-clock duration.
                    if run_mode == "duration" and duration_s is not None and elapsed >= duration_s:
                        exit_reason = "duration_reached"
                        break

                    iteration += 1

                    sample = self.collect_sample(benchmark_start_perf)
                    sample["benchmark_alive_after_sample"] = proc.poll() is None

                    self.data_buffer.append(sample)
                    log_f.write(json.dumps(sample) + "\n")
                    log_f.flush()

                    print(
                        f"[{iteration:03d}] "
                        f"t={sample['elapsed_s']:8.2f}s | "
                        f"{sample['dvfs_level']:6s} | "
                        f"IPC={sample['ipc']:5.2f} | "
                        f"P={sample['cpu_power']:6.2f} W | "
                        f"f={sample['cpu_frequency']:7.1f} MHz | "
                        f"CPU={sample['cpu_usage_overall']:5.1f}% | "
                        f"infer={sample['inference_ms']:7.3f} ms"
                    )

                    # If the benchmark completed while telemetry was being read,
                    # do not sleep before checking again.
                    if proc.poll() is not None and run_mode == "completion":
                        exit_reason = "benchmark_completed"
                        break

                    if self.sample_sleep_s > 0:
                        time.sleep(self.sample_sleep_s)

            if STOP_REQUESTED:
                exit_reason = "user_interrupted"

            if run_mode == "duration" and proc.poll() is None:
                self.stop_benchmark()

            if proc.poll() is None:
                # User interruption or unexpected path.
                self.stop_benchmark()

            try:
                return_code = proc.wait(timeout=10)
            except Exception:
                return_code = proc.poll()

        finally:
            benchmark_end_perf = time.perf_counter()
            benchmark_end_wall = datetime.now()
            self.close_benchmark_streams(proc)

            # Restore Balanced/Medium after the experiment.
            try:
                self.set_power_plan("Medium", force=True, record_experiment=False)
            except Exception as exc:
                print(f"WARNING: could not restore Medium power plan: {exc}")

        benchmark_duration_s = benchmark_end_perf - benchmark_start_perf

        summary = self.build_summary(
            workload=workload,
            workload_args=workload_args,
            cores=cores,
            run_mode=run_mode,
            requested_duration_s=duration_s,
            benchmark_duration_s=benchmark_duration_s,
            benchmark_start_wall=benchmark_start_wall,
            benchmark_end_wall=benchmark_end_wall,
            exit_reason=exit_reason,
            return_code=return_code,
            log_path=log_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("=" * 72)
        print("EXPERIMENT SUMMARY")
        print(f"Duration:              {summary['benchmark_duration_s']:.3f} s")
        print(f"Package energy:        {summary['cpu_package_energy_j']:.3f} J")
        print(f"Average package power: {summary['avg_cpu_power_w']:.3f} W")
        print(f"Average IPC:           {summary['avg_ipc']:.4f}")
        print(f"Average frequency:     {summary['avg_cpu_frequency_mhz']:.2f} MHz")
        print(f"Power-plan switches:   {summary['power_plan_switches']}")
        print(f"Samples:               {summary['samples']}")
        fw = summary["fixed_work_evidence"]
        print(f"Completed successfully:{str(fw['completed_successfully']):>9s}")
        print(f"Work signature:        {fw['work_signature']['signature_sha256'][:16]}...")
        print(f"Summary saved to:      {summary_path}")
        print(f"Time-series saved to:  {log_path}")
        print(f"Benchmark stdout:      {stdout_path}")
        print(f"Benchmark stderr:      {stderr_path}")

        return summary

    def build_summary(
        self,
        workload: Path,
        workload_args: List[str],
        cores: int,
        run_mode: str,
        requested_duration_s: Optional[float],
        benchmark_duration_s: float,
        benchmark_start_wall: datetime,
        benchmark_end_wall: datetime,
        exit_reason: str,
        return_code: Optional[int],
        log_path: Path,
        stdout_path: Path,
        stderr_path: Path,
    ) -> Dict:
        if self.data_buffer:
            df = pd.DataFrame(self.data_buffer)
            times = df["elapsed_s"].to_numpy(dtype=float)

            energy_j = trapezoid_integral(
                times,
                df["cpu_power"].to_numpy(dtype=float),
                benchmark_duration_s,
            )

            avg_power = energy_j / benchmark_duration_s if benchmark_duration_s > 0 else float("nan")
            avg_ipc = time_weighted_average(
                times, df["ipc"].to_numpy(dtype=float), benchmark_duration_s
            )
            avg_freq = time_weighted_average(
                times, df["cpu_frequency"].to_numpy(dtype=float), benchmark_duration_s
            )
            avg_cpu = time_weighted_average(
                times, df["cpu_usage_overall"].to_numpy(dtype=float), benchmark_duration_s
            )
            avg_temp = time_weighted_average(
                times, df["cpu_temperature"].to_numpy(dtype=float), benchmark_duration_s
            )

            state_counts = {
                str(k): int(v)
                for k, v in df["dvfs_level"].value_counts().to_dict().items()
            }

            avg_inference_ms = float(df["inference_ms"].mean())
            max_inference_ms = float(df["inference_ms"].max())
            total_monitoring_collection_s = float(df["sample_collection_s"].sum())
            avg_sample_collection_s = float(df["sample_collection_s"].mean())
        else:
            energy_j = 0.0
            avg_power = float("nan")
            avg_ipc = float("nan")
            avg_freq = float("nan")
            avg_cpu = float("nan")
            avg_temp = float("nan")
            state_counts = {}
            avg_inference_ms = 0.0
            max_inference_ms = 0.0
            total_monitoring_collection_s = 0.0
            avg_sample_collection_s = 0.0

        stdout_work_metadata = parse_benchmark_work_metadata(stdout_path)
        work_signature = fixed_work_signature(workload, workload_args, cores)
        completed_successfully = bool(
            run_mode == "completion"
            and exit_reason == "benchmark_completed"
            and return_code == 0
        )

        fixed_work_evidence = {
            "basis": (
                "same workload/input/thread-count executed to process completion"
                if run_mode == "completion"
                else "fixed-duration run; equal completed work is not implied"
            ),
            "eligible_for_equal_work_pairing": bool(run_mode == "completion"),
            "completed_successfully": completed_successfully,
            "work_signature": work_signature,
            "benchmark_reported_metadata": stdout_work_metadata,
        }

        return {
            "experiment_timestamp": datetime.now().isoformat(),
            "controller": self.controller,
            "workload": str(workload),
            "workload_args": workload_args,
            "cores": int(cores),
            "run_mode": run_mode,
            "requested_duration_s": requested_duration_s,
            "benchmark_start": benchmark_start_wall.isoformat(),
            "benchmark_end": benchmark_end_wall.isoformat(),
            "benchmark_duration_s": float(benchmark_duration_s),
            "exit_reason": exit_reason,
            "return_code": return_code,
            "samples": int(len(self.data_buffer)),
            "cpu_package_energy_j": float(energy_j),
            "avg_cpu_power_w": float(avg_power),
            "avg_ipc": float(avg_ipc),
            "avg_cpu_frequency_mhz": float(avg_freq),
            "avg_cpu_usage_percent": float(avg_cpu),
            "avg_cpu_temperature_c": float(avg_temp),
            "power_plan_switches": int(self.switch_count),
            "state_decision_counts": state_counts,
            "avg_inference_ms": float(avg_inference_ms),
            "max_inference_ms": float(max_inference_ms),
            "total_monitoring_collection_s": float(total_monitoring_collection_s),
            "avg_sample_collection_s": float(avg_sample_collection_s),
            "powercfg_total_time_s": float(self.powercfg_time_s),
            "model_path": str(self.model.model_path),
            "scaler_path": str(self.model.scaler_path) if self.model.scaler_path else None,
            "feature_order": list(self.model.FEATURE_ORDER),
            "rolling_history_samples": int(N_ROLLING_SAMPLES),
            "fixed_work_evidence": fixed_work_evidence,
            "pcm_path": self.pcm_path,
            "pcm_power_path": self.pcm_power_path,
            "log_file": str(log_path),
            "benchmark_stdout_file": str(stdout_path),
            "benchmark_stderr_file": str(stderr_path),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run benchmarks with the actual HistGBDT-RPM controller or a Balanced baseline."
    )

    parser.add_argument(
        "--workload",
        required=True,
        help="Path to benchmark wrapper/script/executable.",
    )
    parser.add_argument(
        "--workload-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments passed to the benchmark. Put this option last.",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Benchmark working directory. Default: workload's parent directory.",
    )
    parser.add_argument(
        "--controller",
        choices=["hgbdt", "balanced"],
        required=True,
        help="hgbdt = trained HistGBDT controller; balanced = fixed Medium baseline.",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=4,
        help="OMP/MKL/OpenBLAS/NumExpr thread count (default: 4).",
    )
    parser.add_argument(
        "--run-mode",
        choices=["completion", "duration"],
        default="completion",
        help="completion = fixed-work benchmark to process exit; duration = fixed wall-clock run.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Required only for --run-mode duration.",
    )
    parser.add_argument(
        "--sample-sleep",
        type=float,
        default=3.0,
        help="Sleep after each telemetry/control cycle. Existing pipeline uses ~3 s sleep.",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=3.0,
        help="Seconds to hold Medium before launching benchmark (default: 3).",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_FILE),
        help="Path to HistGBDT .joblib.",
    )
    parser.add_argument(
        "--scaler",
        default=str(DEFAULT_SCALER_FILE),
        help="Path to scaler_stats.npz.",
    )
    parser.add_argument(
        "--pcm",
        default=None,
        help="Optional explicit path to pcm.exe.",
    )
    parser.add_argument(
        "--pcm-power",
        default=None,
        help="Optional explicit path to pcm-power.exe.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: results/<controller>_<timestamp>.",
    )
    parser.add_argument(
        "--proba-hysteresis",
        action="store_true",
        help="Enable probability-margin hysteresis if model supports predict_proba.",
    )
    parser.add_argument(
        "--proba-margin",
        type=float,
        default=0.15,
        help="Minimum top1-top2 probability margin for a state switch.",
    )
    parser.add_argument(
        "--allow-simulated",
        action="store_true",
        help="DEBUG ONLY: allow simulated telemetry if PCM tools fail. Never use for paper results.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.cores < 1:
        raise SystemExit("--cores must be >= 1")

    if args.run_mode == "duration" and (args.duration is None or args.duration <= 0):
        raise SystemExit("--duration must be provided and > 0 in duration mode")

    workload = Path(args.workload).expanduser()
    if not workload.is_absolute():
        workload = (ROOT / workload).resolve()

    workdir = None
    if args.workdir:
        workdir = Path(args.workdir).expanduser()
        if not workdir.is_absolute():
            workdir = (ROOT / workdir).resolve()

    model_path = Path(args.model).expanduser().resolve()
    scaler_path = Path(args.scaler).expanduser().resolve() if args.scaler else None

    pcm_path = resolve_executable(args.pcm, PCM_CANDIDATES, "pcm.exe")
    pcm_power_path = resolve_executable(args.pcm_power, PCM_POWER_CANDIDATES, "pcm-power.exe")

    if pcm_path is None and not args.allow_simulated:
        raise SystemExit(
            "pcm.exe not found. Pass --pcm <path>. "
            "The experiment will not silently use simulated data."
        )

    if pcm_power_path is None and not args.allow_simulated:
        raise SystemExit(
            "pcm-power.exe not found. Pass --pcm-power <path>. "
            "The experiment will not silently use simulated data."
        )

    # Dummy names are only used if explicit debugging simulation is enabled.
    pcm_path = pcm_path or "pcm.exe"
    pcm_power_path = pcm_power_path or "pcm-power.exe"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = (ROOT / output_dir).resolve()
    else:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", workload.stem)
        output_dir = ROOT / "results" / f"{safe_name}_{args.controller}_{timestamp}"

    model = HistGBDTModel(
        model_path=model_path,
        scaler_path=scaler_path,
        enable_proba_hysteresis=args.proba_hysteresis,
        proba_margin=args.proba_margin,
    )

    runner = HistGBDTBenchmarkRunner(
        model=model,
        controller=args.controller,
        pcm_path=pcm_path,
        pcm_power_path=pcm_power_path,
        output_dir=output_dir,
        sample_sleep_s=args.sample_sleep,
        allow_simulated=args.allow_simulated,
    )

    try:
        runner.run(
            workload=workload,
            workload_args=args.workload_args,
            cores=args.cores,
            run_mode=args.run_mode,
            duration_s=args.duration,
            settle_s=args.settle,
            workdir=workdir,
        )
    except KeyboardInterrupt:
        print("Interrupted.")
        runner.stop_benchmark()
        raise
    except Exception as exc:
        runner.stop_benchmark()
        print(f"\nFATAL EXPERIMENT ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
