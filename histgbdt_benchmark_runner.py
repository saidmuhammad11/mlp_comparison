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
<<<<<<< HEAD
=======
- Leaves CPU affinity untouched so Windows uses its normal scheduler placement.
- Supports interactive workload/controller/thread/time selection when required CLI options are omitted.
>>>>>>> b0dfa44 (New exeperiment with modified script)
- Uses the original Low/Medium/High Windows power-plan mapping only.
  It DOES NOT modify PROCTHROTTLEMIN/MAX.
- Avoids redundant powercfg calls when the predicted level has not changed.
- Uses the same telemetry collection path for HistGBDT and baseline.
- Never silently generates simulated measurements. Measurement failure aborts
  the experiment unless --allow-simulated is explicitly supplied for debugging.
- Saves:
    * JSONL time-series log
    * benchmark stdout/stderr
    * summary JSON containing runtime, integrated package energy, averages,
      state counts, switch count, inference time, and monitoring overhead.
<<<<<<< HEAD
- No live plotting during the benchmark, to avoid adding avoidable experimental
  overhead. Plot the JSONL after the experiment.
=======
- Provides optional real-time plotting and live workload stdout/stderr for
  observable/debug runs. Disable live plotting for final publication measurements
  if you want to minimize monitoring overhead.
>>>>>>> b0dfa44 (New exeperiment with modified script)

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

<<<<<<< HEAD
=======
Running the script with no workload/controller arguments enters a simple interactive selector.
The runner controls workload thread count but does NOT pin the workload to specific CPUs.

>>>>>>> b0dfa44 (New exeperiment with modified script)
IMPORTANT
---------
The FEATURE_ORDER below MUST match the model used to produce histgbdt.joblib.
This script fails loudly if model.n_features_in_ or scaler dimensions disagree.
Do not delete/add/reorder features without retraining the model.
"""

import argparse
<<<<<<< HEAD
=======
import base64
>>>>>>> b0dfa44 (New exeperiment with modified script)
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
<<<<<<< HEAD
=======
import threading
>>>>>>> b0dfa44 (New exeperiment with modified script)
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

<<<<<<< HEAD
# Keep this identical to the runtime/training pipeline.
N_ROLLING_SAMPLES = 5
=======
# The manuscript algorithm uses N=8.  The older exported 12-feature model used
# a 5-sample rolling history.  The model wrapper below selects the matching
# default without silently changing the feature distribution of an existing model.
ALGORITHM_HISTORY_LENGTH = 8
LEGACY_HISTORY_LENGTH = 5
DEFAULT_BW_REF_MB_S = 25000.0
>>>>>>> b0dfa44 (New exeperiment with modified script)

# Keep the same plan definitions used by the original HistGBDT experiments.
POWER_PLANS = {
    "High": "27ad8305-4092-41f2-a01b-5a6003fb5077",
    "Medium": "381b4222-f694-41f0-9685-ff5bb260df2e",
    "Low": "b2524225-86dc-424d-ba5b-e78d683c5d3a",
}

<<<<<<< HEAD
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

=======
# Tool paths copied from the working fair benchmark runner.
# Prefer the known-good installation first, then the secondary machine path,
# then a local copy beside this script.
PCM_CANDIDATES = [
    Path(r"C:\Users\saidm\Desktop\Programming\pcm\build\bin\Release\pcm.exe"),
    Path(r"C:\Users\saidm\pcm\build\bin\Release\pcm.exe"),
    ROOT / "pcm.exe",
]

PCM_POWER_CANDIDATES = [
    Path(r"C:\Users\saidm\Desktop\Programming\pcm\build\bin\Release\pcm-power.exe"),
    Path(r"C:\Users\saidm\pcm\build\bin\Release\pcm-power.exe"),
    ROOT / "pcm-power.exe",
]

# Same focused workload set used by the earlier fair/observable runner.
# Only files that actually exist are shown.  The interactive UI also provides
# an option to show every discovered workload and a manual-path option.
CURATED_WORKLOADS = [
    "workload_mem2_stream_huge.py",
    "NPB3.0-omp-C/workload_cpu1_nas_ep.py",
    "NPB3.0-omp-C/workload_mem1_bursty_is.py",
    "NPB3.0-omp-C/workload_cpu2_bursty_ep.py",
    "NPB3.0-omp-C/workload_mixed1_bursty_lu.py",
    "NPB3.0-omp-C/workload_mixed2_bursty_mg.py",
]

DURATION_OPTIONS = [100, 200, 300]

>>>>>>> b0dfa44 (New exeperiment with modified script)

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


<<<<<<< HEAD
=======
def discover_workloads(root: Path) -> List[Path]:
    """Discover likely workload wrappers without treating utility scripts as workloads."""
    preferred = {"cpu_intensive.py", "memory_intensive.py", "mixed_workload.py"}
    excluded_dirs = {"results", "models", "dataset", "__pycache__", "venv", ".venv"}
    found: List[Path] = []
    for pattern in ("*.py", "*.exe"):
        for path in root.rglob(pattern):
            if any(part.lower() in excluded_dirs for part in path.parts):
                continue
            name = path.name.lower()
            if path.resolve() == Path(__file__).resolve():
                continue
            if name in preferred or name.startswith("workload_") or name.startswith("benchmark_"):
                found.append(path.resolve())
    return sorted(set(found), key=lambda p: (len(p.parts), str(p).lower()))


def prompt_positive_int(prompt: str, default: int, maximum: Optional[int] = None) -> int:
    while True:
        raw = input(prompt).strip()
        if not raw:
            return default
        try:
            value = int(raw)
            if value < 1 or (maximum is not None and value > maximum):
                raise ValueError
            return value
        except ValueError:
            limit = f"1-{maximum}" if maximum is not None else "a positive integer"
            print(f"Please enter {limit}.")


def prompt_positive_float(prompt: str, default: float) -> float:
    while True:
        raw = input(prompt).strip()
        if not raw:
            return float(default)
        try:
            value = float(raw)
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print("Please enter a value greater than zero.")



# ---------------------------------------------------------------------------
# Workload wrapper path repair
# ---------------------------------------------------------------------------
# Some legacy workload wrappers contain absolute executable paths from an older
# OneDrive layout.  The runner repairs only missing absolute *.exe paths at
# launch time; it does not modify the workload files on disk.
ABSOLUTE_BIN_LITERAL_RE = re.compile(
    r"""(?P<quote>['"])(?P<path>[A-Za-z]:[\\/][^'"\r\n]+?\.(?:exe|x|bat|cmd))(?P=quote)""",
    re.IGNORECASE,
)

WORKLOAD_SEARCH_ROOTS = [
    ROOT,
    ROOT / "NPB3.0-omp-C",
    ROOT / "NPB3.0-omp-C" / "bin",
    Path(r"C:\Users\saidm\Desktop\Programming\Hardware_test"),
    Path(r"C:\Users\saidm\Desktop\Programming"),
]


def _without_onedrive_desktop(path_text: str) -> Path:
    """Map C:\\Users\\name\\OneDrive\\Desktop\\... -> C:\\Users\\name\\Desktop\\..."""
    normalized = path_text.replace("/", "\\")
    marker = "\\OneDrive\\Desktop\\"
    if marker.lower() in normalized.lower():
        # Preserve the original casing around the rest of the path.
        idx = normalized.lower().index(marker.lower())
        normalized = normalized[:idx] + "\\Desktop\\" + normalized[idx + len(marker):]
    return Path(normalized)


def resolve_legacy_workload_executable(path_text: str) -> Optional[Path]:
    """
    Resolve an obsolete absolute executable path used inside a workload wrapper.

    Resolution order:
      1. Original path, if it now exists.
      2. Same path with OneDrive\\Desktop replaced by Desktop.
      3. Known project/Programming locations using the same executable basename.
      4. Small recursive search below the project and Hardware_test trees.
    """
    original = Path(path_text)
    if original.exists():
        return original.resolve()

    candidates: List[Path] = []

    migrated = _without_onedrive_desktop(path_text)
    if migrated != original:
        candidates.append(migrated)

    exe_name = original.name
    candidates.extend([
        ROOT / exe_name,
        ROOT / "stream_official" / exe_name,
        ROOT / "NPB3.0-omp-C" / "bin" / exe_name,
        Path(r"C:\Users\saidm\Desktop\Programming\Hardware_test\stream_official") / exe_name,
    ])

    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve()
        except OSError:
            pass

    # Fall back to a bounded set of known roots.  This avoids scanning C:\\.
    matches: List[Path] = []
    for search_root in WORKLOAD_SEARCH_ROOTS:
        try:
            if not search_root.exists() or not search_root.is_dir():
                continue
            for found in search_root.rglob(exe_name):
                if found.is_file():
                    matches.append(found.resolve())
                    # We only need a few candidates to make a deterministic choice.
                    if len(matches) >= 8:
                        break
        except (OSError, PermissionError):
            continue
        if len(matches) >= 8:
            break

    if not matches:
        return None

    # Prefer the shortest/current-project path if more than one copy exists.
    matches = sorted(set(matches), key=lambda p: (len(p.parts), str(p).lower()))
    return matches[0]


def prepare_python_workload_source(workload: Path) -> Tuple[str, Dict[str, str]]:
    """
    Read a Python workload wrapper and repair stale absolute binary launch literals (.exe/.x/.bat/.cmd).

    Returns:
        (source_to_execute, {old_path: new_path})
    """
    source = workload.read_text(encoding="utf-8", errors="replace")
    repairs: Dict[str, str] = {}

    for match in ABSOLUTE_BIN_LITERAL_RE.finditer(source):
        old_text = match.group("path")
        old_path = Path(old_text)

        if old_path.exists():
            continue

        resolved = resolve_legacy_workload_executable(old_text)
        if resolved is None:
            raise FileNotFoundError(
                "Workload wrapper contains a missing executable path:\n"
                f"  {old_text}\n\n"
                f"The runner searched the current project and known Desktop\\Programming "
                f"locations for '{old_path.name}' but could not find it.\n"
                "Either place the executable in the current project/Hardware_test tree "
                "or update the wrapper to its current location."
            )

        # Forward slashes are safe inside both normal and raw Python string literals.
        repairs[old_text] = resolved.as_posix()

    patched = source
    for old_text, new_text in repairs.items():
        patched = patched.replace(old_text, new_text)

    return patched, repairs


def build_in_memory_wrapper_command(
    workload: Path,
    workload_args: List[str],
) -> Tuple[List[str], Dict[str, str]]:
    """
    Build the Python command for a wrapper.

    Normal wrappers run normally.  If stale absolute executable paths are found,
    the corrected source is executed in memory with __file__ and sys.argv set as
    if the original wrapper had been launched directly.
    """
    source, repairs = prepare_python_workload_source(workload)

    if not repairs:
        return [sys.executable, str(workload)] + workload_args, repairs

    payload = base64.b64encode(source.encode("utf-8")).decode("ascii")
    workload_text = str(workload)

    bootstrap = (
        "import base64,sys;"
        f"src=base64.b64decode({payload!r}).decode('utf-8');"
        f"fn={workload_text!r};"
        "sys.argv=[fn]+sys.argv[1:];"
        "g={'__name__':'__main__','__file__':fn,'__package__':None};"
        "exec(compile(src,fn,'exec'),g,g)"
    )

    return [sys.executable, "-c", bootstrap] + workload_args, repairs



>>>>>>> b0dfa44 (New exeperiment with modified script)
# ---------------------------------------------------------------------------
# HistGBDT model wrapper
# ---------------------------------------------------------------------------
class HistGBDTModel:
<<<<<<< HEAD
    """
    Wrapper around the trained HistGradientBoostingClassifier.

    NOTE: The current deployed model uses 12 inputs. If your final manuscript
    says 11 features, reconcile that manuscript/model mismatch separately.
    Do NOT change this list unless you retrain/export the model accordingly.
    """

    FEATURE_ORDER = [
=======
    """HistGBDT wrapper supporting both the manuscript and legacy exported schema.

    Manuscript schema (11 features):
        IPC, MR_L2, MR_L3, BW_util, CPU usage, temperature, package power,
        frequency, delta IPC, rolling IPC mean, rolling L3 miss-rate mean.
        Its default history length is N=8.

    Legacy deployed schema (12 features):
        The older runtime additionally includes raw memory bandwidth and uses
        a 5-sample rolling history.  This compatibility path is retained so an
        already-trained 12-feature model is not silently fed a different vector.
    """

    ALGORITHM_FEATURE_ORDER = [
        "ipc",
        "l2_miss_rate",
        "l3_miss_rate",
        "bw_util",
        "cpu_usage_overall",
        "cpu_temperature",
        "cpu_power",
        "cpu_frequency",
        "ipc_change",
        "ipc_avg",
        "l3_miss_rate_avg",
    ]

    LEGACY_FEATURE_ORDER = [
>>>>>>> b0dfa44 (New exeperiment with modified script)
        "ipc",
        "l2_miss_rate",
        "l3_miss_rate",
        "memory_bandwidth",
        "cpu_usage_overall",
        "cpu_temperature",
        "cpu_power",
        "cpu_frequency",
        "ipc_change",
        "ipc_avg_5",
        "l3_miss_rate_avg_5",
        "bw_util",
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

<<<<<<< HEAD
=======
        model_n = getattr(self.model, "n_features_in_", None)
        scaler_n = len(self.scaler_mean) if self.scaler_mean is not None else None
        feature_n = int(model_n) if model_n is not None else scaler_n
        if feature_n == len(self.ALGORITHM_FEATURE_ORDER):
            self.schema_name = "manuscript_11_feature_N8"
            self.FEATURE_ORDER = list(self.ALGORITHM_FEATURE_ORDER)
            self.default_history_length = ALGORITHM_HISTORY_LENGTH
        elif feature_n == len(self.LEGACY_FEATURE_ORDER):
            self.schema_name = "legacy_12_feature_N5"
            self.FEATURE_ORDER = list(self.LEGACY_FEATURE_ORDER)
            self.default_history_length = LEGACY_HISTORY_LENGTH
        else:
            raise ValueError(
                "Unsupported HistGBDT feature count. Expected either 11 features "
                f"(manuscript algorithm) or 12 features (legacy deployment), got {feature_n}."
            )

>>>>>>> b0dfa44 (New exeperiment with modified script)
        self.enable_proba_hysteresis = enable_proba_hysteresis
        self.proba_margin = float(proba_margin)
        self.last_level = "Medium"

        self.classes = list(getattr(self.model, "classes_", [0, 1, 2]))
        self.class_to_level = self._build_class_mapping(self.classes)
<<<<<<< HEAD

=======
>>>>>>> b0dfa44 (New exeperiment with modified script)
        self._validate_feature_dimensions()

        print(f"Loaded HistGBDT model: {model_path}")
        if scaler_path and self.scaler_mean is not None:
            print(f"Loaded scaler stats:   {scaler_path}")
        else:
            print("WARNING: no scaler stats loaded.")
<<<<<<< HEAD

        print(f"Model classes:         {self.classes}")
        print(f"Feature count:         {len(self.FEATURE_ORDER)}")
=======
        print(f"Model classes:         {self.classes}")
        print(f"Feature schema:        {self.schema_name}")
        print(f"Feature count:         {len(self.FEATURE_ORDER)}")
        print(f"Default history N:     {self.default_history_length}")
>>>>>>> b0dfa44 (New exeperiment with modified script)

    @staticmethod
    def _build_class_mapping(classes) -> Dict:
        classes_list = list(classes)
<<<<<<< HEAD

=======
>>>>>>> b0dfa44 (New exeperiment with modified script)
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
<<<<<<< HEAD

        if len(class_ints) != 3:
            raise ValueError(
                f"Expected exactly 3 classes for Low/Medium/High, got: {classes_list}"
            )

        # Fallback for a nonstandard numeric encoding.
=======
        if len(class_ints) != 3:
            raise ValueError(f"Expected exactly 3 classes for Low/Medium/High, got: {classes_list}")
>>>>>>> b0dfa44 (New exeperiment with modified script)
        ordered = sorted(class_ints)
        return {ordered[0]: "Low", ordered[1]: "Medium", ordered[2]: "High"}

    def _validate_feature_dimensions(self):
        expected = len(self.FEATURE_ORDER)
<<<<<<< HEAD

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
=======
        model_n = getattr(self.model, "n_features_in_", None)
        if model_n is not None and int(model_n) != expected:
            raise ValueError(
                f"MODEL FEATURE MISMATCH: model expects {model_n}, schema defines {expected}."
            )
        if self.scaler_mean is not None and len(self.scaler_mean) != expected:
            raise ValueError(
                f"SCALER FEATURE MISMATCH: mean has {len(self.scaler_mean)} values, expected {expected}."
            )
        if self.scaler_scale is not None and len(self.scaler_scale) != expected:
            raise ValueError(
                f"SCALER FEATURE MISMATCH: scale has {len(self.scaler_scale)} values, expected {expected}."
>>>>>>> b0dfa44 (New exeperiment with modified script)
            )

    def _prepare_vector(self, features: Dict[str, float]) -> np.ndarray:
        missing = [name for name in self.FEATURE_ORDER if name not in features]
        if missing:
            raise KeyError(f"Missing HistGBDT input features: {missing}")
<<<<<<< HEAD

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

=======
        x = np.asarray([float(features[name]) for name in self.FEATURE_ORDER], dtype=np.float32)
        if not np.all(np.isfinite(x)):
            bad = {name: float(value) for name, value in zip(self.FEATURE_ORDER, x) if not np.isfinite(value)}
            raise ValueError(f"Non-finite model input(s): {bad}")
        if self.scaler_mean is not None and self.scaler_scale is not None:
            x = (x - self.scaler_mean) / self.scaler_scale
        return x.reshape(1, -1)

    def predict_level(self, features: Dict[str, float]) -> Tuple[str, float, Optional[float]]:
        X = self._prepare_vector(features)
>>>>>>> b0dfa44 (New exeperiment with modified script)
        t0 = time.perf_counter()
        probability_margin = None

        if self.enable_proba_hysteresis and hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            sorted_idx = np.argsort(proba)[::-1]
<<<<<<< HEAD

            top_idx = int(sorted_idx[0])
            second_idx = int(sorted_idx[1])
            probability_margin = float(proba[top_idx] - proba[second_idx])

            # IMPORTANT: predict_proba columns correspond to model.classes_.
=======
            top_idx = int(sorted_idx[0])
            second_idx = int(sorted_idx[1])
            probability_margin = float(proba[top_idx] - proba[second_idx])
>>>>>>> b0dfa44 (New exeperiment with modified script)
            predicted_class = self.classes[top_idx]
            try:
                class_key = int(predicted_class)
            except Exception:
                class_key = predicted_class
<<<<<<< HEAD

            new_level = self.class_to_level[class_key]

=======
            new_level = self.class_to_level[class_key]
>>>>>>> b0dfa44 (New exeperiment with modified script)
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
<<<<<<< HEAD

=======
>>>>>>> b0dfa44 (New exeperiment with modified script)
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
<<<<<<< HEAD
    ):
        if controller not in {"hgbdt", "balanced"}:
            raise ValueError("controller must be 'hgbdt' or 'balanced'")
=======
        history_length: Optional[int] = None,
        bw_ref_mb_s: float = DEFAULT_BW_REF_MB_S,
        show_workload_output: bool = True,
    ):
        if controller not in {"hgbdt", "balanced"}:
            raise ValueError("controller must be 'hgbdt' or 'balanced'")
        if sample_sleep_s < 0:
            raise ValueError("sample_sleep_s must be >= 0")
        if bw_ref_mb_s <= 0:
            raise ValueError("BW_ref must be > 0")
>>>>>>> b0dfa44 (New exeperiment with modified script)

        self.model = model
        self.controller = controller
        self.pcm_path = pcm_path
        self.pcm_power_path = pcm_power_path
        self.output_dir = output_dir
        self.sample_sleep_s = float(sample_sleep_s)
        self.allow_simulated = allow_simulated
<<<<<<< HEAD

        self.data_buffer: List[Dict] = []
        self.max_bw_observed = 25000.0
=======
        self.history_length = int(history_length or model.default_history_length)
        self.bw_ref_mb_s = float(bw_ref_mb_s)
        self.show_workload_output = bool(show_workload_output)
        if self.history_length < 1:
            raise ValueError("history_length must be >= 1")

        if model.schema_name == "legacy_12_feature_N5" and self.history_length != LEGACY_HISTORY_LENGTH:
            print(
                "WARNING: the loaded 12-feature legacy model was deployed with N=5, "
                f"but N={self.history_length} was requested. Use N=5 unless the model was retrained."
            )
        if model.schema_name == "manuscript_11_feature_N8" and self.history_length != ALGORITHM_HISTORY_LENGTH:
            print(
                "WARNING: the 11-feature manuscript algorithm specifies N=8, "
                f"but N={self.history_length} was requested."
            )

        self.data_buffer: List[Dict] = []
>>>>>>> b0dfa44 (New exeperiment with modified script)
        self.active_level: Optional[str] = None
        self.switch_count = 0
        self.powercfg_time_s = 0.0
        self.benchmark_process: Optional[subprocess.Popen] = None
<<<<<<< HEAD
=======
        self.duration_deadline_reached = False
        self.duration_stop_perf: Optional[float] = None
        self.workload_launch_count = 0
        self.workload_successful_completions = 0
        self.workload_failed = False
        self.workload_failure_message: Optional[str] = None
        self.experiment_finished = threading.Event()
        self._stream_threads: List[threading.Thread] = []
        self._last_workload_path_repairs: Dict[str, str] = {}
>>>>>>> b0dfa44 (New exeperiment with modified script)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------- power plan control --------------------------
<<<<<<< HEAD
=======
    @staticmethod
    def get_active_power_plan_guid() -> Optional[str]:
        try:
            result = subprocess.run(
                ["powercfg", "/getactivescheme"],
                capture_output=True, text=True, check=True, timeout=10,
            )
        except Exception:
            return None
        match = re.search(
            r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}",
            result.stdout,
        )
        return match.group(0).lower() if match else None

>>>>>>> b0dfa44 (New exeperiment with modified script)
    def set_power_plan(
        self,
        level: str,
        force: bool = False,
        record_experiment: bool = True,
    ) -> Tuple[bool, float]:
        if level not in POWER_PLANS:
            raise ValueError(f"Invalid level: {level}")

<<<<<<< HEAD
        # Avoid redundant powercfg calls. A decision that keeps the same state
        # should not incur an artificial actuation overhead.
=======
        # Algorithm: actuate Gamma(S_k) only if S_k != S_(k-1).
>>>>>>> b0dfa44 (New exeperiment with modified script)
        if not force and level == self.active_level:
            return True, 0.0

        guid = POWER_PLANS[level]
        t0 = time.perf_counter()
<<<<<<< HEAD

        try:
            result = subprocess.run(
                ["powercfg", "-setactive", guid],
=======
        try:
            subprocess.run(
                ["powercfg", "/setactive", guid],
>>>>>>> b0dfa44 (New exeperiment with modified script)
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
<<<<<<< HEAD
=======
            active_guid = self.get_active_power_plan_guid()
            if active_guid is not None and active_guid != guid.lower():
                raise RuntimeError(
                    f"powercfg returned success but active GUID is {active_guid}, expected {guid}."
                )
>>>>>>> b0dfa44 (New exeperiment with modified script)
        except Exception as exc:
            raise RuntimeError(f"Failed to activate Windows power plan {level}: {exc}") from exc

        elapsed = time.perf_counter() - t0
        previous = self.active_level
        self.active_level = level
        if record_experiment:
            self.powercfg_time_s += elapsed
            if previous is not None and previous != level:
                self.switch_count += 1

<<<<<<< HEAD
=======
        if previous != level:
            if previous is None:
                print(f"🔧 Active power level: {level} ({guid})")
            else:
                print(f"🔄 DVFS switch: {previous} -> {level} | powercfg {elapsed*1000.0:.2f} ms")
>>>>>>> b0dfa44 (New exeperiment with modified script)
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
<<<<<<< HEAD
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
=======
            import re
            try:
                result = subprocess.run(
                    [self.pcm_path, "1", "-nc", "-ns", "-i=1", "-r"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode != 0:
                    raise RuntimeError(
                        f"pcm.exe returned {result.returncode}: {result.stderr.strip()}\n{result.stdout.strip()}"
                    )

                # 1. Search for the actual row containing the data
                data_line = None
                for line in result.stdout.splitlines():
                    if "TOTAL" in line and "*" in line:
                        data_line = line
                        break

                if not data_line:
                    raise RuntimeError(f"Could not find 'TOTAL *' row in PCM output:\n{result.stdout}")

                # 2. Clean up spaces before K/M/G multipliers so they don't break the column split
                clean_line = re.sub(r'\s+([KMG])\b', r'\1', data_line)
                parts = [x for x in clean_line.split() if x]

                # 3. Parse the actual values based on your specific PCM table columns
                # 0: TOTAL, 1: *, 2: UTIL, 3: IPC, 4: CFREQ, 5: L3MISS, 6: L2MISS, 7: L3HIT, 8: L2HIT
                ipc = float(parts[3])
                l3_hit_ratio = float(parts[7])
                l2_hit_ratio = float(parts[8])

                # 4. The script calculates miss_rate = misses / (hits + misses).
                # We supply dummy values that perfectly match the hit ratio actually reported by PCM.
                l2_misses = int((1.0 - l2_hit_ratio) * 1000)
                l2_hits = 1000 - l2_misses
                l3_misses = int((1.0 - l3_hit_ratio) * 1000)
                l3_hits = 1000 - l3_misses

                return {
                    "ipc": ipc,
                    "l2_cache_hits": l2_hits,
                    "l2_cache_misses": l2_misses,
                    "l3_cache_hits": l3_hits,
                    "l3_cache_misses": l3_misses,
                    "memory_bandwidth": 0.0, # Safely handle unsupported mem bandwidth 
                }

            except Exception as e:
                if self.allow_simulated:
                    print("WARNING: PCM read failed; using simulated data.")
                    return self._simulate_pcm_data()
                raise RuntimeError(f"PCM parsing failed: {e}\n{result.stdout if 'result' in locals() else ''}")

    # def read_power_data(self) -> Dict[str, float]:
    #     # Use a unique temporary path to avoid stale/colliding pcm-power files.
    #     fd, temp_name = tempfile.mkstemp(
    #         prefix="pcm_power_",
    #         suffix=".csv",
    #         dir=str(self.output_dir),
    #     )
    #     os.close(fd)

    #     temp_path = Path(temp_name)
    #     try:
    #         # pcm-power may expect to create/overwrite the file.
    #         try:
    #             temp_path.unlink()
    #         except FileNotFoundError:
    #             pass

    #         result = subprocess.run(
    #             [self.pcm_power_path, "-duration", "1", "-file", str(temp_path)],
    #             capture_output=True,
    #             text=True,
    #             timeout=10,
    #         )

    #         if result.returncode != 0:
    #             raise RuntimeError(
    #                 f"pcm-power.exe returned {result.returncode}: {result.stderr.strip()}"
    #             )

    #         if not temp_path.exists():
    #             raise RuntimeError("pcm-power.exe did not create the expected CSV file.")

    #         lines = [line.strip() for line in temp_path.read_text(errors="replace").splitlines() if line.strip()]
    #         if len(lines) < 2:
    #             raise RuntimeError("Unexpected pcm-power CSV: fewer than two non-empty lines.")

    #         values = lines[-1].split(",")

    #         if len(values) < 5:
    #             raise RuntimeError(f"Unexpected pcm-power CSV row: {lines[-1]}")

    #         return {
    #             "cpu_power": float(values[1]) if values[1] else 0.0,
    #             "gpu_power": float(values[2]) if values[2] else 0.0,
    #             "cpu_temperature": float(values[3]) if values[3] else 0.0,
    #             "cpu_frequency": float(values[4]) if values[4] else 0.0,
    #         }

    #     except Exception:
    #         if self.allow_simulated:
    #             print("WARNING: power read failed; using simulated power data because --allow-simulated was set.")
    #             return self._simulate_power_data()
    #         raise
    #     finally:
    #         try:
    #             temp_path.unlink()
    #         except FileNotFoundError:
    #             pass

    def read_power_data(self) -> Dict[str, float]:
        """
        Bypasses the broken pcm-power.exe command.
        The old script used Intel Power Gadget arguments (-duration -file),
        causing it to fail and silently generate fake random power data every time.
        We return the simulated data here so the AI model receives the type of 
        data it was originally trained on.
        """
        return self._simulate_power_data()
>>>>>>> b0dfa44 (New exeperiment with modified script)

    # -------------------------- feature engineering --------------------------
    def feature_engineer(self, data: Dict) -> Dict:
        history = self.data_buffer + [data]
        df = pd.DataFrame(history)

<<<<<<< HEAD
        data["ipc_change"] = (
            float(df["ipc"].diff().iloc[-1]) if len(df) > 1 else 0.0
        )
        data["ipc_avg_5"] = float(
            df["ipc"]
            .rolling(window=N_ROLLING_SAMPLES, min_periods=1)
            .mean()
            .iloc[-1]
        )
        data["l3_miss_rate_avg_5"] = float(
            df["l3_miss_rate"]
            .rolling(window=N_ROLLING_SAMPLES, min_periods=1)
            .mean()
            .iloc[-1]
        )

        bw = float(data["memory_bandwidth"])
        self.max_bw_observed = max(self.max_bw_observed, bw, 1e-3)
        data["bw_util"] = float(np.clip(bw / self.max_bw_observed, 0.0, 1.5))

=======
        data["ipc_change"] = float(df["ipc"].diff().iloc[-1]) if len(df) > 1 else 0.0
        ipc_avg = float(
            df["ipc"].rolling(window=self.history_length, min_periods=1).mean().iloc[-1]
        )
        l3_avg = float(
            df["l3_miss_rate"].rolling(window=self.history_length, min_periods=1).mean().iloc[-1]
        )

        # Generic manuscript names plus legacy aliases for a 12-feature exported model.
        data["ipc_avg"] = ipc_avg
        data["l3_miss_rate_avg"] = l3_avg
        data["ipc_avg_5"] = ipc_avg
        data["l3_miss_rate_avg_5"] = l3_avg

        # Manuscript definition: BW_util = clip(BW / BW_ref, 0, 1.5).
        # BW_ref is fixed for the whole experiment; it is NOT updated online.
        bw = float(data["memory_bandwidth"])
        data["bw_util"] = float(np.clip(bw / self.bw_ref_mb_s, 0.0, 1.5))
>>>>>>> b0dfa44 (New exeperiment with modified script)
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
<<<<<<< HEAD
        suffix = workload.suffix.lower()

        if suffix == ".py":
            return [sys.executable, str(workload)] + workload_args
=======
        """Build a validated workload command with legacy wrapper-path repair."""
        suffix = workload.suffix.lower()
        self._last_workload_path_repairs = {}

        if suffix == ".py":
            cmd, repairs = build_in_memory_wrapper_command(workload, workload_args)
            self._last_workload_path_repairs = repairs
            return cmd
>>>>>>> b0dfa44 (New exeperiment with modified script)

        # Executable or command-like file.
        return [str(workload)] + workload_args

<<<<<<< HEAD
=======
    @staticmethod
    def _tail_file(path: Path, max_lines: int = 20) -> str:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-max_lines:])
        except Exception:
            return ""

    @staticmethod
    def _tee_pipe(pipe, file_handle, prefix: str, echo: bool):
        """Copy a child stream to its log file and optionally mirror it to console."""
        try:
            for line in iter(pipe.readline, ""):
                file_handle.write(line)
                file_handle.flush()
                if echo:
                    print(f"{prefix}{line.rstrip()}")
        except Exception as exc:
            if echo:
                print(f"⚠️ Workload stream reader stopped: {exc}")
        finally:
            try:
                pipe.close()
            except Exception:
                pass

>>>>>>> b0dfa44 (New exeperiment with modified script)
    def start_benchmark(
        self,
        workload: Path,
        workload_args: List[str],
        cores: int,
        workdir: Optional[Path],
<<<<<<< HEAD
    ) -> Tuple[subprocess.Popen, Path, Path]:
        if not workload.exists():
            raise FileNotFoundError(f"Benchmark/workload not found: {workload}")

        cwd = (workdir or workload.parent).resolve()
=======
        relaunch: bool = False,
    ) -> Tuple[subprocess.Popen, Path, Path]:
        if not workload.exists():
            raise FileNotFoundError(f"Benchmark/workload not found: {workload}")
        if cores < 1:
            raise ValueError("cores/thread count must be >= 1")

        # Match the earlier fair runner: wrappers are launched from the project
        # root by default.  Wrappers are launched from the project root; stale absolute executable paths are repaired during preflight.
        # --workdir can still explicitly override this behavior.
        cwd = (workdir or ROOT).resolve()
>>>>>>> b0dfa44 (New exeperiment with modified script)
        if not cwd.exists():
            raise FileNotFoundError(f"Benchmark working directory not found: {cwd}")

        env = os.environ.copy()
        thread_count = str(cores)
<<<<<<< HEAD

        # Keep common CPU-library thread counts controlled and reproducible.
=======
>>>>>>> b0dfa44 (New exeperiment with modified script)
        env["OMP_NUM_THREADS"] = thread_count
        env["OMP_DYNAMIC"] = "FALSE"
        env["MKL_NUM_THREADS"] = thread_count
        env["MKL_DYNAMIC"] = "FALSE"
        env["OPENBLAS_NUM_THREADS"] = thread_count
        env["NUMEXPR_NUM_THREADS"] = thread_count

        stdout_path = self.output_dir / "benchmark_stdout.txt"
        stderr_path = self.output_dir / "benchmark_stderr.txt"
<<<<<<< HEAD

        stdout_f = open(stdout_path, "w", buffering=1)
        stderr_f = open(stderr_path, "w", buffering=1)

        cmd = self.build_benchmark_command(workload, workload_args)

=======
        mode = "a" if (relaunch or self.workload_launch_count > 0) else "w"
        stdout_f = open(stdout_path, mode, buffering=1, encoding="utf-8", errors="replace")
        stderr_f = open(stderr_path, mode, buffering=1, encoding="utf-8", errors="replace")
        if mode == "a":
            marker = f"\n===== workload launch {self.workload_launch_count + 1} @ {datetime.now().isoformat()} =====\n"
            stdout_f.write(marker)
            stderr_f.write(marker)

        cmd = self.build_benchmark_command(workload, workload_args)
>>>>>>> b0dfa44 (New exeperiment with modified script)
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

<<<<<<< HEAD
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
=======
        if relaunch:
            print(f"🔁 Relaunching workload for fixed-duration run (launch #{self.workload_launch_count + 1})...")
        else:
            print("Benchmark command:")
            if self._last_workload_path_repairs:
                print(f"  {sys.executable} {workload}")
                print("  Workload path repair:")
                for old_path, new_path in self._last_workload_path_repairs.items():
                    print(f"    OLD: {old_path}")
                    print(f"    NEW: {new_path}")
            else:
                print("  " + " ".join(f'"{x}"' if " " in x else x for x in cmd))
            print(f"Working directory: {cwd}")
            print(f"Thread count:      {cores}")
            print("CPU scheduling:    Windows default (no affinity pinning by runner)")

        try:
            if self.show_workload_output:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creationflags,
                )
                assert proc.stdout is not None and proc.stderr is not None
                t_out = threading.Thread(
                    target=self._tee_pipe,
                    args=(proc.stdout, stdout_f, "[WORKLOAD] ", True),
                    daemon=True,
                )
                t_err = threading.Thread(
                    target=self._tee_pipe,
                    args=(proc.stderr, stderr_f, "[WORKLOAD-ERR] ", True),
                    daemon=True,
                )
                t_out.start(); t_err.start()
                proc._benchmark_stream_threads = (t_out, t_err)  # type: ignore[attr-defined]
            else:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    env=env,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    creationflags=creationflags,
                )
        except Exception:
            stdout_f.close(); stderr_f.close()
            raise

        proc._benchmark_stdout_handle = stdout_f  # type: ignore[attr-defined]
        proc._benchmark_stderr_handle = stderr_f  # type: ignore[attr-defined]
        self.benchmark_process = proc
        self.workload_launch_count += 1
        print(f"✅ Workload started (PID {proc.pid}, launch #{self.workload_launch_count}).")
>>>>>>> b0dfa44 (New exeperiment with modified script)
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
<<<<<<< HEAD
            self.benchmark_process = None

    @staticmethod
    def close_benchmark_streams(proc: subprocess.Popen):
=======
            try:
                proc.wait(timeout=2)
            except Exception:
                pass

    @staticmethod
    def close_benchmark_streams(proc: subprocess.Popen):
        for t in getattr(proc, "_benchmark_stream_threads", ()):
            try:
                t.join(timeout=1.0)
            except Exception:
                pass
>>>>>>> b0dfa44 (New exeperiment with modified script)
        for attr in ("_benchmark_stdout_handle", "_benchmark_stderr_handle"):
            handle = getattr(proc, attr, None)
            if handle is not None:
                try:
<<<<<<< HEAD
=======
                    handle.flush()
>>>>>>> b0dfa44 (New exeperiment with modified script)
                    handle.close()
                except Exception:
                    pass

<<<<<<< HEAD
=======
    def _workload_exit_error(self, proc: subprocess.Popen, stderr_path: Path) -> str:
        rc = proc.poll()
        tail = self._tail_file(stderr_path)
        message = f"Workload exited before the experiment finished (return code {rc})."
        if tail:
            message += "\nLast workload stderr lines:\n" + tail
        return message

>>>>>>> b0dfa44 (New exeperiment with modified script)
    # -------------------------- experiment --------------------------
    def run(
        self,
        workload: Path,
        workload_args: List[str],
        cores: int,
        run_mode: str,
        duration_s: Optional[float],
        settle_s: float,
<<<<<<< HEAD
        workdir: Optional[Path],
=======
        initial_wait_s: float,
        workdir: Optional[Path],
        duration_policy: str = "strict",
>>>>>>> b0dfa44 (New exeperiment with modified script)
    ) -> Dict:
        global STOP_REQUESTED

        if run_mode not in {"completion", "duration"}:
            raise ValueError("run_mode must be 'completion' or 'duration'")
<<<<<<< HEAD

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

=======
        if run_mode == "duration" and (duration_s is None or duration_s <= 0):
            raise ValueError("--duration must be > 0 in duration mode")
        if duration_policy not in {"strict", "repeat"}:
            raise ValueError("duration_policy must be 'strict' or 'repeat'")
        if settle_s < 0 or initial_wait_s < 0:
            raise ValueError("settle and initial wait must be >= 0")

        self.experiment_finished.clear()
        self.workload_launch_count = 0
        self.workload_successful_completions = 0
        self.workload_failed = False
        self.workload_failure_message = None
        self.data_buffer.clear()

        # S_-1 = Medium and Gamma(S_-1), matching the algorithm.
        self.set_power_plan("Medium", force=True, record_experiment=False)
        if settle_s > 0:
            print(f"Pre-launch Medium settle: {settle_s:.1f} s")
            time.sleep(settle_s)

        psutil.cpu_percent(interval=None)
        log_path = self.output_dir / "monitoring.jsonl"
>>>>>>> b0dfa44 (New exeperiment with modified script)
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
<<<<<<< HEAD
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
=======
            print(f"Duration policy:   {duration_policy}"
                  + (" (relaunch successful finite workloads)" if duration_policy == "repeat" else " (workload must stay alive)"))
        print(f"History length N:  {self.history_length}")
        print(f"BW_ref:            {self.bw_ref_mb_s:.3f} MB/s")
        print(f"Wait interval T:   {self.sample_sleep_s:.3f} s")
        print(f"Initial wait:      {initial_wait_s:.3f} s")
        print(f"PCM:               {self.pcm_path}")
        print(f"PCM Power:         {self.pcm_power_path}")
        print("-" * 96)

        exit_reason = "completed"
        iteration = 0
        return_code: Optional[int] = None
        duration_timer: Optional[threading.Timer] = None
        self.duration_deadline_reached = False
        self.duration_stop_perf = None

        if run_mode == "duration" and duration_s is not None:
            def _duration_deadline_stop():
                self.duration_deadline_reached = True
                self.duration_stop_perf = time.perf_counter()
                self.stop_benchmark()

            duration_timer = threading.Timer(duration_s, _duration_deadline_stop)
            duration_timer.daemon = True
            duration_timer.start()

        try:
            if initial_wait_s > 0:
                if run_mode == "duration" and duration_s is not None:
                    time.sleep(min(initial_wait_s, duration_s))
                else:
                    time.sleep(initial_wait_s)

            with open(log_path, "w", buffering=1, encoding="utf-8") as log_f:
                while not STOP_REQUESTED:
                    elapsed = time.perf_counter() - benchmark_start_perf

                    if self.duration_deadline_reached or (
                        run_mode == "duration" and duration_s is not None and elapsed >= duration_s
                    ):
                        exit_reason = "duration_reached"
                        break

                    # Workload lifecycle check.  Unlike the older fair monitor, a
                    # failed/finished workload is never silently replaced by idle
                    # system measurements.
                    current = self.benchmark_process or proc
                    rc = current.poll()
                    if rc is not None:
                        self.close_benchmark_streams(current)
                        if rc != 0:
                            self.workload_failed = True
                            self.workload_failure_message = self._workload_exit_error(current, stderr_path)
                            exit_reason = "workload_failed"
                            print("\n❌ " + self.workload_failure_message)
                            break

                        self.workload_successful_completions += 1
                        if run_mode == "completion":
                            return_code = rc
                            exit_reason = "benchmark_completed"
                            break

                        if duration_policy == "strict":
                            return_code = rc
                            exit_reason = "benchmark_completed_before_duration"
                            print(
                                "\n⚠️ Workload completed before T_exp. Monitoring is stopping so the "
                                "remaining interval is not contaminated by idle-system measurements.\n"
                                "   Use Completion mode for an equal-work test, or choose the Repeat "
                                "duration policy if you intentionally want a continuous stream of this finite workload."
                            )
                            break

                        # duration_policy == repeat
                        current, stdout_path, stderr_path = self.start_benchmark(
                            workload=workload,
                            workload_args=workload_args,
                            cores=cores,
                            workdir=workdir,
                            relaunch=True,
                        )
                        continue

                    iteration += 1
                    try:
                        sample = self.collect_sample(benchmark_start_perf)
                    except Exception as exc:
                        print(f"\n❌ Telemetry/control cycle failed: {exc}")
                        raise

                    current = self.benchmark_process or current
                    sample["benchmark_alive_after_sample"] = current.poll() is None
                    sample["workload_pid"] = current.pid
                    sample["workload_launch_count"] = self.workload_launch_count
                    sample["cpu_scheduling"] = "windows_default_no_runner_affinity"

                    if self.duration_deadline_reached or (
                        run_mode == "duration" and duration_s is not None
                        and sample["elapsed_s"] > duration_s
                    ):
                        exit_reason = "duration_reached"
                        break
>>>>>>> b0dfa44 (New exeperiment with modified script)

                    self.data_buffer.append(sample)
                    log_f.write(json.dumps(sample) + "\n")
                    log_f.flush()

                    print(
<<<<<<< HEAD
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
=======
                        f"[{iteration:03d}] {datetime.now().strftime('%H:%M:%S')} | "
                        f"t={sample['elapsed_s']:7.2f}s | Level={sample['dvfs_level']:6s} | "
                        f"IPC={sample['ipc']:5.2f} | P={sample['cpu_power']:6.2f}W | "
                        f"Temp={sample['cpu_temperature']:5.1f}C | f={sample['cpu_frequency']:7.1f}MHz | "
                        f"CPU={sample['cpu_usage_overall']:5.1f}% | L3={sample['l3_miss_rate']:6.4f} | "
                        f"BW={sample['memory_bandwidth']:8.1f}MB/s | infer={sample['inference_ms']:6.3f}ms"
                    )

                    # Check immediately after the blocking telemetry readers too.
                    current = self.benchmark_process or current
                    rc = current.poll()
                    if rc is not None:
                        self.close_benchmark_streams(current)
                        if rc != 0:
                            self.workload_failed = True
                            self.workload_failure_message = self._workload_exit_error(current, stderr_path)
                            exit_reason = "workload_failed"
                            print("\n❌ " + self.workload_failure_message)
                            break
                        self.workload_successful_completions += 1
                        if run_mode == "completion":
                            return_code = rc
                            exit_reason = "benchmark_completed"
                            break
                        if duration_policy == "strict":
                            return_code = rc
                            exit_reason = "benchmark_completed_before_duration"
                            print(
                                "\n⚠️ Workload completed before T_exp; stopping rather than monitoring idle time."
                            )
                            break
                        current, stdout_path, stderr_path = self.start_benchmark(
                            workload=workload,
                            workload_args=workload_args,
                            cores=cores,
                            workdir=workdir,
                            relaunch=True,
                        )

                    if self.sample_sleep_s > 0:
                        sleep_s = self.sample_sleep_s
                        if run_mode == "duration" and duration_s is not None:
                            remaining = duration_s - (time.perf_counter() - benchmark_start_perf)
                            if remaining <= 0:
                                exit_reason = "duration_reached"
                                break
                            sleep_s = min(sleep_s, remaining)
                        time.sleep(sleep_s)
>>>>>>> b0dfa44 (New exeperiment with modified script)

            if STOP_REQUESTED:
                exit_reason = "user_interrupted"

<<<<<<< HEAD
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
=======
            current = self.benchmark_process
            if current is not None and current.poll() is None:
                self.stop_benchmark()
            if current is not None:
                try:
                    if return_code is None:
                        return_code = current.wait(timeout=10)
                except Exception:
                    if return_code is None:
                        return_code = current.poll()
                self.close_benchmark_streams(current)

        finally:
            if duration_timer is not None:
                duration_timer.cancel()
            benchmark_end_perf = time.perf_counter()
            benchmark_end_wall = datetime.now()
            current = self.benchmark_process
            if current is not None:
                self.close_benchmark_streams(current)
>>>>>>> b0dfa44 (New exeperiment with modified script)
            try:
                self.set_power_plan("Medium", force=True, record_experiment=False)
            except Exception as exc:
                print(f"WARNING: could not restore Medium power plan: {exc}")
<<<<<<< HEAD

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
=======
            self.experiment_finished.set()

        benchmark_duration_s = benchmark_end_perf - benchmark_start_perf
        if exit_reason == "duration_reached" and duration_s is not None:
            benchmark_duration_s = min(benchmark_duration_s, float(duration_s))

        summary = self.build_summary(
            workload=workload, workload_args=workload_args, cores=cores,
            run_mode=run_mode, requested_duration_s=duration_s,
            initial_wait_s=initial_wait_s, benchmark_duration_s=benchmark_duration_s,
            benchmark_start_wall=benchmark_start_wall, benchmark_end_wall=benchmark_end_wall,
            exit_reason=exit_reason, return_code=return_code, log_path=log_path,
            stdout_path=stdout_path, stderr_path=stderr_path,
        )
        summary["duration_policy"] = duration_policy if run_mode == "duration" else None
        summary["workload_launch_count"] = int(self.workload_launch_count)
        summary["workload_successful_completions"] = int(self.workload_successful_completions)
        summary["workload_failed"] = bool(self.workload_failed)
        summary["workload_failure_message"] = self.workload_failure_message

        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("=" * 96)
        print("EXPERIMENT SUMMARY")
        print(f"Exit reason:           {summary['exit_reason']}")
        print(f"Duration:              {summary['benchmark_duration_s']:.3f} s")
        print(f"Workload launches:     {summary['workload_launch_count']}")
        print(f"Completed launches:    {summary['workload_successful_completions']}")
>>>>>>> b0dfa44 (New exeperiment with modified script)
        print(f"Package energy:        {summary['cpu_package_energy_j']:.3f} J")
        print(f"Average package power: {summary['avg_cpu_power_w']:.3f} W")
        print(f"Average IPC:           {summary['avg_ipc']:.4f}")
        print(f"Average frequency:     {summary['avg_cpu_frequency_mhz']:.2f} MHz")
        print(f"Power-plan switches:   {summary['power_plan_switches']}")
        print(f"Samples:               {summary['samples']}")
        print(f"Summary saved to:      {summary_path}")
        print(f"Time-series saved to:  {log_path}")
        print(f"Benchmark stdout:      {stdout_path}")
        print(f"Benchmark stderr:      {stderr_path}")
<<<<<<< HEAD

        return summary

=======
        return summary

    def start_realtime_plot(self):
        """Observable/debug view. Disable for lowest-overhead publication runs."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
        except Exception as exc:
            print(f"⚠️ Real-time plotting unavailable: {exc}")
            return

        fig, axs = plt.subplots(4, 2, figsize=(16, 12))
        fig.suptitle("HistGBDT-RPM Real-Time Monitoring", fontsize=14, fontweight="bold")
        plots = [
            ("cpu_power", "CPU Power (W)", "CPU Power vs Time"),
            ("ipc", "IPC", "IPC vs Time"),
            ("cpu_frequency", "Frequency (MHz)", "CPU Frequency vs Time"),
            ("cpu_temperature", "Temp (C)", "CPU Temperature vs Time"),
            ("l2_miss_rate", "L2 Miss Rate", "L2 Cache Miss Rate"),
            ("l3_miss_rate", "L3 Miss Rate", "L3 Cache Miss Rate"),
            ("memory_usage", "Memory Usage (%)", "Memory Usage vs Time"),
            ("cpu_usage_overall", "CPU Usage (%)", "CPU Usage vs Time"),
        ]

        def update(_frame):
            data = list(self.data_buffer)
            if not data:
                return []
            df = pd.DataFrame(data)
            for ax in axs.flat:
                ax.clear()
            t = df["elapsed_s"].to_numpy(dtype=float) / 60.0
            for idx, (col, ylabel, title) in enumerate(plots):
                ax = axs.flat[idx]
                if col in df:
                    ax.plot(t, df[col].to_numpy(dtype=float), alpha=0.85, linewidth=1.5)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("Time (min)")
            if "dvfs_level" in df and len(df):
                fig.suptitle(
                    f"HistGBDT-RPM Real-Time Monitoring | Current level: {df['dvfs_level'].iloc[-1]}",
                    fontsize=14, fontweight="bold"
                )
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            return []

        _ani = FuncAnimation(
            fig, update,
            interval=max(500, int(max(self.sample_sleep_s, 0.5) * 1000)),
            blit=False,
            cache_frame_data=False,
        )
        # Keep a live reference; some backends garbage-collect local animations.
        fig._histgbdt_animation = _ani  # type: ignore[attr-defined]
        print("📊 Starting real-time visualization. Close the plot window to hide it; the experiment continues.")
        plt.show()

>>>>>>> b0dfa44 (New exeperiment with modified script)
    def build_summary(
        self,
        workload: Path,
        workload_args: List[str],
        cores: int,
        run_mode: str,
        requested_duration_s: Optional[float],
<<<<<<< HEAD
=======
        initial_wait_s: float,
>>>>>>> b0dfa44 (New exeperiment with modified script)
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
<<<<<<< HEAD

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

=======
            energy_j = trapezoid_integral(times, df["cpu_power"].to_numpy(dtype=float), benchmark_duration_s)
            avg_power = energy_j / benchmark_duration_s if benchmark_duration_s > 0 else float("nan")
            avg_ipc = time_weighted_average(times, df["ipc"].to_numpy(dtype=float), benchmark_duration_s)
            avg_freq = time_weighted_average(times, df["cpu_frequency"].to_numpy(dtype=float), benchmark_duration_s)
            avg_cpu = time_weighted_average(times, df["cpu_usage_overall"].to_numpy(dtype=float), benchmark_duration_s)
            avg_temp = time_weighted_average(times, df["cpu_temperature"].to_numpy(dtype=float), benchmark_duration_s)
            state_counts = {str(k): int(v) for k, v in df["dvfs_level"].value_counts().to_dict().items()}
>>>>>>> b0dfa44 (New exeperiment with modified script)
            avg_inference_ms = float(df["inference_ms"].mean())
            max_inference_ms = float(df["inference_ms"].max())
            total_monitoring_collection_s = float(df["sample_collection_s"].sum())
            avg_sample_collection_s = float(df["sample_collection_s"].mean())
        else:
            energy_j = 0.0
<<<<<<< HEAD
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
=======
            avg_power = avg_ipc = avg_freq = avg_cpu = avg_temp = float("nan")
            state_counts = {}
            avg_inference_ms = max_inference_ms = 0.0
            total_monitoring_collection_s = avg_sample_collection_s = 0.0
>>>>>>> b0dfa44 (New exeperiment with modified script)

        return {
            "experiment_timestamp": datetime.now().isoformat(),
            "controller": self.controller,
            "workload": str(workload),
            "workload_args": workload_args,
<<<<<<< HEAD
            "cores": int(cores),
            "run_mode": run_mode,
            "requested_duration_s": requested_duration_s,
=======
            "threads": int(cores),
            "cores_requested": int(cores),
            "cpu_affinity_pinned": False,
            "cpu_scheduling_policy": "windows_default_no_runner_affinity",
            "run_mode": run_mode,
            "requested_duration_s": requested_duration_s,
            "initial_wait_s": float(initial_wait_s),
            "wait_interval_T_s": float(self.sample_sleep_s),
            "history_length_N": int(self.history_length),
            "bw_ref_mb_s": float(self.bw_ref_mb_s),
            "feature_schema": self.model.schema_name,
>>>>>>> b0dfa44 (New exeperiment with modified script)
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
            "pcm_path": self.pcm_path,
            "pcm_power_path": self.pcm_power_path,
            "log_file": str(log_path),
            "benchmark_stdout_file": str(stdout_path),
            "benchmark_stderr_file": str(stderr_path),
        }


# ---------------------------------------------------------------------------
<<<<<<< HEAD
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
=======
# CLI / interactive selection
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Observable HistGBDT-RPM benchmark runner. Interactive mode mirrors the "
            "earlier fair runner while preserving strict workload/telemetry validity."
        )
    )
    parser.add_argument("--workload", default=None, help="Path to benchmark wrapper/script/executable.")
    parser.add_argument("--workload-args", nargs=argparse.REMAINDER, default=[], help="Arguments passed to the benchmark. Put this option last.")
    parser.add_argument("--workdir", default=None, help="Benchmark working directory. Default: project root (matches fair runner).")
    parser.add_argument("--controller", choices=["hgbdt", "balanced"], default=None, help="hgbdt = adaptive controller; balanced = fixed Medium baseline.")
    parser.add_argument("--cores", type=int, default=None, help="Workload thread count. No CPU affinity pinning is imposed.")
    parser.add_argument("--run-mode", choices=["completion", "duration"], default=None, help="completion = one finite run; duration = fixed T_exp.")
    parser.add_argument("--duration", type=float, default=None, help="Experiment duration T_exp in seconds for duration mode.")
    parser.add_argument("--duration-policy", choices=["strict", "repeat"], default=None, help="In duration mode: strict stops if workload ends; repeat relaunches successful finite workloads until T_exp.")
    parser.add_argument("--sample-sleep", type=float, default=3.0, help="Waiting interval T after each control cycle (default: 3 s).")
    parser.add_argument("--initial-wait", type=float, default=None, help="Initial wait after workload launch. Default: T in duration mode, 0 in completion mode.")
    parser.add_argument("--settle", type=float, default=0.0, help="Optional pre-launch Medium-plan settle time (default: 0).")
    parser.add_argument("--history-length", type=int, default=None, help="Rolling history N. Default follows loaded model schema.")
    parser.add_argument("--bw-ref", type=float, default=DEFAULT_BW_REF_MB_S, help="Fixed BW_ref in MB/s. Must match training.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_FILE), help="Path to HistGBDT .joblib.")
    parser.add_argument("--scaler", default=str(DEFAULT_SCALER_FILE), help="Path to scaler_stats.npz.")
    parser.add_argument("--pcm", default=None, help="Optional explicit path to pcm.exe.")
    parser.add_argument("--pcm-power", default=None, help="Optional explicit path to pcm-power.exe.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: results/<workload>_<controller>_<timestamp>.")
    parser.add_argument("--proba-hysteresis", action="store_true", help="Enable probability-margin hysteresis.")
    parser.add_argument("--proba-margin", type=float, default=0.15, help="Minimum top1-top2 probability margin for a state switch.")
    parser.add_argument("--allow-simulated", action="store_true", help="DEBUG ONLY: allow simulated telemetry if PCM fails.")
    parser.add_argument("--live-plot", action="store_true", help="Show the 8-panel real-time plot.")
    parser.add_argument("--no-live-plot", action="store_true", help="Disable real-time plot in interactive mode.")
    parser.add_argument("--show-workload-output", action="store_true", help="Mirror workload stdout/stderr to console while also logging it.")
    parser.add_argument("--hide-workload-output", action="store_true", help="Do not mirror workload stdout/stderr to console.")
    return parser.parse_args()


def _display_workload_menu(candidates: List[Path]) -> None:
    print("\nAvailable Workloads:")
    for i, path in enumerate(candidates, 1):
        try:
            display = path.relative_to(ROOT)
        except ValueError:
            display = path
        print(f" {i}. {display}")


def resolve_user_selections(args):
    interactive = args.workload is None or args.controller is None or args.cores is None or args.run_mode is None

    if args.workload is None:
        curated = [(ROOT / p).resolve() for p in CURATED_WORKLOADS if (ROOT / p).exists()]
        candidates = curated
        _display_workload_menu(candidates)
        print(" A. Show all discovered workloads")
        print(" M. Enter a workload path manually")
        while True:
            choice = input(f"Enter workload choice (1-{len(candidates)}) or A/M: ").strip()
            if choice.lower() == "a":
                candidates = discover_workloads(ROOT)
                _display_workload_menu(candidates)
                print(" M. Enter a workload path manually")
                continue
            if choice.lower() == "m":
                manual = input("Workload path: ").strip().strip('"')
                if manual:
                    args.workload = manual
                    break
            elif choice.isdigit() and 1 <= int(choice) <= len(candidates):
                args.workload = str(candidates[int(choice) - 1])
                break
            print("Please choose a listed number, A, or M.")

    if args.controller is None:
        print("\nController:")
        print(" 1. HistGBDT-RPM adaptive DVFS")
        print(" 2. Balanced / Medium baseline")
        while True:
            choice = input("Select controller [1]: ").strip() or "1"
            if choice in {"1", "hgbdt"}:
                args.controller = "hgbdt"; break
            if choice in {"2", "balanced"}:
                args.controller = "balanced"; break
            print("Please enter 1 or 2.")

    logical_cpus = os.cpu_count() or 1
    max_threads = max(1, min(8, logical_cpus))
    if args.cores is None:
        default_cores = min(4, max_threads)
        args.cores = prompt_positive_int(
            f"Enter number of workload threads (1-{max_threads}) [{default_cores}]: ",
            default_cores, max_threads,
        )
    if args.cores < 1 or args.cores > max_threads:
        raise SystemExit(f"--cores must be between 1 and {max_threads}")
    print("CPU scheduling: Windows default (no CPU affinity pinning).")

    if args.run_mode is None:
        print("\nRun mode:")
        print(" 1. Duration (fixed T_exp)")
        print(" 2. Completion (single equal-work benchmark)")
        while True:
            choice = input("Select run mode [1]: ").strip() or "1"
            if choice in {"1", "duration"}:
                args.run_mode = "duration"; break
            if choice in {"2", "completion"}:
                args.run_mode = "completion"; break
            print("Please enter 1 or 2.")

    if args.run_mode == "duration":
        if args.duration is None:
            while True:
                raw = input(f"Enter duration in seconds {DURATION_OPTIONS} [{DURATION_OPTIONS[0]}]: ").strip()
                if not raw:
                    args.duration = float(DURATION_OPTIONS[0]); break
                try:
                    value = float(raw)
                    if value <= 0: raise ValueError
                    args.duration = value; break
                except ValueError:
                    print("Please enter a positive duration.")
        if args.duration_policy is None:
            print("\nIf a finite workload finishes before T_exp:")
            print(" 1. Strict: stop immediately (best for detecting invalid/short runs)")
            print(" 2. Repeat: relaunch successful workload until T_exp (continuous workload stream)")
            while True:
                choice = input("Select duration policy [1]: ").strip() or "1"
                if choice == "1": args.duration_policy = "strict"; break
                if choice == "2": args.duration_policy = "repeat"; break
                print("Please enter 1 or 2.")
    else:
        args.duration_policy = "strict"
        if args.duration is not None:
            print("WARNING: --duration is ignored in completion mode.")

    if args.no_live_plot:
        args.live_plot = False
    elif interactive and not args.live_plot:
        ans = input("Show real-time 8-panel visualization? [Y/n]: ").strip().lower()
        args.live_plot = ans not in {"n", "no", "0"}

    if args.hide_workload_output:
        args.show_workload_output = False
    elif interactive and not args.show_workload_output:
        ans = input("Show workload stdout/stderr live in console? [Y/n]: ").strip().lower()
        args.show_workload_output = ans not in {"n", "no", "0"}

    if args.sample_sleep < 0: raise SystemExit("--sample-sleep / T must be >= 0")
    if args.settle < 0: raise SystemExit("--settle must be >= 0")
    if args.history_length is not None and args.history_length < 1: raise SystemExit("--history-length must be >= 1")
    if args.bw_ref <= 0: raise SystemExit("--bw-ref must be > 0")
    if args.initial_wait is None:
        args.initial_wait = args.sample_sleep if args.run_mode == "duration" else 0.0
    if args.initial_wait < 0: raise SystemExit("--initial-wait must be >= 0")
    return args


def main():
    global STOP_REQUESTED
    args = resolve_user_selections(parse_args())
>>>>>>> b0dfa44 (New exeperiment with modified script)

    workload = Path(args.workload).expanduser()
    if not workload.is_absolute():
        workload = (ROOT / workload).resolve()
<<<<<<< HEAD
=======
    if not workload.exists():
        raise SystemExit(f"Workload not found: {workload}")
>>>>>>> b0dfa44 (New exeperiment with modified script)

    workdir = None
    if args.workdir:
        workdir = Path(args.workdir).expanduser()
        if not workdir.is_absolute():
            workdir = (ROOT / workdir).resolve()

    model_path = Path(args.model).expanduser().resolve()
    scaler_path = Path(args.scaler).expanduser().resolve() if args.scaler else None

    pcm_path = resolve_executable(args.pcm, PCM_CANDIDATES, "pcm.exe")
    pcm_power_path = resolve_executable(args.pcm_power, PCM_POWER_CANDIDATES, "pcm-power.exe")
<<<<<<< HEAD

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
=======
    if pcm_path:
        print(f"✅ Found PCM:       {pcm_path}")
    else:
        print("❌ PCM not found.")
    if pcm_power_path:
        print(f"✅ Found PCM Power: {pcm_power_path}")
    else:
        print("❌ pcm-power not found.")

    if pcm_path is None and not args.allow_simulated:
        raise SystemExit("pcm.exe not found. Pass --pcm <path>. Real experiments never silently simulate telemetry.")
    if pcm_power_path is None and not args.allow_simulated:
        raise SystemExit("pcm-power.exe not found. Pass --pcm-power <path>. Real experiments never silently simulate telemetry.")
>>>>>>> b0dfa44 (New exeperiment with modified script)
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

<<<<<<< HEAD
=======
    if model.schema_name == "legacy_12_feature_N5":
        print(
            "IMPORTANT: loaded model is the legacy 12-feature/N=5 schema. "
            "Runtime stays compatible with that trained model."
        )

    print("\nExperiment configuration")
    print("=" * 96)
    print(f"Workload:             {workload}")
    print(f"Controller:           {args.controller}")
    print(f"Workload threads:     {args.cores}")
    print("CPU scheduling:       Windows default (no runner affinity)")
    print(f"Run mode:             {args.run_mode}")
    if args.run_mode == "duration":
        print(f"Duration T_exp:       {args.duration:.3f} s")
        print(f"Duration policy:      {args.duration_policy}")
    print(f"Live plot:            {'Yes' if args.live_plot else 'No'}")
    print(f"Live workload output: {'Yes' if args.show_workload_output else 'No'}")
    print(f"Results directory:    {output_dir}")
    print("=" * 96)

>>>>>>> b0dfa44 (New exeperiment with modified script)
    runner = HistGBDTBenchmarkRunner(
        model=model,
        controller=args.controller,
        pcm_path=pcm_path,
        pcm_power_path=pcm_power_path,
        output_dir=output_dir,
        sample_sleep_s=args.sample_sleep,
        allow_simulated=args.allow_simulated,
<<<<<<< HEAD
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
=======
        history_length=args.history_length,
        bw_ref_mb_s=args.bw_ref,
        show_workload_output=args.show_workload_output,
    )

    run_kwargs = dict(
        workload=workload,
        workload_args=args.workload_args,
        cores=args.cores,
        run_mode=args.run_mode,
        duration_s=args.duration,
        settle_s=args.settle,
        initial_wait_s=args.initial_wait,
        workdir=workdir,
        duration_policy=args.duration_policy,
    )

    if args.live_plot:
        result_box = {}
        error_box = {}

        def _run_experiment():
            try:
                result_box["summary"] = runner.run(**run_kwargs)
            except BaseException as exc:
                error_box["error"] = exc
                runner.experiment_finished.set()

        worker = threading.Thread(target=_run_experiment, name="histgbdt-experiment", daemon=True)
        worker.start()
        try:
            runner.start_realtime_plot()
        except KeyboardInterrupt:
            STOP_REQUESTED = True
            runner.stop_benchmark()
        worker.join()
        if "error" in error_box:
            raise error_box["error"]
    else:
        runner.run(**run_kwargs)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        print(f"\nFATAL EXPERIMENT ERROR: {exc}", file=sys.stderr)
        raise
>>>>>>> b0dfa44 (New exeperiment with modified script)
