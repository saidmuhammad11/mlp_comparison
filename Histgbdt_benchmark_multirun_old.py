#!/usr/bin/env python3
"""
HistGBDT Benchmark Runner for Windows (Fully Integrated)
========================================================
Features:
- Real hardware telemetry via PCM (running from correct directory).
- Unified Policy Engine (Energy Save & Memory-Bound Protection).
- Interactive workload/controller selection.
- Real-time 8-panel visualization (disabled for batch runs).
- Live workload output streaming to console and log file.
- Multi-run batch execution for automated averaging.
"""
import argparse
import base64
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
    print("\n⚠️ Ctrl+C detected - requesting a clean shutdown...")

signal.signal(signal.SIGINT, handle_ctrl_c)

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_FILE = ROOT / "models" / "histgbdt.joblib"
DEFAULT_SCALER_FILE = ROOT / "dataset" / "scaler_stats.npz"

ALGORITHM_HISTORY_LENGTH = 8
LEGACY_HISTORY_LENGTH = 5
DEFAULT_BW_REF_MB_S = 25000.0

# Power Plan GUIDs (Custom Low plan GUID included)
POWER_PLANS = {
    "High": "27ad8305-4092-41f2-a01b-5a6003fb5077",
    "Medium": "381b4222-f694-41f0-9685-ff5bb260df2e",
    "Low": "b2524225-86dc-424d-ba5b-e78d683c5d3a",
}

# CRITICAL FIX: ROOT / "pcm.exe" MUST BE FIRST to find the local msr.sys driver
PCM_CANDIDATES = [
    ROOT / "pcm.exe",
    Path(r"C:\Users\saidm\Desktop\Programming\pcm\build\bin\Release\pcm.exe"),
    Path(r"C:\Users\saidm\pcm\build\bin\Release\pcm.exe"),
]

PCM_POWER_CANDIDATES = [
    ROOT / "pcm-power.exe",
    Path(r"C:\Users\saidm\Desktop\Programming\pcm\build\bin\Release\pcm-power.exe"),
    Path(r"C:\Users\saidm\pcm\build\bin\Release\pcm-power.exe"),
]

CURATED_WORKLOADS = [
    "workload_mem2_stream_huge.py",
    "NPB3.0-omp-C/workload_cpu1_nas_ep.py",
    "NPB3.0-omp-C/workload_mem1_bursty_is.py",
    "NPB3.0-omp-C/workload_cpu2_bursty_ep.py",
    "NPB3.0-omp-C/workload_mixed1_bursty_lu.py",
    "NPB3.0-omp-C/workload_mixed2_bursty_mg.py",
]

DURATION_OPTIONS = [100, 200, 300]

# --- UNIFIED POLICY FLAGS ---
ENERGY_SAVE_MODE = True
IPC_PERFORMANCE_FLOOR = 1.0

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def resolve_executable(explicit: Optional[str], candidates: List[Path], command_name: str) -> Optional[str]:
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
    if len(times_s) == 0 or duration_s <= 0:
        return 0.0
    order = np.argsort(times_s)
    t = np.asarray(times_s[order], dtype=float)
    y = np.asarray(values[order], dtype=float)
    
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    
    if len(t) == 0:
        return 0.0
        
    t = np.clip(t, 0.0, duration_s)
    
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
        
    return float(np.trapz(y, t))

def time_weighted_average(times_s: np.ndarray, values: np.ndarray, duration_s: float) -> float:
    if duration_s <= 0:
        return float("nan")
    return trapezoid_integral(times_s, values, duration_s) / duration_s

def discover_workloads(root: Path) -> List[Path]:
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

# ---------------------------------------------------------------------------
# Workload wrapper path repair
# ---------------------------------------------------------------------------
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
    normalized = path_text.replace("/", "\\")
    marker = "\\OneDrive\\Desktop\\"
    if marker.lower() in normalized.lower():
        idx = normalized.lower().index(marker.lower())
        normalized = normalized[:idx] + "\\Desktop\\" + normalized[idx + len(marker):]
    return Path(normalized)

def resolve_legacy_workload_executable(path_text: str) -> Optional[Path]:
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
            
    matches: List[Path] = []
    for search_root in WORKLOAD_SEARCH_ROOTS:
        try:
            if not search_root.exists() or not search_root.is_dir():
                continue
            for found in search_root.rglob(exe_name):
                if found.is_file():
                    matches.append(found.resolve())
            if len(matches) >= 8:
                break
        except (OSError, PermissionError):
            continue
            
    if not matches:
        return None
        
    matches = sorted(set(matches), key=lambda p: (len(p.parts), str(p).lower()))
    return matches[0]

def prepare_python_workload_source(workload: Path) -> Tuple[str, Dict[str, str]]:
    source = workload.read_text(encoding="utf-8", errors="replace")
    repairs: Dict[str, str] = {}
    for match in ABSOLUTE_BIN_LITERAL_RE.finditer(source):
        old_text = match.group("path")
        old_path = Path(old_text)
        if old_path.exists():
            continue
        resolved = resolve_legacy_workload_executable(old_text)
        if resolved is None:
            raise FileNotFoundError(f"Workload wrapper contains a missing executable path: {old_text}")
        repairs[old_text] = resolved.as_posix()
        
    patched = source
    for old_text, new_text in repairs.items():
        patched = patched.replace(old_text, new_text)
    return patched, repairs

def build_in_memory_wrapper_command(workload: Path, workload_args: List[str]) -> Tuple[List[str], Dict[str, str]]:
    source, repairs = prepare_python_workload_source(workload)
    if not repairs:
        return [sys.executable, "-u", str(workload)] + workload_args, repairs
        
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
    return [sys.executable, "-u", "-c", bootstrap] + workload_args, repairs

# ---------------------------------------------------------------------------
# HistGBDT model wrapper
# ---------------------------------------------------------------------------
class HistGBDTModel:
    ALGORITHM_FEATURE_ORDER = [
        "ipc", "l2_miss_rate", "l3_miss_rate", "bw_util", "cpu_usage_overall",
        "cpu_temperature", "cpu_power", "cpu_frequency", "ipc_change",
        "ipc_avg", "l3_miss_rate_avg",
    ]
    LEGACY_FEATURE_ORDER = [
        "ipc", "l2_miss_rate", "l3_miss_rate", "memory_bandwidth",
        "cpu_usage_overall", "cpu_temperature", "cpu_power", "cpu_frequency",
        "ipc_change", "ipc_avg_5", "l3_miss_rate_avg_5", "bw_util",
    ]

    def __init__(self, model_path: Path, scaler_path: Optional[Path],
                 enable_proba_hysteresis: bool = False, proba_margin: float = 0.15):
        if not model_path.exists():
            raise FileNotFoundError(f"HistGBDT model not found: {model_path}")
            
        self.model = joblib.load(model_path)
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.scaler_mean, self.scaler_scale = load_scaler_stats(scaler_path)
        
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
            raise ValueError(f"Unsupported HistGBDT feature count. Expected 11 or 12, got {feature_n}.")
            
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
        print(f"Feature schema:        {self.schema_name}")
        print(f"Feature count:         {len(self.FEATURE_ORDER)}")

    @staticmethod
    def _build_class_mapping(classes) -> Dict:
        classes_list = list(classes)
        if all(isinstance(c, str) for c in classes_list):
            mapping = {}
            for c in classes_list:
                norm = c.strip().lower()
                if norm == "low": mapping[c] = "Low"
                elif norm == "medium": mapping[c] = "Medium"
                elif norm == "high": mapping[c] = "High"
                else: raise ValueError(f"Unexpected string class label: {c!r}")
            return mapping
            
        try:
            class_ints = [int(c) for c in classes_list]
        except Exception as exc:
            raise ValueError(f"Unsupported HistGBDT classes_: {classes_list}") from exc
            
        if set(class_ints) == {0, 1, 2}:
            return {0: "Low", 1: "Medium", 2: "High"}
        if len(class_ints) != 3:
            raise ValueError(f"Expected exactly 3 classes, got: {classes_list}")
            
        ordered = sorted(class_ints)
        return {ordered[0]: "Low", ordered[1]: "Medium", ordered[2]: "High"}

    def _validate_feature_dimensions(self):
        expected = len(self.FEATURE_ORDER)
        model_n = getattr(self.model, "n_features_in_", None)
        if model_n is not None and int(model_n) != expected:
            raise ValueError(f"MODEL FEATURE MISMATCH: model expects {model_n}, schema defines {expected}.")
        if self.scaler_mean is not None and len(self.scaler_mean) != expected:
            raise ValueError(f"SCALER FEATURE MISMATCH: mean has {len(self.scaler_mean)} values, expected {expected}.")
        if self.scaler_scale is not None and len(self.scaler_scale) != expected:
            raise ValueError(f"SCALER FEATURE MISMATCH: scale has {len(self.scaler_scale)} values, expected {expected}.")

    def _prepare_vector(self, features: Dict[str, float]) -> np.ndarray:
        missing = [name for name in self.FEATURE_ORDER if name not in features]
        if missing:
            raise KeyError(f"Missing HistGBDT input features: {missing}")
            
        x = np.asarray([float(features[name]) for name in self.FEATURE_ORDER], dtype=np.float32)
        if not np.all(np.isfinite(x)):
            bad = {name: float(value) for name, value in zip(self.FEATURE_ORDER, x) if not np.isfinite(value)}
            raise ValueError(f"Non-finite model input(s): {bad}")
            
        if self.scaler_mean is not None and self.scaler_scale is not None:
            x = (x - self.scaler_mean) / self.scaler_scale
            
        return x.reshape(1, -1)

    def predict_level(self, features: Dict[str, float]) -> Tuple[str, float, Optional[float]]:
        X = self._prepare_vector(features)
        t0 = time.perf_counter()
        probability_margin = None
        
        if self.enable_proba_hysteresis and hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            sorted_idx = np.argsort(proba)[::-1]
            top_idx = int(sorted_idx[0])
            second_idx = int(sorted_idx[1])
            probability_margin = float(proba[top_idx] - proba[second_idx])
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
    def __init__(self, model: HistGBDTModel, controller: str, pcm_path: str,
                 pcm_power_path: str, output_dir: Path, sample_sleep_s: float = 3.0,
                 allow_simulated: bool = False, history_length: Optional[int] = None,
                 bw_ref_mb_s: float = DEFAULT_BW_REF_MB_S, show_workload_output: bool = True):
                 
        if controller not in {"hgbdt", "high", "medium", "low"}:
            raise ValueError("controller must be 'hgbdt', 'high', 'medium', or 'low'")
            
        self.model = model
        self.controller = controller
        self.pcm_path = pcm_path
        self.pcm_power_path = pcm_power_path
        self.output_dir = output_dir
        self.sample_sleep_s = float(sample_sleep_s)
        self.allow_simulated = allow_simulated
        self.history_length = int(history_length or model.default_history_length)
        self.bw_ref_mb_s = float(bw_ref_mb_s)
        self.show_workload_output = bool(show_workload_output)
        
        self.data_buffer: List[Dict] = []
        self.active_level: Optional[str] = None
        self.switch_count = 0
        self.powercfg_time_s = 0.0
        self.benchmark_process: Optional[subprocess.Popen] = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_finished = threading.Event()
        self.workload_launch_count = 0

    @staticmethod
    def get_active_power_plan_guid() -> Optional[str]:
        try:
            result = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, check=True, timeout=10)
        except Exception:
            return None
        match = re.search(r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}", result.stdout)
        return match.group(0).lower() if match else None

    def set_power_plan(self, level: str, force: bool = False, record_experiment: bool = True) -> Tuple[bool, float]:
        if level not in POWER_PLANS:
            raise ValueError(f"Invalid level: {level}")
        if not force and level == self.active_level:
            return True, 0.0
            
        guid = POWER_PLANS[level]
        t0 = time.perf_counter()
        try:
            subprocess.run(["powercfg", "/setactive", guid], capture_output=True, text=True, check=True, timeout=10)
        except Exception as exc:
            raise RuntimeError(f"Failed to activate Windows power plan {level}: {exc}") from exc
            
        elapsed = time.perf_counter() - t0
        previous = self.active_level
        self.active_level = level
        
        if record_experiment:
            self.powercfg_time_s += elapsed
            
        if previous is not None and previous != level:
            self.switch_count += 1
            print(f"🔄 DVFS switch: {previous} -> {level} | powercfg {elapsed*1000.0:.2f} ms")
            
        return True, elapsed

    def _simulate_pcm_data(self) -> Dict[str, float]:
        return {"ipc": 1.0, "l2_cache_hits": 90000, "l2_cache_misses": 10000, "l3_cache_hits": 85000, "l3_cache_misses": 15000, "memory_bandwidth": 15000.0}

    def _simulate_power_data(self) -> Dict[str, float]:
        return {"cpu_power": 15.0, "gpu_power": 5.0, "cpu_temperature": 55.0, "cpu_frequency": 1500.0}

    def read_pcm_data(self) -> Dict[str, float]:
        try:
            pcm_dir = os.path.dirname(self.pcm_path)
            result = subprocess.run(
                [self.pcm_path, "1", "-nc", "-ns", "-i=1", "-r"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=pcm_dir
            )
            if result.returncode != 0:
                raise RuntimeError(f"pcm.exe returned {result.returncode}: {result.stderr.strip()}")
                
            data_line = None
            for line in result.stdout.splitlines():
                if "TOTAL" in line and "*" in line:
                    data_line = line
                    break
                    
            if not data_line:
                raise RuntimeError("Could not find TOTAL row in PCM output")
                
            clean_line = re.sub(r'\s+([KMG])\b', r'\1', data_line)
            clean_line = clean_line.replace('|', ' ')
            parts = [x for x in clean_line.split() if x]
            
            ipc = float(parts[3])
            cfreq = float(parts[4])
            l3_misses_raw = parts[5]
            l2_misses_raw = parts[6]
            l3_hit_ratio = float(parts[7])
            l2_hit_ratio = float(parts[8])
            
            def parse_misses(val):
                if val.endswith('K'): return float(val[:-1]) * 1000
                if val.endswith('M'): return float(val[:-1]) * 1000000
                return float(val)
                
            l3_misses = int(parse_misses(l3_misses_raw))
            l2_misses = int(parse_misses(l2_misses_raw))
            
            l2_hits = int((l2_misses / (1.0 - l2_hit_ratio)) * l2_hit_ratio) if l2_hit_ratio < 1.0 else 100000
            l3_hits = int((l3_misses / (1.0 - l3_hit_ratio)) * l3_hit_ratio) if l3_hit_ratio < 1.0 else 100000
            
            power = 0.0
            energy_match = re.search(r"SYS energy:\s*([\d.]+)\s*J", result.stdout)
            if energy_match:
                power = float(energy_match.group(1))
            
            bw_mb_s = (l3_misses * 64) / (1024 * 1024)
            
            return {
                "ipc": ipc,
                "memory_bandwidth": bw_mb_s,
                "l2_cache_hits": l2_hits,
                "l2_cache_misses": l2_misses,
                "l3_cache_hits": l3_hits,
                "l3_cache_misses": l3_misses,
                "cpu_frequency": cfreq,
                "cpu_power": power,
            }
        except Exception as e:
            if self.allow_simulated:
                return self._simulate_pcm_data()
            if STOP_REQUESTED:
                return self._simulate_pcm_data()
            raise RuntimeError(f"PCM parsing failed: {e}")

    def read_power_data(self) -> Dict[str, float]:
        return {
            'cpu_power': 0.0,
            'gpu_power': 0.0,
            'cpu_temperature': 0.0,
            'cpu_frequency': 0.0
        }

    def feature_engineer(self, data: Dict) -> Dict:
        history = self.data_buffer + [data]
        df = pd.DataFrame(history)
        data["ipc_change"] = float(df["ipc"].diff().iloc[-1]) if len(df) > 1 else 0.0
        
        ipc_avg = float(df["ipc"].rolling(window=self.history_length, min_periods=1).mean().iloc[-1])
        l3_avg = float(df["l3_miss_rate"].rolling(window=self.history_length, min_periods=1).mean().iloc[-1])
        
        data["ipc_avg"] = ipc_avg
        data["l3_miss_rate_avg"] = l3_avg
        data["ipc_avg_5"] = ipc_avg
        data["l3_miss_rate_avg_5"] = l3_avg
        
        bw = float(data["memory_bandwidth"])
        data["bw_util"] = float(np.clip(bw / self.bw_ref_mb_s, 0.0, 1.5))
        return data

    def collect_sample(self, benchmark_start_perf: float) -> Dict:
        sample_collect_start = time.perf_counter()
        pcm = self.read_pcm_data()
        power = self.read_power_data()
        
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().percent
        
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
            "cpu_power": round(float(pcm.get("cpu_power", 0.0)), 3),
            "cpu_temperature": 0.0,
            "cpu_frequency": round(float(pcm.get("cpu_frequency", 0.0)), 3),
        }
        
        data = self.feature_engineer(data)
        
        inference_ms = 0.0
        probability_margin = None
        
        if self.controller == "hgbdt":
            predicted_level, inference_ms, probability_margin = self.model.predict_level(data)
            level = predicted_level
            
            if ENERGY_SAVE_MODE:
                if level == "High" and data.get('ipc_avg_5', 0) < IPC_PERFORMANCE_FLOOR:
                    level = "Medium"
                    
            if data.get('l3_miss_rate', 0) > 0.40 and data.get('ipc', 0) < 1.2:
                level = "Low"
                
            applied, actuation_s = self.set_power_plan(level)
        else:
            level = self.controller.capitalize()
            applied, actuation_s = self.set_power_plan(level)
            
        data["controller"] = self.controller
        data["dvfs_level"] = level
        data["dvfs_applied"] = bool(applied)
        data["inference_ms"] = float(inference_ms)
        data["probability_margin"] = None if probability_margin is None else float(probability_margin)
        data["actuation_s"] = float(actuation_s)
        data["sample_collection_s"] = float(time.perf_counter() - sample_collect_start)
        return data

    def build_benchmark_command(self, workload: Path, workload_args: List[str]) -> List[str]:
        suffix = workload.suffix.lower()
        self._last_workload_path_repairs = {}
        if suffix == ".py":
            cmd, repairs = build_in_memory_wrapper_command(workload, workload_args)
            self._last_workload_path_repairs = repairs
            return cmd
        return [str(workload)] + workload_args

    def start_benchmark(self, workload: Path, workload_args: List[str], cores: int, workdir: Optional[Path], relaunch: bool = False) -> Tuple[subprocess.Popen, Path, Path]:
        if not workload.exists():
            raise FileNotFoundError(f"Benchmark/workload not found: {workload}")
            
        cwd = (workdir or ROOT).resolve()
        if not cwd.exists():
            raise FileNotFoundError(f"Benchmark working directory not found: {cwd}")
            
        env = os.environ.copy()
        thread_count = str(cores)
        env["OMP_NUM_THREADS"] = thread_count
        env["OMP_DYNAMIC"] = "FALSE"
        env["MKL_NUM_THREADS"] = thread_count
        env["MKL_DYNAMIC"] = "FALSE"
        env["OPENBLAS_NUM_THREADS"] = thread_count
        env["NUMEXPR_NUM_THREADS"] = thread_count
        
        stdout_path = self.output_dir / "benchmark_stdout.txt"
        stderr_path = self.output_dir / "benchmark_stderr.txt"
        mode = "a" if (relaunch or self.workload_launch_count > 0) else "w"
        
        stdout_f = open(stdout_path, mode, buffering=1, encoding="utf-8", errors="replace")
        stderr_f = open(stderr_path, mode, buffering=1, encoding="utf-8", errors="replace")
        
        cmd = self.build_benchmark_command(workload, workload_args)
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, creationflags=creationflags
        )
        
        def stream_output():
            try:
                for line in proc.stdout:
                    stdout_f.write(line)
                    stdout_f.flush()
                    if self.show_workload_output:
                        print(f"[WORKLOAD] {line}", end="", flush=True)
            except Exception:
                pass
            finally:
                try:
                    stdout_f.close()
                except Exception:
                    pass
                    
        threading.Thread(target=stream_output, daemon=True).start()
        
        proc._benchmark_stdout_handle = stdout_f
        proc._benchmark_stderr_handle = stderr_f
        self.benchmark_process = proc
        self.workload_launch_count += 1
        print(f"✅ Workload started (PID {proc.pid}, launch #{self.workload_launch_count}).")
        
        return proc, stdout_path, stderr_path

    def stop_benchmark(self):
        proc = self.benchmark_process
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, text=True, check=False)
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
                    handle.flush()
                    handle.close()
                except Exception:
                    pass

    def start_realtime_plot(self):
        print("📊 Starting real-time visualization...")
        fig, axs = plt.subplots(4, 2, figsize=(16, 12))
        fig.suptitle("HistGBDT-RPM Real-Time Monitoring", fontsize=14, fontweight='bold')

        def update(frame):
            if not self.data_buffer:
                return
            df = pd.DataFrame(self.data_buffer)
            df['time_min'] = df['elapsed_s'] / 60.0
            
            for ax in axs.flat:
                ax.clear()
                
            plots = [
                ('cpu_power', 'CPU Power (W)', 'CPU Power vs Time'),
                ('ipc', 'IPC', 'IPC vs Time'),
                ('cpu_frequency', 'Frequency (MHz)', 'CPU Frequency vs Time'),
                ('cpu_temperature', 'Temp (°C)', 'CPU Temperature vs Time'),
                ('l2_miss_rate', 'L2 Miss Rate', 'L2 Cache Miss Rate'),
                ('l3_miss_rate', 'L3 Miss Rate', 'L3 Cache Miss Rate'),
                ('memory_usage', 'Memory Usage (%)', 'Memory Usage vs Time'),
                ('cpu_usage_overall', 'CPU Usage (%)', 'CPU Usage vs Time'),
            ]
            
            for idx, (col, ylabel, title) in enumerate(plots):
                if col in df.columns:
                    ax = axs.flat[idx]
                    ax.plot(df['time_min'], df[col], alpha=0.8, linewidth=1.5)
                    ax.set_ylabel(ylabel)
                    ax.set_title(title)
                    ax.grid(True, alpha=0.3)
                    
            plt.tight_layout(rect=[0, 0, 1, 0.96])

        _ = FuncAnimation(fig, update, interval=3000, blit=False, cache_frame_data=False)
        plt.show()

    def run(self, workload: Path, workload_args: List[str], cores: int, run_mode: str,
            duration_s: Optional[float], settle_s: float, initial_wait_s: float,
            workdir: Optional[Path], duration_policy: str = "strict") -> Dict:
        global STOP_REQUESTED
        if run_mode not in {"completion", "duration"}:
            raise ValueError("run_mode must be 'completion' or 'duration'")
        if run_mode == "duration" and (duration_s is None or duration_s <= 0):
            raise ValueError("--duration must be > 0 in duration mode")
            
        self.workload_launch_count = 0
        self.data_buffer.clear()
        
        initial_plan = "Medium" if self.controller == "hgbdt" else self.controller.capitalize()
        self.set_power_plan(initial_plan, force=True, record_experiment=False)
        
        if settle_s > 0:
            print(f"Pre-launch {initial_plan} settle: {settle_s:.1f} s")
            time.sleep(settle_s)
            
        psutil.cpu_percent(interval=None)
        log_path = self.output_dir / "monitoring.jsonl"
        
        proc, stdout_path, stderr_path = self.start_benchmark(workload=workload, workload_args=workload_args, cores=cores, workdir=workdir)
        benchmark_start_perf = time.perf_counter()
        benchmark_start_wall = datetime.now()
        
        print(f"Benchmark PID:     {proc.pid}")
        print(f"Controller:        {self.controller}")
        print(f"Run mode:          {run_mode}")
        if run_mode == "duration":
            print(f"Target duration:   {duration_s:.3f} s")
            print(f"Duration policy:   {duration_policy}")
        print("-" * 72)
        
        exit_reason = "completed"
        iteration = 0
        
        with open(log_path, "w", buffering=1, encoding="utf-8") as log_f:
            while not STOP_REQUESTED:
                elapsed = time.perf_counter() - benchmark_start_perf
                
                if run_mode == "duration" and duration_s is not None and elapsed >= duration_s:
                    exit_reason = "duration_reached"
                    break
                    
                rc = self.benchmark_process.poll()
                if rc is not None:
                    self.close_benchmark_streams(self.benchmark_process)
                    if run_mode == "completion":
                        exit_reason = "benchmark_completed"
                        break
                    if duration_policy == "strict":
                        exit_reason = "benchmark_completed_before_duration"
                        print("\n️ Workload completed before T_exp; stopping rather than monitoring idle time.")
                        break
                        
                iteration += 1
                sample = self.collect_sample(benchmark_start_perf)
                self.data_buffer.append(sample)
                log_f.write(json.dumps(sample) + "\n")
                log_f.flush()
                
                if iteration % 3 == 1:
                    print(f"[{iteration:03d}] t={sample['elapsed_s']:7.2f}s | Level={sample['dvfs_level']:6s} | IPC={sample['ipc']:5.2f} | P={sample['cpu_power']:6.2f}W")
                    
                if self.sample_sleep_s > 0:
                    time.sleep(self.sample_sleep_s)
                    
        if STOP_REQUESTED:
            exit_reason = "user_interrupted"
            
        if self.benchmark_process is not None and self.benchmark_process.poll() is None:
            self.stop_benchmark()
        if self.benchmark_process is not None:
            self.close_benchmark_streams(self.benchmark_process)
            
        try:
            self.set_power_plan("Medium", force=True, record_experiment=False)
        except Exception as exc:
            print(f"WARNING: could not restore Medium power plan: {exc}")
            
        benchmark_end_perf = time.perf_counter()
        benchmark_end_wall = datetime.now()
        benchmark_duration_s = benchmark_end_perf - benchmark_start_perf
        
        if exit_reason == "duration_reached" and duration_s is not None:
            benchmark_duration_s = min(benchmark_duration_s, float(duration_s))
            
        if self.data_buffer:
            df = pd.DataFrame(self.data_buffer)
            times = df["elapsed_s"].to_numpy(dtype=float)
            energy_j = trapezoid_integral(times, df["cpu_power"].to_numpy(dtype=float), benchmark_duration_s)
            avg_power = energy_j / benchmark_duration_s if benchmark_duration_s > 0 else float("nan")
            avg_ipc = time_weighted_average(times, df["ipc"].to_numpy(dtype=float), benchmark_duration_s)
            avg_freq = time_weighted_average(times, df["cpu_frequency"].to_numpy(dtype=float), benchmark_duration_s)
            avg_cpu = time_weighted_average(times, df["cpu_usage_overall"].to_numpy(dtype=float), benchmark_duration_s)
            state_counts = {str(k): int(v) for k, v in df["dvfs_level"].value_counts().to_dict().items()}
        else:
            energy_j = 0.0
            avg_power = avg_ipc = avg_freq = avg_cpu = float("nan")
            state_counts = {}
            
        summary = {
            "experiment_timestamp": datetime.now().isoformat(),
            "controller": self.controller,
            "workload": str(workload),
            "threads": int(cores),
            "run_mode": run_mode,
            "requested_duration_s": duration_s,
            "duration_policy": duration_policy if run_mode == "duration" else None,
            "benchmark_start": benchmark_start_wall.isoformat(),
            "benchmark_end": benchmark_end_wall.isoformat(),
            "benchmark_duration_s": float(benchmark_duration_s),
            "exit_reason": exit_reason,
            "samples": int(len(self.data_buffer)),
            "cpu_package_energy_j": float(energy_j),
            "avg_cpu_power_w": float(avg_power),
            "avg_ipc": float(avg_ipc),
            "avg_cpu_frequency_mhz": float(avg_freq),
            "avg_cpu_usage_percent": float(avg_cpu),
            "power_plan_switches": int(self.switch_count),
            "state_decision_counts": state_counts,
        }
        
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        
        print("=" * 72)
        print("EXPERIMENT SUMMARY")
        print(f"Exit reason:           {exit_reason}")
        print(f"Duration:              {summary['benchmark_duration_s']:.3f} s")
        print(f"Package energy:        {summary['cpu_package_energy_j']:.3f} J")
        print(f"Average package power: {summary['avg_cpu_power_w']:.3f} W")
        print(f"Average IPC:           {summary['avg_ipc']:.4f}")
        print(f"Power-plan switches:   {summary['power_plan_switches']}")
        print(f"Summary saved to:      {summary_path}")
        
        return summary

# ---------------------------------------------------------------------------
# CLI / interactive selection
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Observable HistGBDT-RPM benchmark runner.")
    parser.add_argument("--runs", type=int, default=None, help="Number of consecutive times to run the workload.")
    parser.add_argument("--workload", default=None, help="Path to benchmark wrapper/script/executable.")
    parser.add_argument("--workload-args", nargs=argparse.REMAINDER, default=[], help="Arguments passed to the benchmark.")
    parser.add_argument("--workdir", default=None, help="Benchmark working directory.")
    parser.add_argument("--controller", choices=["hgbdt", "high", "medium", "low"], default=None, help="hgbdt = adaptive controller; high/medium/low = static baselines.")
    parser.add_argument("--cores", type=int, default=None, help="Workload thread count.")
    parser.add_argument("--run-mode", choices=["completion", "duration"], default=None, help="completion = one finite run; duration = fixed T_exp.")
    parser.add_argument("--duration", type=float, default=None, help="Experiment duration T_exp in seconds for duration mode.")
    parser.add_argument("--duration-policy", choices=["strict", "repeat"], default=None, help="In duration mode: strict stops if workload ends; repeat relaunches.")
    parser.add_argument("--sample-sleep", type=float, default=3.0, help="Waiting interval T after each control cycle.")
    parser.add_argument("--initial-wait", type=float, default=None, help="Initial wait after workload launch.")
    parser.add_argument("--settle", type=float, default=0.0, help="Optional pre-launch Medium-plan settle time.")
    parser.add_argument("--history-length", type=int, default=None, help="Rolling history N.")
    parser.add_argument("--bw-ref", type=float, default=DEFAULT_BW_REF_MB_S, help="Fixed BW_ref in MB/s.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_FILE), help="Path to HistGBDT .joblib.")
    parser.add_argument("--scaler", default=str(DEFAULT_SCALER_FILE), help="Path to scaler_stats.npz.")
    parser.add_argument("--pcm", default=None, help="Optional explicit path to pcm.exe.")
    parser.add_argument("--pcm-power", default=None, help="Optional explicit path to pcm-power.exe.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument("--proba-hysteresis", action="store_true", help="Enable probability-margin hysteresis.")
    parser.add_argument("--proba-margin", type=float, default=0.15, help="Minimum top1-top2 probability margin.")
    parser.add_argument("--allow-simulated", action="store_true", help="DEBUG ONLY: allow simulated telemetry.")
    parser.add_argument("--live-plot", action="store_true", default=True, help="Show the 8-panel real-time plot.")
    parser.add_argument("--show-workload-output", action="store_true", default=True, help="Mirror workload stdout/stderr to console.")
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
    if args.workload is None:
        curated = [(ROOT / p).resolve() for p in CURATED_WORKLOADS if (ROOT / p).exists()]
        candidates = curated
        _display_workload_menu(candidates)
        while True:
            choice = input(f"Enter workload choice (1-{len(candidates)}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                args.workload = str(candidates[int(choice) - 1])
                break
            print("Please choose a listed number.")
            
    if args.controller is None:
        print("\nSelect Controller Mode:")
        print(" 1. HistGBDT-RPM (Proposed ML Controller)")
        print(" 2. Baseline: Fixed High")
        print(" 3. Baseline: Fixed Medium")
        print(" 4. Baseline: Fixed Low")
        while True:
            choice = input("Enter choice (1-4) [1]: ").strip() or "1"
            if choice == "1":
                args.controller = "hgbdt"; break
            elif choice == "2":
                args.controller = "high"; break
            elif choice == "3":
                args.controller = "medium"; break
            elif choice == "4":
                args.controller = "low"; break
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
                
    if args.cores is None:
        logical_cpus = os.cpu_count() or 1
        max_threads = max(1, min(8, logical_cpus))
        args.cores = prompt_positive_int(f"Enter number of workload threads (1-{max_threads}) [4]: ", 4, max_threads)

    if args.runs is None:
        args.runs = prompt_positive_int("Enter number of consecutive runs (e.g., 5 for automated averaging) [1]: ", 1)
        
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
                
    if args.run_mode == "duration" and args.duration is None:
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
        if args.run_mode == "duration":
            print("\nIf a finite workload finishes before T_exp:")
            print(" 1. Strict: stop immediately")
            print(" 2. Repeat: relaunch successful workload until T_exp")
            while True:
                choice = input("Select duration policy [1]: ").strip() or "1"
                if choice == "1": args.duration_policy = "strict"; break
                if choice == "2": args.duration_policy = "repeat"; break
        else:
            args.duration_policy = "strict"
            
    if args.initial_wait is None:
        args.initial_wait = args.sample_sleep if args.run_mode == "duration" else 0.0
        
    return args

def main():
    global STOP_REQUESTED
    args = resolve_user_selections(parse_args())
    
    workload = Path(args.workload).expanduser()
    if not workload.is_absolute():
        workload = (ROOT / workload).resolve()
    if not workload.exists():
        raise SystemExit(f"Workload not found: {workload}")
        
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
        raise SystemExit("pcm.exe not found. Pass --pcm <path>.")
        
    pcm_path = pcm_path or "pcm.exe"
    pcm_power_path = pcm_power_path or "pcm-power.exe"
    
    try:
        model = HistGBDTModel(model_path=model_path, scaler_path=scaler_path,
                              enable_proba_hysteresis=args.proba_hysteresis, proba_margin=args.proba_margin)
    except Exception as e:
        print(f"❌ FATAL: could not load model: {e}")
        return

    # Disable live plot if running an automated batch to prevent pausing the loop
    if args.runs > 1 and args.live_plot:
        print("\n⚠️  Disabling live plot for unattended multi-run batch execution.")
        args.live_plot = False
                          
    print(f"\nExperiment configuration")
    print("=" * 72)
    print(f"Workload:             {workload}")
    print(f"Controller:           {args.controller}")
    print(f"Workload threads:     {args.cores}")
    print(f"Consecutive runs:     {args.runs}")
    print(f"Run mode:             {args.run_mode}")
    if args.run_mode == "duration":
        print(f"Duration T_exp:       {args.duration:.3f} s")
        print(f"Duration policy:      {args.duration_policy}")
    print(f"Live plot:            {'Yes' if args.live_plot else 'No'}")
    print(f"Live workload output: {'Yes' if args.show_workload_output else 'No'}")
    print("=" * 72)

    # Master execution loop for N runs
    for run_idx in range(1, args.runs + 1):
        if STOP_REQUESTED:
            print("Batch execution aborted by user.")
            break
            
        if args.runs > 1:
            print(f"\n{'='*72}")
            print(f"🚀 STARTING BATCH RUN {run_idx} OF {args.runs}")
            print(f"{'='*72}")

        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", workload.stem)
        
        # Create unique output directory for this specific run
        run_output_dir = ROOT / "results" / f"{safe_name}_{args.controller}_run{run_idx}_{run_timestamp}"
        if args.output_dir and args.runs == 1:
            run_output_dir = Path(args.output_dir).expanduser()

        runner = HistGBDTBenchmarkRunner(model=model, controller=args.controller,
                                         pcm_path=pcm_path, pcm_power_path=pcm_power_path,
                                         output_dir=run_output_dir, sample_sleep_s=args.sample_sleep,
                                         allow_simulated=args.allow_simulated,
                                         history_length=args.history_length,
                                         bw_ref_mb_s=args.bw_ref,
                                         show_workload_output=args.show_workload_output)
                                         
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

        if run_idx < args.runs and not STOP_REQUESTED:
            print(f"\n⏳ Run {run_idx} complete. Waiting 5 seconds before starting next run...")
            time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        print(f"\nFATAL EXPERIMENT ERROR: {exc}", file=sys.stderr)
        raise