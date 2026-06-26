#!/usr/bin/env python3
"""
HistGBDT-Driven DVFS Monitor (Windows-friendly, joblib-safe)

Fixes the current hgbdt_rpm_runtime.py:
- No hardcoded absolute paths for MODEL_FILE / SCALER_FILE.
- Doesn't crash at import time when the model is missing.
- Clear instructions on how to create the missing .joblib (run train_export_histgbdt.py).
- Optional CLI overrides: --model / --scaler

Expected project layout:
  <project_root>/
    hgbdt_rpm_runtime_fixed.py   (this file)
    train_export_histgbdt.py
    dataset/
      X_train.npy, y_train.npy, X_test.npy, y_test.npy, scaler_stats.npz
    models/
      histgbdt.joblib
"""
import os
import json
import time
import threading
import subprocess
import psutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import sys
import pandas as pd
import re
import signal
import argparse
from pathlib import Path

import joblib

# ============ GLOBAL STOP FLAG & CTRL+C HANDLER ============
STOP_REQUESTED = False

def handle_ctrl_c(sig, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n⚠️ Ctrl+C detected — requesting shutdown...")

signal.signal(signal.SIGINT, handle_ctrl_c)
# ===========================================================

ROOT = Path(__file__).resolve().parent

# =================== CONFIGURATION & CONSTANTS ===================
LOG_FILE = str((ROOT / f"monitoring_log_HGBDT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json").resolve())

PCM_PATHS = [
    # Prefer local copy if you place pcm.exe next to the script
    str((ROOT / "pcm.exe").resolve()),
    r"C:\Users\saidm\OneDrive\Desktop\Programming\pcm\build\bin\Release\pcm.exe",
    "pcm.exe"
]
POWER_GADGET_PATHS = [
    str((ROOT / "pcm-power.exe").resolve()),
    r"C:\Users\saidm\OneDrive\Desktop\Programming\pcm\build\bin\Release\pcm-power.exe",
    "pcm-power.exe"
]
DURATION_OPTIONS = [100, 200, 300]

# Default deployment files (relative to project root)
DEFAULT_MODEL_FILE  = str((ROOT / "models" / "histgbdt.joblib").resolve())
DEFAULT_SCALER_FILE = str((ROOT / "dataset" / "scaler_stats.npz").resolve())

# Rolling window for temporal features (must match training: 5)
N_ROLLING_SAMPLES = 5

# Power Plan GUIDs
POWER_PLANS = {
    "High":   "27ad8305-4092-41f2-a01b-5a6003fb5077",
    "Medium": "381b4222-f694-41f0-9685-ff5bb260df2e",
    "Low":    "b2524225-86dc-424d-ba5b-e78d683c5d3a"
}

# Control mode: True = HistGBDT-driven DVFS, False = baseline fixed "Medium"
USE_HGBDT_DVFS = False  # Set to True to enable the ML-driven DVFS control logic

# Optional anti-thrashing:
ENABLE_PROBA_HYSTERESIS = False
PROBA_MARGIN = 0.15  # only switch if top1-top2 >= this margin
# =====================================================


def _best_effort_find_joblib(models_dir: Path) -> str | None:
    """If histgbdt.joblib is missing, try to find a single .joblib in models/."""
    if not models_dir.exists():
        return None
    candidates = list(models_dir.glob("*.joblib"))
    if len(candidates) == 1:
        return str(candidates[0].resolve())
    # Prefer anything with 'hgb' in the name
    hgb = [p for p in candidates if "hgb" in p.name.lower()]
    if len(hgb) == 1:
        return str(hgb[0].resolve())
    return None


# -----------------------------------------------------
# 1. HistGBDT Model Wrapper (MODEL + SCALER LOADING)
# -----------------------------------------------------
class HistGBDTModel:
    """Encapsulates the trained HistGradientBoostingClassifier for real-time inference."""

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
        "ipc_avg_5",
        "l3_miss_rate_avg_5",
        "bw_util",
    ]

    def __init__(self, model_path: str, scaler_path: str | None):
        self.model_path = model_path
        self.scaler_path = scaler_path

        # --- LOAD MODEL ---
        self.model = joblib.load(model_path)
        print(f"✅ Loaded HistGBDT model from: {model_path}")

        # --- LOAD SCALER STATS (mean & scale from training) ---
        self.scaler_mean = None
        self.scaler_scale = None
        if scaler_path and os.path.exists(scaler_path):
            try:
                scaler_stats = np.load(scaler_path)

                # Support both conventions: (mean, scale) or (mean_, scale_)
                if "mean" in scaler_stats:
                    self.scaler_mean = scaler_stats["mean"].astype(np.float32)
                elif "mean_" in scaler_stats:
                    self.scaler_mean = scaler_stats["mean_"].astype(np.float32)

                if "scale" in scaler_stats:
                    self.scaler_scale = scaler_stats["scale"].astype(np.float32)
                elif "scale_" in scaler_stats:
                    self.scaler_scale = scaler_stats["scale_"].astype(np.float32)

                if self.scaler_mean is None or self.scaler_scale is None:
                    raise KeyError(f"Scaler keys not found. Keys present: {list(scaler_stats.keys())}")

                # avoid div-by-zero
                self.scaler_scale = np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)

                print(f"✅ Loaded scaler stats from: {scaler_path}")

            except Exception as e:
                print(f"⚠️ WARNING: Could not load scaler stats from {scaler_path}: {e}")
                print("   Model will run WITHOUT normalization. Predictions may be wrong.")
                self.scaler_mean = None
                self.scaler_scale = None
        else:
            if scaler_path:
                print(f"⚠️ Scaler file not found at: {scaler_path}. Running without normalization.")

        self.last_level = "Medium"

        # Derive class mapping safely
        self._init_class_mapping()

    def _init_class_mapping(self):
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            # Default expectation
            self.class_to_level = {0: "Low", 1: "Medium", 2: "High"}
            return

        classes_list = list(classes)
        # If the model was trained on strings already
        if all(isinstance(c, str) for c in classes_list):
            # Expecting "Low"/"Medium"/"High"
            self.class_to_level = {c: c for c in classes_list}
            return

        # Numeric classes: map by value if it's the standard {0,1,2}, else by sorted order
        try:
            cls_int = [int(c) for c in classes_list]
            if set(cls_int) == {0, 1, 2}:
                self.class_to_level = {0: "Low", 1: "Medium", 2: "High"}
            else:
                cls_sorted = sorted(cls_int)
                levels = ["Low", "Medium", "High"]
                self.class_to_level = {cls_sorted[i]: levels[i] for i in range(min(3, len(cls_sorted)))}
        except Exception:
            self.class_to_level = {0: "Low", 1: "Medium", 2: "High"}

    def predict_level(self, features_dict):
        """
        Build the 12-D feature vector exactly as used in training:
          [ipc, l2_miss_rate, l3_miss_rate, memory_bandwidth,
           cpu_usage_overall, cpu_temperature, cpu_power, cpu_frequency,
           ipc_change, ipc_avg_5, l3_miss_rate_avg_5, bw_util]
        """
        x = np.array([float(features_dict.get(k, 0.0)) for k in self.FEATURE_ORDER], dtype=np.float32)

        # Apply scaler if available
        if self.scaler_mean is not None and self.scaler_scale is not None:
            if x.shape[0] == self.scaler_mean.shape[0]:
                x = (x - self.scaler_mean) / self.scaler_scale

        X = x.reshape(1, -1)

        # Optional hysteresis based on probability margin
        if ENABLE_PROBA_HYSTERESIS and hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(X)[0]
                sorted_idx = np.argsort(proba)[::-1]
                top1 = sorted_idx[0]
                margin = float(proba[sorted_idx[0]] - proba[sorted_idx[1]])
                new_level = self.class_to_level.get(int(top1) if not isinstance(top1, str) else top1, "Medium")

                if new_level != self.last_level and margin < PROBA_MARGIN:
                    return self.last_level

                self.last_level = new_level
                return new_level
            except Exception:
                # Fall back to predict if proba fails
                pass

        # Default predict
        cls = self.model.predict(X)[0]
        # normalize key type
        try:
            cls_key = int(cls)
        except Exception:
            cls_key = cls
        level = self.class_to_level.get(cls_key, "Medium")
        self.last_level = level
        return level


# -----------------------------------------------------
# 2. DVFS Controller Class (Main Control Logic)
# -----------------------------------------------------
class HistGBDTDVFSController:

    def __init__(self, model: HistGBDTModel):
        self.running = True
        self.data_buffer = []
        self.max_bw_observed = 25000.0  # MB/s heuristic; updated online
        self.model = model

        self.pcm_path = self.find_tool(PCM_PATHS)
        self.power_gadget_path = self.find_tool(POWER_GADGET_PATHS)

        self.fig, self.axs = plt.subplots(4, 2, figsize=(16, 12))
        self.fig.suptitle("HistGBDT-Driven DVFS Monitoring", fontsize=14, fontweight='bold')

        self.POWER_PLANS = POWER_PLANS.copy()

    # ---------- Utility ----------
    def find_tool(self, possible_paths):
        for path in possible_paths:
            if path and os.path.exists(path):
                print(f"✅ Found: {path}")
                return path
        print(f"⚠️ Tool not found in any of: {possible_paths}")
        return None

    def set_power_plan(self, level):
        guid = self.POWER_PLANS.get(level)
        if not guid:
            print(f"❌ Invalid level: {level}")
            return False
        try:
            subprocess.run(["powercfg", "-setactive", guid],
                           check=True, capture_output=True)
            return True
        except Exception as e:
            print(f"❌ Failed to set power plan: {e}")
            return False

    # ---------- Simulation fallback ----------
    def simulate_pcm_data(self):
        ipc = np.random.uniform(0.5, 2.0)
        l3_miss_rate = np.random.uniform(0.01, 0.20)
        mem_bw = np.random.uniform(10000, 25000)

        total_accesses = 100000
        l3_misses = int(total_accesses * l3_miss_rate)
        l3_hits = total_accesses - l3_misses
        l2_misses = int(total_accesses * np.random.uniform(0.05, 0.15))
        l2_hits = total_accesses - l2_misses

        return {
            'ipc': ipc,
            'l2_cache_hits': l2_hits,
            'l2_cache_misses': l2_misses,
            'l3_cache_hits': l3_hits,
            'l3_cache_misses': l3_misses,
            'memory_bandwidth': mem_bw,
        }

    def simulate_power_data(self):
        # Simple heuristic based on active power plan
        try:
            result = subprocess.run(["powercfg", "/getactivescheme"],
                                    capture_output=True, text=True, check=True)
            match = re.search(r'\((.*?)\)', result.stdout)
            active_plan_name = match.group(1) if match else "Balanced"
        except Exception:
            active_plan_name = "Balanced"

        if "High" in active_plan_name:
            power = np.random.uniform(70, 90)
            freq = np.random.uniform(3800, 4200)
        elif "Power" in active_plan_name or "Low" in active_plan_name:
            power = np.random.uniform(20, 30)
            freq = np.random.uniform(1200, 1800)
        else:  # Balanced / Medium
            power = np.random.uniform(40, 60)
            freq = np.random.uniform(2800, 3600)

        return {
            'cpu_power': power,
            'gpu_power': np.random.uniform(5, 20),
            'cpu_temperature': np.random.uniform(50, 75),
            'cpu_frequency': freq
        }

    # ---------- REAL PCM + POWER (with fallback) ----------
    def read_pcm_data(self):
        """Try to read from Intel PCM; fall back to simulation on failure."""
        if not self.pcm_path:
            return self.simulate_pcm_data()

        try:
            result = subprocess.run(
                [self.pcm_path, "1", "-nc", "-ns"],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                return self.simulate_pcm_data()

            data_line = lines[-1].strip()
            parts = [x for x in data_line.split(' ') if x]

            # WARNING: parsing depends on your pcm.exe output format.
            # Adjust indices if needed.
            if len(parts) >= 8:
                ipc = float(parts[1])
                mem_bw = float(parts[2])      # MB/s
                l2_hits = int(parts[4])
                l2_misses = int(parts[5])
                l3_hits = int(parts[6])
                l3_misses = int(parts[7])

                return {
                    'ipc': ipc,
                    'l2_cache_hits': l2_hits,
                    'l2_cache_misses': l2_misses,
                    'l3_cache_hits': l3_hits,
                    'l3_cache_misses': l3_misses,
                    'memory_bandwidth': mem_bw
                }
        except Exception as e:
            print(f"PCM Error: {e}")
        return self.simulate_pcm_data()

    def read_power_gadget_data(self):
        """Try to read from pcm-power.exe; fall back to simulation on failure."""
        if not self.power_gadget_path:
            return self.simulate_power_data()

        temp_file = "power_temp.csv"
        cmd = [self.power_gadget_path, "-duration", "1", "-file", temp_file]

        try:
            subprocess.run(cmd, capture_output=True, timeout=5)
            if not os.path.exists(temp_file):
                return self.simulate_power_data()

            with open(temp_file, 'r') as f:
                lines = f.readlines()
                if len(lines) < 2:
                    os.remove(temp_file)
                    return self.simulate_power_data()

                data = lines[-1].strip().split(',')
                power_data = {
                    'cpu_power': float(data[1]) if len(data) > 1 and data[1] else 0.0,
                    'gpu_power': float(data[2]) if len(data) > 2 and data[2] else 0.0,
                    'cpu_temperature': float(data[3]) if len(data) > 3 and data[3] else 0.0,
                    'cpu_frequency': float(data[4]) if len(data) > 4 and data[4] else 0.0
                }
                os.remove(temp_file)
                return power_data
        except Exception as e:
            print(f"Power Gadget Error: {e}")
        return self.simulate_power_data()

    # ---------- Feature Engineering ----------
    def feature_engineer_data(self, data):
        """
        Compute temporal features to match training:
          - ipc_change
          - ipc_avg_5
          - l3_miss_rate_avg_5
          - bw_util
        """
        df = pd.DataFrame(self.data_buffer + [data])

        # ipc_change
        data['ipc_change'] = float(df['ipc'].diff().iloc[-1]) if len(df) > 1 else 0.0

        # rolling IPC avg
        data['ipc_avg_5'] = float(df['ipc'].rolling(window=N_ROLLING_SAMPLES, min_periods=1).mean().iloc[-1])

        # rolling L3 miss avg
        data['l3_miss_rate_avg_5'] = float(df['l3_miss_rate'].rolling(window=N_ROLLING_SAMPLES, min_periods=1).mean().iloc[-1])

        # BW util
        bw = float(data['memory_bandwidth'])
        self.max_bw_observed = max(self.max_bw_observed, bw, 1e-3)
        bw_util = bw / self.max_bw_observed
        data['bw_util'] = float(min(max(bw_util, 0.0), 1.5))

        return data

    # ---------- Data Collection + DVFS Logic ----------
    def collect_data(self):
        pcm_data = self.read_pcm_data()
        power_data = self.read_power_gadget_data()

        # L2/L3 miss rates
        total_l2 = pcm_data['l2_cache_hits'] + pcm_data['l2_cache_misses']
        total_l3 = pcm_data['l3_cache_hits'] + pcm_data['l3_cache_misses']
        l2_miss_rate = pcm_data['l2_cache_misses'] / total_l2 if total_l2 > 0 else 0.0
        l3_miss_rate = pcm_data['l3_cache_misses'] / total_l3 if total_l3 > 0 else 0.0

        data = {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage_overall': psutil.cpu_percent(interval=0.1),
            'memory_usage': psutil.virtual_memory().percent,
            'ipc': float(pcm_data['ipc']),
            'l2_miss_rate': round(float(l2_miss_rate), 4),
            'l3_miss_rate': round(float(l3_miss_rate), 4),
            'memory_bandwidth': float(pcm_data['memory_bandwidth']),
            'cpu_power': round(float(power_data['cpu_power']), 1),
            'cpu_temperature': round(float(power_data['cpu_temperature']), 1),
            'cpu_frequency': round(float(power_data['cpu_frequency']), 1)
        }

        data = self.feature_engineer_data(data)

        # DVFS
        if USE_HGBDT_DVFS:
            level = self.model.predict_level(data)
            applied = self.set_power_plan(level)
        else:
            level = "Medium"
            applied = self.set_power_plan("Medium")

        data['dvfs_level'] = level
        data['dvfs_applied'] = bool(applied)
        return data

    # ---------- Workload Control ----------
    def start_workload(self, script_name, num_cores):
        if USE_HGBDT_DVFS:
            print(f"💡 HistGBDT-based DVFS active for {script_name}")
        else:
            print(f"💡 Baseline fixed-Medium mode for {script_name}")

        try:
            print(f"🚀 Starting {script_name} on {num_cores} core(s)...")
            subprocess.Popen([sys.executable, script_name])
            print("✅ Workload started.")
            return True
        except Exception as e:
            print(f"❌ Failed to start workload: {e}")
            return False

    # ---------- Monitoring Loop ----------
    def start_monitoring(self, workload_script, num_cores, duration):
        print(f"🚀 Starting monitoring for {duration} seconds...")
        print(f"📊 Log: {LOG_FILE}")
        print(f"🔧 DVFS Mode: {'HGBDT-DRIVEN DVFS' if USE_HGBDT_DVFS else 'Baseline (Fixed Medium)'}")
        print("=" * 80)

        # Set a sane initial state
        self.set_power_plan("Medium")

        start_time = time.time()
        with open(LOG_FILE, 'w') as f:
            iteration = 0
            while self.running and not STOP_REQUESTED and (time.time() - start_time < duration):
                try:
                    iteration += 1
                    data = self.collect_data()
                    self.data_buffer.append(data)

                    f.write(json.dumps(data) + '\n')
                    f.flush()

                    if iteration % 3 == 1:
                        print(f"[{iteration:3}] {datetime.now().strftime('%H:%M:%S')} | "
                              f"Level: {data['dvfs_level']:6} | IPC: {data['ipc']:4.2f} | "
                              f"Power: {data['cpu_power']:5.1f}W | Temp: {data['cpu_temperature']:4.1f}°C | "
                              f"CPU: {data['cpu_usage_overall']:5.1f}% | L3 Miss: {data['l3_miss_rate']:5.3f}")

                    time.sleep(3)

                except Exception as e:
                    print(f"❌ Monitoring error: {e}")
                    time.sleep(3)

        print("⏹️ Workload stopped.")
        print("✅ Experiment completed. Results logged.")

    # ---------- Plotting ----------
    def update_plots(self, frame):
        if not self.data_buffer:
            return

        df = pd.DataFrame(self.data_buffer)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['time_min'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 60

        for ax_row in self.axs:
            for ax in ax_row:
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
                ax = self.axs[idx // 2, idx % 2]
                ax.plot(df['time_min'], df[col], alpha=0.8, linewidth=1.5)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

    def start_realtime_plot(self):
        print("📊 Starting real-time visualization...")
        _ = FuncAnimation(self.fig, self.update_plots,
                          interval=3000, blit=False, cache_frame_data=False)
        plt.show()


# -----------------------------------------------------
# 3. Main Entry Point
# -----------------------------------------------------
def main():
    print("HistGBDT-Driven DVFS Monitor")
    print("=" * 60)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_FILE, help="Path to HistGBDT .joblib")
    ap.add_argument("--scaler", default=DEFAULT_SCALER_FILE, help="Path to scaler_stats.npz (optional)")
    args = ap.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        # Best-effort recovery: try models/*.joblib
        recovered = _best_effort_find_joblib(ROOT / "models")
        if recovered:
            print(f"⚠️ MODEL FILE NOT FOUND at: {model_path}")
            print(f"✅ Found a plausible model instead: {recovered}")
            model_path = recovered
        else:
            print(f"❌ MODEL FILE NOT FOUND: {model_path}")
            print("Fix:")
            print("  1) Export the model:  python train_export_histgbdt.py")
            print("  2) Confirm you have:  models/histgbdt.joblib")
            print("  3) Then run:          python hgbdt_rpm_runtime_fixed.py")
            return

    scaler_path = args.scaler if args.scaler and os.path.exists(args.scaler) else None

    try:
        model = HistGBDTModel(model_path=model_path, scaler_path=scaler_path)
    except Exception as e:
        print(f"❌ FATAL: could not load model: {e}")
        return

    # Discover workloads
    scripts = []
    if os.path.exists("cpu_intensive.py"):
        print(" 1. cpu_intensive.py")
        scripts.append("cpu_intensive.py")
    if os.path.exists("memory_intensive.py"):
        print(" 2. memory_intensive.py")
        scripts.append("memory_intensive.py")
    if os.path.exists("mixed_workload.py"):
        print(" 3. mixed_workload.py")
        scripts.append("mixed_workload.py")

    if not scripts:
        print("❌ No workload scripts found. Place cpu_intensive.py or memory_intensive.py here.")
        return

    # Workload choice
    while True:
        choice = input(f"Enter workload choice (1-{len(scripts)}): ").strip()
        if choice in [str(i + 1) for i in range(len(scripts))]:
            workload_script = scripts[int(choice) - 1]
            break
        print(f"Please enter 1 to {len(scripts)}.")

    # Core count
    while True:
        num_cores_input = input("Enter number of cores (1-8) [4]: ").strip()
        if not num_cores_input:
            num_cores = 4
            break
        if num_cores_input.isdigit() and 1 <= int(num_cores_input) <= 8:
            num_cores = int(num_cores_input)
            break
        print("Please enter a number between 1 and 8.")

    # Duration
    while True:
        duration_input = input(f"Enter duration in seconds {DURATION_OPTIONS} [{DURATION_OPTIONS[0]}]: ").strip()
        if not duration_input:
            duration = DURATION_OPTIONS[0]
            break
        if duration_input.isdigit() and int(duration_input) in DURATION_OPTIONS:
            duration = int(duration_input)
            break
        print(f"Please enter one of: {DURATION_OPTIONS}")

    monitor = HistGBDTDVFSController(model)

    if not monitor.start_workload(workload_script, num_cores):
        print("❌ Could not start workload.")
        return

    monitoring_thread = threading.Thread(
        target=monitor.start_monitoring,
        args=(workload_script, num_cores, duration),
        daemon=True
    )
    monitoring_thread.start()

    try:
        monitor.start_realtime_plot()
    except Exception as e:
        print(f"❌ Plotting error: {e}")

    monitoring_thread.join()
    print("✅ All analysis complete.")
    print(f"📄 Results saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
