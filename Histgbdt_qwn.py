#!/usr/bin/env python3
"""
HistGBDT-Driven DVFS Monitor (Windows-friendly, joblib-safe)
Updated with:
- Fixed bandwidth reference (25000 MB/s) to match training data.
- Enabled hysteresis to reduce state switching.
- Support for "Run to Completion" mode.
- Placeholder for Custom Low Power Plan GUID.
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

# Default deployment files
DEFAULT_MODEL_FILE  = str((ROOT / "models" / "histgbdt.joblib").resolve())
DEFAULT_SCALER_FILE = str((ROOT / "dataset" / "scaler_stats.npz").resolve())

# Rolling window for temporal features
N_ROLLING_SAMPLES = 5

# FIXED REFERENCE BANDWIDTH (Matches training scaler)
BW_REF = 25000.0 

# Power Plan GUIDs
# IMPORTANT: Replace the "Low" GUID with your custom plan's GUID!
# Create a plan in Windows: Min Processor State 40%, Max 80%.
# Run 'powercfg /list' in cmd to get the GUID.
POWER_PLANS = {
    "High":   "27ad8305-4092-41f2-a01b-5a6003fb5077",
    "Medium": "381b4222-f694-41f0-9685-ff5bb260df2e",
    "Low":    "YOUR_CUSTOM_LOW_PLAN_GUID_HERE" # <--- REPLACE THIS
}

# Control mode
USE_HGBDT_DVFS = False  # Set to True to enable ML-driven control

# Hysteresis settings (Enabled to reduce switching)
ENABLE_PROBA_HYSTERESIS = True
PROBA_MARGIN = 0.20  # Only switch if probability margin is >= 0.20

# =====================================================
def _best_effort_find_joblib(models_dir: Path) -> str | None:
    if not models_dir.exists():
        return None
    candidates = list(models_dir.glob("*.joblib"))
    if len(candidates) == 1:
        return str(candidates[0].resolve())
    hgb = [p for p in candidates if "hgb" in p.name.lower()]
    if len(hgb) == 1:
        return str(hgb[0].resolve())
    return None

# -----------------------------------------------------
# 1. HistGBDT Model Wrapper
# -----------------------------------------------------
class HistGBDTModel:
    FEATURE_ORDER = [
        "ipc", "l2_miss_rate", "l3_miss_rate", "memory_bandwidth",
        "cpu_usage_overall", "cpu_temperature", "cpu_power", "cpu_frequency",
        "ipc_change", "ipc_avg_5", "l3_miss_rate_avg_5", "bw_util",
    ]

    def __init__(self, model_path: str, scaler_path: str | None):
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        self.model = joblib.load(model_path)
        print(f"✅ Loaded HistGBDT model from: {model_path}")
        
        self.scaler_mean = None
        self.scaler_scale = None
        
        if scaler_path and os.path.exists(scaler_path):
            try:
                scaler_stats = np.load(scaler_path)
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
                    
                self.scaler_scale = np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)
                print(f"✅ Loaded scaler stats from: {scaler_path}")
            except Exception as e:
                print(f"⚠️ WARNING: Could not load scaler stats: {e}")
                self.scaler_mean = None
                self.scaler_scale = None
        else:
            print(f"⚠️ Scaler file not found. Running without normalization.")
            
        self.last_level = "Medium"
        self._init_class_mapping()

    def _init_class_mapping(self):
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            self.class_to_level = {0: "Low", 1: "Medium", 2: "High"}
            return
            
        classes_list = list(classes)
        if all(isinstance(c, str) for c in classes_list):
            self.class_to_level = {c: c for c in classes_list}
            return
            
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
        x = np.array([float(features_dict.get(k, 0.0)) for k in self.FEATURE_ORDER], dtype=np.float32)
        
        if self.scaler_mean is not None and self.scaler_scale is not None:
            if x.shape[0] == self.scaler_mean.shape[0]:
                x = (x - self.scaler_mean) / self.scaler_scale
                
        X = x.reshape(1, -1)
        
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
                pass
                
        cls = self.model.predict(X)[0]
        try:
            cls_key = int(cls)
        except Exception:
            cls_key = cls
            
        level = self.class_to_level.get(cls_key, "Medium")
        self.last_level = level
        return level

# -----------------------------------------------------
# 2. DVFS Controller Class
# -----------------------------------------------------
class HistGBDTDVFSController:
    def __init__(self, model: HistGBDTModel):
        self.running = True
        self.data_buffer = []
        self.model = model
        self.workload_process = None # Store process handle for completion check
        
        self.pcm_path = self.find_tool(PCM_PATHS)
        self.power_gadget_path = self.find_tool(POWER_GADGET_PATHS)
        
        self.fig, self.axs = plt.subplots(4, 2, figsize=(16, 12))
        self.fig.suptitle("HistGBDT-Driven DVFS Monitoring", fontsize=14, fontweight='bold')
        self.POWER_PLANS = POWER_PLANS.copy()

    def find_tool(self, possible_paths):
        for path in possible_paths:
            if path and os.path.exists(path):
                return path
        return None

    def set_power_plan(self, level):
        guid = self.POWER_PLANS.get(level)
        if not guid or guid == "YOUR_CUSTOM_LOW_PLAN_GUID_HERE":
            print(f"❌ Invalid or placeholder GUID for level: {level}")
            return False
        try:
            subprocess.run(["powercfg", "-setactive", guid], check=True, capture_output=True)
            return True
        except Exception as e:
            print(f"❌ Failed to set power plan: {e}")
            return False

    def simulate_pcm_data(self):
        return {
            'ipc': np.random.uniform(0.5, 2.0),
            'l2_cache_hits': 90000, 'l2_cache_misses': 10000,
            'l3_cache_hits': 85000, 'l3_cache_misses': 15000,
            'memory_bandwidth': np.random.uniform(10000, 25000),
        }

    def simulate_power_data(self):
        return {
            'cpu_power': np.random.uniform(40, 60),
            'gpu_power': np.random.uniform(5, 20),
            'cpu_temperature': np.random.uniform(50, 75),
            'cpu_frequency': np.random.uniform(2800, 3600)
        }

    def read_pcm_data(self):
        if not self.pcm_path:
            return self.simulate_pcm_data()
        try:
            result = subprocess.run([self.pcm_path, "1", "-nc", "-ns"], capture_output=True, text=True, timeout=5)
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                return self.simulate_pcm_data()
            parts = [x for x in lines[-1].strip().split(' ') if x]
            if len(parts) >= 8:
                return {
                    'ipc': float(parts[1]),
                    'memory_bandwidth': float(parts[2]),
                    'l2_cache_hits': int(parts[4]), 'l2_cache_misses': int(parts[5]),
                    'l3_cache_hits': int(parts[6]), 'l3_cache_misses': int(parts[7])
                }
        except Exception:
            pass
        return self.simulate_pcm_data()

    def read_power_gadget_data(self):
        if not self.power_gadget_path:
            return self.simulate_power_data()
        temp_file = "power_temp.csv"
        try:
            subprocess.run([self.power_gadget_path, "-duration", "1", "-file", temp_file], capture_output=True, timeout=5)
            if not os.path.exists(temp_file):
                return self.simulate_power_data()
            with open(temp_file, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                os.remove(temp_file)
                return self.simulate_power_data()
            data = lines[-1].strip().split(',')
            power_data = {
                'cpu_power': float(data[1]) if len(data) > 1 else 0.0,
                'gpu_power': float(data[2]) if len(data) > 2 else 0.0,
                'cpu_temperature': float(data[3]) if len(data) > 3 else 0.0,
                'cpu_frequency': float(data[4]) if len(data) > 4 else 0.0
            }
            os.remove(temp_file)
            return power_data
        except Exception:
            return self.simulate_power_data()

    def feature_engineer_data(self, data):
        df = pd.DataFrame(self.data_buffer + [data])
        data['ipc_change'] = float(df['ipc'].diff().iloc[-1]) if len(df) > 1 else 0.0
        data['ipc_avg_5'] = float(df['ipc'].rolling(window=N_ROLLING_SAMPLES, min_periods=1).mean().iloc[-1])
        data['l3_miss_rate_avg_5'] = float(df['l3_miss_rate'].rolling(window=N_ROLLING_SAMPLES, min_periods=1).mean().iloc[-1])
        
        # FIXED: Use constant reference bandwidth instead of dynamic max
        bw = float(data['memory_bandwidth'])
        bw_util = bw / BW_REF
        data['bw_util'] = float(min(max(bw_util, 0.0), 1.5))
        
        return data

    def collect_data(self):
        pcm_data = self.read_pcm_data()
        power_data = self.read_power_gadget_data()
        
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
        
        if USE_HGBDT_DVFS:
            level = self.model.predict_level(data)
            applied = self.set_power_plan(level)
        else:
            level = "Medium"
            applied = self.set_power_plan("Medium")
            
        data['dvfs_level'] = level
        data['dvfs_applied'] = bool(applied)
        return data

    def start_workload(self, script_name, num_cores):
        if USE_HGBDT_DVFS:
            print(f"💡 HistGBDT-based DVFS active for {script_name}")
        else:
            print(f"💡 Baseline fixed-Medium mode for {script_name}")
        try:
            print(f"🚀 Starting {script_name}...")
            # Store the process handle to check for completion later
            self.workload_process = subprocess.Popen([sys.executable, script_name])
            print("✅ Workload started.")
            return True
        except Exception as e:
            print(f"❌ Failed to start workload: {e}")
            return False

    def start_monitoring(self, workload_script, num_cores, duration, run_to_completion=False):
        print(f"🚀 Starting monitoring...")
        print(f"📊 Log: {LOG_FILE}")
        print(f"🔧 Mode: {'Run to Completion' if run_to_completion else f'Fixed Duration ({duration}s)'}")
        print("=" * 80)
        
        self.set_power_plan("Medium")
        start_time = time.time()
        tail_time = 10 # Seconds to log after workload finishes
        
        with open(LOG_FILE, 'w') as f:
            iteration = 0
            workload_finished = False
            finish_time = None
            
            while self.running and not STOP_REQUESTED:
                try:
                    iteration += 1
                    data = self.collect_data()
                    self.data_buffer.append(data)
                    f.write(json.dumps(data) + '\n')
                    f.flush()
                    
                    if iteration % 3 == 1:
                        print(f"[{iteration:3}] {datetime.now().strftime('%H:%M:%S')} | "
                              f"Level: {data['dvfs_level']:6} | IPC: {data['ipc']:4.2f} | "
                              f"Power: {data['cpu_power']:5.1f}W | Temp: {data['cpu_temperature']:4.1f}°C")
                    
                    # Logic for Run to Completion
                    if run_to_completion:
                        if self.workload_process and self.workload_process.poll() is not None:
                            if not workload_finished:
                                print("✅ Workload process finished.")
                                workload_finished = True
                                finish_time = time.time()
                        
                        if workload_finished and (time.time() - finish_time > tail_time):
                            print("️ Tail logging complete. Stopping monitor.")
                            break
                    else:
                        # Logic for Fixed Duration
                        if (time.time() - start_time >= duration):
                            break
                            
                    time.sleep(3)
                except Exception as e:
                    print(f"❌ Monitoring error: {e}")
                    time.sleep(3)
                    
        print("✅ Experiment completed.")

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
        _ = FuncAnimation(self.fig, self.update_plots, interval=3000, blit=False, cache_frame_data=False)
        plt.show()

# -----------------------------------------------------
# 3. Main Entry Point
# -----------------------------------------------------
def main():
    print("HistGBDT-Driven DVFS Monitor")
    print("=" * 60)
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_FILE, help="Path to HistGBDT .joblib")
    ap.add_argument("--scaler", default=DEFAULT_SCALER_FILE, help="Path to scaler_stats.npz")
    ap.add_argument("--completion", action="store_true", help="Run until workload script finishes")
    args = ap.parse_args()
    
    model_path = args.model
    if not os.path.exists(model_path):
        recovered = _best_effort_find_joblib(ROOT / "models")
        if recovered:
            model_path = recovered
        else:
            print(f"❌ MODEL FILE NOT FOUND: {model_path}")
            return
            
    scaler_path = args.scaler if args.scaler and os.path.exists(args.scaler) else None
    
    try:
        model = HistGBDTModel(model_path=model_path, scaler_path=scaler_path)
    except Exception as e:
        print(f"❌ FATAL: could not load model: {e}")
        return
        
    scripts = []
    if os.path.exists("cpu_intensive.py"): scripts.append("cpu_intensive.py")
    if os.path.exists("memory_intensive.py"): scripts.append("memory_intensive.py")
    if os.path.exists("mixed_workload.py"): scripts.append("mixed_workload.py")
    
    if not scripts:
        print("❌ No workload scripts found.")
        return
        
    while True:
        choice = input(f"Enter workload choice (1-{len(scripts)}): ").strip()
        if choice in [str(i + 1) for i in range(len(scripts))]:
            workload_script = scripts[int(choice) - 1]
            break
            
    while True:
        num_cores_input = input("Enter number of cores (1-8) [4]: ").strip()
        if not num_cores_input:
            num_cores = 4
            break
        if num_cores_input.isdigit() and 1 <= int(num_cores_input) <= 8:
            num_cores = int(num_cores_input)
            break
            
    duration = 300
    if not args.completion:
        while True:
            duration_input = input(f"Enter duration in seconds {DURATION_OPTIONS} [{DURATION_OPTIONS[0]}]: ").strip()
            if not duration_input:
                duration = DURATION_OPTIONS[0]
                break
            if duration_input.isdigit() and int(duration_input) in DURATION_OPTIONS:
                duration = int(duration_input)
                break
                
    monitor = HistGBDTDVFSController(model)
    if not monitor.start_workload(workload_script, num_cores):
        return
        
    monitoring_thread = threading.Thread(
        target=monitor.start_monitoring,
        args=(workload_script, num_cores, duration, args.completion),
        daemon=True
    )
    monitoring_thread.start()
    
    try:
        monitor.start_realtime_plot()
    except Exception as e:
        print(f"❌ Plotting error: {e}")
        
    monitoring_thread.join()
    print(f"📄 Results saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()