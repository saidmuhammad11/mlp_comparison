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
import joblib

# ============ GLOBAL STOP FLAG & CTRL+C HANDLER ============
STOP_REQUESTED = False

def handle_ctrl_c(sig, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n⚠️ Ctrl+C detected — requesting shutdown...")

signal.signal(signal.SIGINT, handle_ctrl_c)
# ===========================================================

# =================== CONFIGURATION & CONSTANTS ===================
LOG_FILE = f"monitoring_log_DT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

PCM_PATHS = [
    r"C:\Users\saidm\OneDrive\Desktop\Programming\pcm\build\bin\Release\pcm.exe",
    "pcm.exe"
]
POWER_GADGET_PATHS = [
    r"C:\Users\saidm\OneDrive\Desktop\Programming\pcm\build\bin\Release\pcm-power.exe",
    "pcm-power.exe"
]
DURATION_OPTIONS = [100, 200, 300]

# Deployment files
MODEL_FILE  = r"C:\Users\saidm\OneDrive\Desktop\Programming\MLP_test2\DecisionTree.joblib"
SCALER_FILE = r"C:\Users\saidm\OneDrive\Desktop\Programming\MLP_test2\dvfs_scaler_stats.npz"

# Rolling window for temporal features (must match training: 5)
N_ROLLING_SAMPLES = 5

# Power Plan GUIDs
POWER_PLANS = {
    "High":   "27ad8305-4092-41f2-a01b-5a6003fb5077",
    "Medium": "381b4222-f694-41f0-9685-ff5bb260df2e",
    "Low":    "b2524225-86dc-424d-ba5b-e78d683c5d3a"
}

# Control mode: True = DT-driven DVFS, False = baseline fixed "Medium"
USE_DT_DVFS = True

# =====================================================


# -----------------------------------------------------
# 1. Decision Tree Model Wrapper (MODEL + SCALER LOADING)
# -----------------------------------------------------
class DecisionTreeModel:
    """Encapsulates the trained DecisionTreeClassifier for real-time inference."""

    def __init__(self):
        # --- LOAD MODEL ---
        try:
            self.model = joblib.load(MODEL_FILE)
            print(f"✅ Loaded Decision Tree model from {MODEL_FILE}")
        except Exception as e:
            print(f"❌ FATAL ERROR: Could not load model file {MODEL_FILE}: {e}")
            raise

        # --- LOAD SCALER STATS ---
        self.scaler_mean = None
        self.scaler_scale = None
        try:
            scaler_stats = np.load(SCALER_FILE)
            # Use keys from 2_build_dataset.py
            self.scaler_mean = scaler_stats.get("mean_", scaler_stats.get("mean")).astype(np.float32)
            self.scaler_scale = scaler_stats.get("scale_", scaler_stats.get("scale")).astype(np.float32)
            print(f"✅ Loaded scaler stats from {SCALER_FILE}")
        except Exception as e:
            print(f"⚠️ WARNING: Scaler error: {e}")

        self.LEVEL_MAP = {0: "Low", 1: "Medium", 2: "High"}

    def predict_level(self, features_dict):
        """
        Build the 12-D feature vector exactly as used in training:
        [ipc, mr_l2, mr_l3, bw, u, tau, p, f, dipc, ipc_avg, mr3_avg, bw_util]
        """
        x = np.array([
            features_dict.get("ipc", 0.0),
            features_dict.get("l2_miss_rate", 0.0),
            features_dict.get("l3_miss_rate", 0.0),
            features_dict.get("memory_bandwidth", 0.0),
            features_dict.get("cpu_usage_overall", 0.0),
            features_dict.get("cpu_temperature", 0.0),
            features_dict.get("cpu_power", 0.0),
            features_dict.get("cpu_frequency", 0.0),
            features_dict.get("ipc_change", 0.0),
            features_dict.get("ipc_avg_5", 0.0),
            features_dict.get("l3_miss_rate_avg_5", 0.0),
            features_dict.get("bw_util", 0.0),
        ], dtype=np.float32)

        if self.scaler_mean is not None:
            x = (x - self.scaler_mean) / self.scaler_scale

        X = x.reshape(1, -1)
        cls = int(self.model.predict(X)[0])
        return self.LEVEL_MAP.get(cls, "Medium")

dt_controller = DecisionTreeModel()


# -----------------------------------------------------
# 2. DVFS Controller Class (Main Control Logic)
# -----------------------------------------------------
class DecisionTreeDVFSController:

    def __init__(self):
        self.running = True
        self.data_buffer = []
        self.max_bw_observed = 25000.0  # updated online

        self.pcm_path = self.find_tool(PCM_PATHS)
        self.power_gadget_path = self.find_tool(POWER_GADGET_PATHS)

        self.fig, self.axs = plt.subplots(4, 2, figsize=(16, 12))
        self.fig.suptitle("Decision Tree-Driven DVFS Monitoring", fontsize=14, fontweight='bold')

        self.POWER_PLANS = POWER_PLANS.copy()

    def find_tool(self, possible_paths):
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found: {path}")
                return path
        return None

    def set_power_plan(self, level):
        guid = self.POWER_PLANS.get(level)
        if not guid: return False
        try:
            subprocess.run(["powercfg", "-setactive", guid], check=True, capture_output=True)
            return True
        except Exception as e:
            print(f"❌ PowerCfg Error: {e}")
            return False

    # ---------- Real/Simulated Data Handlers ----------
    def simulate_pcm_data(self):
        ipc = np.random.uniform(0.5, 2.0)
        l3_miss_rate = np.random.uniform(0.01, 0.20)
        total = 100000
        l3_m = int(total * l3_miss_rate)
        return {
            'ipc': ipc, 'l2_cache_hits': 90000, 'l2_cache_misses': 5000,
            'l3_cache_hits': total - l3_m, 'l3_cache_misses': l3_m,
            'memory_bandwidth': np.random.uniform(10000, 25000)
        }

    def simulate_power_data(self):
        return {'cpu_power': 50.0, 'gpu_power': 10.0, 'cpu_temperature': 65.0, 'cpu_frequency': 3000.0}

    def read_pcm_data(self):
        if not self.pcm_path: return self.simulate_pcm_data()
        try:
            result = subprocess.run([self.pcm_path, "1", "-nc", "-ns"], capture_output=True, text=True, timeout=5)
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2: return self.simulate_pcm_data()
            parts = [x for x in lines[-1].split(' ') if x]
            if len(parts) >= 8:
                return {
                    'ipc': float(parts[1]), 'memory_bandwidth': float(parts[2]),
                    'l2_cache_hits': int(parts[4]), 'l2_cache_misses': int(parts[5]),
                    'l3_cache_hits': int(parts[6]), 'l3_cache_misses': int(parts[7])
                }
        except Exception: pass
        return self.simulate_pcm_data()

    def read_power_gadget_data(self):
        if not self.power_gadget_path: return self.simulate_power_data()
        temp_file = "power_temp_dt.csv"
        try:
            subprocess.run([self.power_gadget_path, "-duration", "1", "-file", temp_file], capture_output=True, timeout=5)
            if not os.path.exists(temp_file): return self.simulate_power_data()
            with open(temp_file, 'r') as f:
                data = f.readlines()[-1].strip().split(',')
                power_data = {'cpu_power': float(data[1]), 'gpu_power': float(data[2]), 
                              'cpu_temperature': float(data[3]), 'cpu_frequency': float(data[4])}
            os.remove(temp_file)
            return power_data
        except Exception: return self.simulate_power_data()

    # ---------- Feature Engineering ----------
    def feature_engineer_data(self, data):
        df = pd.DataFrame(self.data_buffer + [data])
        data['ipc_change'] = float(df['ipc'].diff().iloc[-1]) if len(df) > 1 else 0.0
        data['ipc_avg_5'] = float(df['ipc'].rolling(window=N_ROLLING_SAMPLES, min_periods=1).mean().iloc[-1])
        data['l3_miss_rate_avg_5'] = float(df['l3_miss_rate'].rolling(window=N_ROLLING_SAMPLES, min_periods=1).mean().iloc[-1])
        bw = float(data['memory_bandwidth'])
        self.max_bw_observed = max(self.max_bw_observed, bw, 1e-3)
        data['bw_util'] = float(min(max(bw / self.max_bw_observed, 0.0), 1.5))
        return data

    def collect_data(self):
        pcm = self.read_pcm_data()
        pwr = self.read_power_gadget_data()
        t_l2 = pcm['l2_cache_hits'] + pcm['l2_cache_misses']
        t_l3 = pcm['l3_cache_hits'] + pcm['l3_cache_misses']
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage_overall': psutil.cpu_percent(interval=0.1),
            'ipc': float(pcm['ipc']),
            'l2_miss_rate': pcm['l2_cache_misses'] / t_l2 if t_l2 > 0 else 0.0,
            'l3_miss_rate': pcm['l3_cache_misses'] / t_l3 if t_l3 > 0 else 0.0,
            'memory_bandwidth': float(pcm['memory_bandwidth']),
            'cpu_power': float(pwr['cpu_power']),
            'cpu_temperature': float(pwr['cpu_temperature']),
            'cpu_frequency': float(pwr['cpu_frequency'])
        }
        data = self.feature_engineer_data(data)

        if USE_DT_DVFS:
            level = dt_controller.predict_level(data)
            applied = self.set_power_plan(level)
        else:
            level = "Medium"
            applied = self.set_power_plan("Medium")

        data['dvfs_level'] = level
        data['dvfs_applied'] = bool(applied)
        return data

    def start_workload(self, script_name, num_cores):
        print(f"🚀 Starting {script_name} on {num_cores} cores...")
        try:
            subprocess.Popen([sys.executable, script_name])
            return True
        except: return False

    def start_monitoring(self, workload_script, num_cores, duration):
        print(f"📊 Log: {LOG_FILE} | Mode: DT-DRIVEN")
        self.set_power_plan("Medium")
        start_time = time.time()
        with open(LOG_FILE, 'w') as f:
            while not STOP_REQUESTED and (time.time() - start_time < duration):
                try:
                    data = self.collect_data()
                    self.data_buffer.append(data)
                    f.write(json.dumps(data) + '\n')
                    f.flush()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] DT Level: {data['dvfs_level']:6} | IPC: {data['ipc']:4.2f} | Power: {data['cpu_power']:5.1f}W")
                    time.sleep(3)
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(3)

    def update_plots(self, frame):
        if not self.data_buffer: return
        df = pd.DataFrame(self.data_buffer)
        df['time_min'] = (pd.to_datetime(df['timestamp']) - pd.to_datetime(df['timestamp'].iloc[0])).dt.total_seconds() / 60
        for ax_row in self.axs: 
            for ax in ax_row: ax.clear()
        
        plots = [('cpu_power', 'Power (W)'), ('ipc', 'IPC'), ('cpu_frequency', 'Freq (MHz)'), 
                 ('cpu_temperature', 'Temp (°C)'), ('l2_miss_rate', 'L2 MR'), ('l3_miss_rate', 'L3 MR'), 
                 ('cpu_usage_overall', 'CPU %'), ('memory_bandwidth', 'BW (MB/s)')]
        
        for idx, (col, lbl) in enumerate(plots):
            ax = self.axs[idx // 2, idx % 2]
            ax.plot(df['time_min'], df[col], alpha=0.8)
            ax.set_ylabel(lbl)
            ax.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    def start_realtime_plot(self):
        ani = FuncAnimation(self.fig, self.update_plots, interval=3000, cache_frame_data=False)
        plt.show()

# -----------------------------------------------------
# 3. Main Entry Point (Workload Selector)
# -----------------------------------------------------
def main():
    print("Decision Tree-Driven DVFS Monitor")
    scripts = [s for s in ["cpu_intensive.py", "memory_intensive.py"] if os.path.exists(s)]
    if not scripts: return
    
    for i, s in enumerate(scripts): print(f" {i+1}. {s}")
    choice = int(input(f"Choice (1-{len(scripts)}): "))
    workload = scripts[choice-1]
    
    cores = int(input("Cores (1-8) [4]: ") or 4)
    duration = int(input(f"Duration {DURATION_OPTIONS}: ") or DURATION_OPTIONS[0])

    monitor = DecisionTreeDVFSController()
    if not monitor.start_workload(workload, cores): return

    m_thread = threading.Thread(target=monitor.start_monitoring, args=(workload, cores, duration), daemon=True)
    m_thread.start()
    
    try: monitor.start_realtime_plot()
    except: pass
    m_thread.join()

if __name__ == "__main__":
    main()