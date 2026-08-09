# energy_aware_dvfs.py
"""
Energy-Aware DVFS Controller with Sliding Window + Majority Vote + Hysteresis
- Real-time monitoring with Intel PCM & Power Gadget
- Sliding Window over PMCs (IPC, L3 miss rate, CPU usage)
- Majority vote over recent labels
- Hysteresis on final DVFS level
- Baseline vs Adaptive comparison via USE_CUSTOM_DVFS
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
from pathlib import Path
import sys

try:
    import pandas as pd
except ImportError:
    print("❌ Please install pandas: pip install pandas")
    sys.exit(1)

# =================== CONFIGURATION ===================
ROOT = Path(__file__).resolve().parent
LOG_FILE = str((ROOT / f"monitoring_log_SW_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json").resolve())

# PCM_PATHS = [
#     str((ROOT / "pcm.exe").resolve()),
#     r"C:\Users\saidm\OneDrive\Desktop\Programming\pcm\build\bin\Release\pcm.exe",
#     "pcm.exe"
# ]

# POWER_GADGET_PATHS = [
#     str((ROOT / "pcm-power.exe").resolve()),
#     r"C:\Users\saidm\OneDrive\Desktop\Programming\pcm\build\bin\Release\pcm-power.exe",
#     "pcm-power.exe"
# ]

PCM_PATHS = [
    r"C:\Users\saidm\Desktop\Programming\pcm\build\bin\Release\pcm.exe",
    r"C:\Users\saidm\pcm\build\bin\Release\pcm.exe",
    str((ROOT / "pcm.exe").resolve()),
]

POWER_GADGET_PATHS = [
    r"C:\Users\saidm\Desktop\Programming\pcm\build\bin\Release\pcm-power.exe",
    r"C:\Users\saidm\pcm\build\bin\Release\pcm-power.exe",
    str((ROOT / "pcm-power.exe").resolve()),
]

DURATION_OPTIONS = [100, 200, 300]

# Sliding window & decision config
WINDOW_SIZE = 8          # samples for PMC averaging
LABEL_WINDOW_SIZE = 8    # samples for label majority vote

# Toggle: False = baseline (fixed Medium), True = SW-DVFS
USE_CUSTOM_DVFS = False
# =====================================================


class RobustSlidingWindowDVFS:
    def __init__(self):
        self.running = True
        self.data_buffer = []
        self.pcm_path = self.find_tool(PCM_PATHS)
        self.power_gadget_path = self.find_tool(POWER_GADGET_PATHS)

        self.fig, self.axs = plt.subplots(4, 2, figsize=(16, 12))
        self.fig.suptitle("Real-Time DVFS Monitoring", fontsize=14, fontweight='bold')

        # Power plan GUIDs
        self.POWER_PLANS = {
            "High": "27ad8305-4092-41f2-a01b-5a6003fb5077",
            "Medium": "381b4222-f694-41f0-9685-ff5bb260df2e",
            "Low": "b2524225-86dc-424d-ba5b-e78d683c5d3a"
        }

        # Sliding windows over PMCs
        self.ipc_window = []      # last N IPC values
        self.l3_window = []       # last N L3 miss rates
        self.cpu_window = []      # last N CPU usage values

        # Sliding window over raw labels (Low/Medium/High)
        self.label_window = []

        # Final DVFS state (after majority+streak)
        self.last_final_label = "Medium"
        self.final_streak = 0  # how many consecutive times the majority label differs from last_final_label
        self.workload_process = None

        # For logging/debugging last decisions and averages
        self.last_raw_label = "Medium"
        self.last_majority_label = "Medium"
        self.last_avg_ipc = 0.0
        self.last_avg_l3 = 0.0
        self.last_avg_cpu = 0.0

    # --------------------------------------------------
    # TOOL DISCOVERY
    # --------------------------------------------------
    def find_tool(self, possible_paths):
        """Find tool in multiple locations."""
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found: {path}")
                return path
        print(f"⚠️ Tool not found in any of: {possible_paths}")
        return None

    # --------------------------------------------------
    # POWER PLAN CONTROL (with min/max P-state limits)
    # --------------------------------------------------
    def set_power_plan(self, level):
        """Switch Windows power plan using powercfg and enforce min/max processor states."""
        guid = self.POWER_PLANS.get(level)
        if not guid:
            print(f"❌ Invalid level: {level}")
            return False

        try:
            # Configure per-level processor throttle settings (AC side)
            if level == "Low":
                # Very energy-saving: cap max to ~30%, allow min 5%
                subprocess.run(
                    ["powercfg", "-setacvalueindex", guid, "SUB_PROCESSOR", "PROCTHROTTLEMAX", "30"],
                    check=True
                )
                subprocess.run(
                    ["powercfg", "-setacvalueindex", guid, "SUB_PROCESSOR", "PROCTHROTTLEMIN", "5"],
                    check=True
                )
            elif level == "Medium":
                # Balanced: 30–60%
                subprocess.run(
                    ["powercfg", "-setacvalueindex", guid, "SUB_PROCESSOR", "PROCTHROTTLEMAX", "60"],
                    check=True
                )
                subprocess.run(
                    ["powercfg", "-setacvalueindex", guid, "SUB_PROCESSOR", "PROCTHROTTLEMIN", "30"],
                    check=True
                )
            elif level == "High":
                # Performance: always 100%
                subprocess.run(
                    ["powercfg", "-setacvalueindex", guid, "SUB_PROCESSOR", "PROCTHROTTLEMAX", "100"],
                    check=True
                )
                subprocess.run(
                    ["powercfg", "-setacvalueindex", guid, "SUB_PROCESSOR", "PROCTHROTTLEMIN", "100"],
                    check=True
                )

            # Activate the chosen plan
            subprocess.run(["powercfg", "-setactive", guid], check=True)
            return True

        except Exception as e:
            print(f"❌ Failed to set power plan: {e}")
            return False

    # --------------------------------------------------
    # PCM (PERF COUNTERS)
    # --------------------------------------------------
    def simulate_pcm_data(self):
        """Simulate when PCM not available."""
        return {
            'ipc': np.random.uniform(0.5, 2.0),
            'l2_cache_hits': np.random.randint(50000, 80000),
            'l2_cache_misses': np.random.randint(1000, 5000),
            'l3_cache_hits': np.random.randint(30000, 60000),
            'l3_cache_misses': np.random.randint(500, 3000),
            'memory_bandwidth': np.random.uniform(10000, 25000)
        }

    def read_pcm_data(self):
        """Read from Intel PCM."""
        if not self.pcm_path:
            return self.simulate_pcm_data()

        try:
            result = subprocess.run(
                [self.pcm_path, "1", "-nc", "-ns"],
                capture_output=True, text=True, timeout=10
            )
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                return self.simulate_pcm_data()

            data_line = lines[-1].strip()
            parts = [x for x in data_line.split(' ') if x]

            if len(parts) >= 8:
                ipc = float(parts[1])
                l2_hits = int(parts[4])
                l2_misses = int(parts[5])
                l3_hits = int(parts[6])
                l3_misses = int(parts[7])
                mem_bw = float(parts[2])  # MB/s

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

    # --------------------------------------------------
    # POWER GADGET / PCM-POWER
    # --------------------------------------------------
    def simulate_power_data(self):
        """Simulate power data."""
        return {
            'cpu_power': np.random.uniform(25, 65),
            'gpu_power': np.random.uniform(5, 20),
            'cpu_temperature': np.random.uniform(50, 85),
            'cpu_frequency': np.random.uniform(1800, 3600)
        }

    def read_power_gadget_data(self):
        """Read from Power Gadget / pcm-power."""
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
                    'cpu_power': float(data[1]) if len(data) > 1 and data[1] else 0,
                    'gpu_power': float(data[2]) if len(data) > 2 and data[2] else 0,
                    'cpu_temperature': float(data[3]) if len(data) > 3 and data[3] else 0,
                    'cpu_frequency': float(data[4]) if len(data) > 4 and data[4] else 0
                }
                os.remove(temp_file)
                return power_data

        except Exception as e:
            print(f"Power Gadget Error: {e}")
        return self.simulate_power_data()

    # --------------------------------------------------
    # PMC SLIDING WINDOWS
    # --------------------------------------------------
    def update_pmc_windows(self, ipc, l3_miss, cpu_usage):
        """Maintain sliding windows over IPC, L3 miss rate, and CPU usage."""
        # IPC window
        self.ipc_window.append(ipc)
        if len(self.ipc_window) > WINDOW_SIZE:
            self.ipc_window.pop(0)

        # L3 miss window
        self.l3_window.append(l3_miss)
        if len(self.l3_window) > WINDOW_SIZE:
            self.l3_window.pop(0)

        # CPU usage window
        self.cpu_window.append(cpu_usage)
        if len(self.cpu_window) > WINDOW_SIZE:
            self.cpu_window.pop(0)

    def compute_window_averages(self):
        """Compute moving averages over the sliding windows."""
        avg_ipc = np.mean(self.ipc_window) if self.ipc_window else 0.0
        avg_l3 = np.mean(self.l3_window) if self.l3_window else 0.0
        avg_cpu = np.mean(self.cpu_window) if self.cpu_window else 0.0
        return avg_ipc, avg_l3, avg_cpu

    # --------------------------------------------------
    # LABEL WINDOW + MAJORITY VOTE
    # --------------------------------------------------
    def majority_vote(self):
        """Return the most frequent label in the label window."""
        if not self.label_window:
            return "Medium"
        counts = {
            "Low": self.label_window.count("Low"),
            "Medium": self.label_window.count("Medium"),
            "High": self.label_window.count("High")
        }
        return max(counts, key=counts.get)

    # --------------------------------------------------
    # SINGLE-SAMPLE CLASSIFIER (USED WITH AVERAGES OR FIRST SAMPLES)
    # --------------------------------------------------
    # def classify_single_sample(self, ipc, l3_miss_rate, cpu_usage):
    #     """
    #     Classify a single (ipc, l3_miss_rate, cpu_usage) triple into Low/Medium/High.
    #     This function is reused with either raw or averaged metrics.
    #     """
    #     # Idle / light-load → LOW
    #     if cpu_usage < 25.0:
    #         return "Low"

    #     # Memory-bound → LOW
    #     if ipc < 1.0 and l3_miss_rate > 0.04:
    #         return "Low"

    #     # Compute-bound → HIGH
    #     if ipc > 1.3 and l3_miss_rate < 0.06:
    #         return "High"

    #     # Otherwise → MEDIUM
    #     return "Medium"


    def classify_single_sample(self, ipc, l3_miss_rate, cpu_usage):
        """
        Calibrated Multi-Variable DVFS Threshold Policy
        Optimized for 8-Core Client/HPC Workload Boundaries
        """
        # 1. Memory-Bound / Bus Stalls → LOW
        # Pipeline stalls out waiting for memory controller. Lowering clock speed 
        # saves power without degrading performance since memory latency dominates.
        if ipc < 0.95 and l3_miss_rate > 0.015:
            return "Low"

        # 2. Low Compute Intensity / Bursty Idle Gaps → LOW
        # Catches intermittent/paused execution models where system load averages 
        # out below 65% core saturation.
        if cpu_usage < 65.0:
            return "Low"

        # 3. Compute-Bound Saturated → HIGH
        # High core utilization combined with efficient pipeline execution loops.
        if cpu_usage > 75.0 and ipc > 1.3:
            return "High"

        # 4. Otherwise / Balanced Dynamics → MEDIUM
        return "Medium"


    # --------------------------------------------------
    # DEBUG: PRINT WINDOWS + AVERAGES + LABELS
    # --------------------------------------------------
    def debug_print_windows(self, avg_ipc, avg_l3, avg_cpu,
                            raw_label, majority_label, final_label):
        print("\n📊 Sliding Window Debug Info")
        print("--------------------------------------------------")

        print(f"IPC Window ({len(self.ipc_window)} values):")
        print(f"  {['{:.2f}'.format(x) for x in self.ipc_window]}")
        print(f"  → avg IPC       = {avg_ipc:.3f}")

        print(f"\nL3 Miss Rate Window ({len(self.l3_window)} values):")
        print(f"  {['{:.3f}'.format(x) for x in self.l3_window]}")
        print(f"  → avg L3 miss   = {avg_l3:.3f}")

        print(f"\nCPU Usage Window ({len(self.cpu_window)} values):")
        print(f"  {['{:.1f}'.format(x) for x in self.cpu_window]}")
        print(f"  → avg CPU usage = {avg_cpu:.1f}%")

        print("\n🧠 Decision Flow:")
        print(f"  Raw label      : {raw_label}")
        print(f"  Majority label : {majority_label}")
        print(f"  Final (Hyst.)  : {final_label}")
        print("--------------------------------------------------\n")

    # --------------------------------------------------
    # MULTI-STAGE DVFS CLASSIFIER (WINDOW + MAJORITY + HYSTERESIS)
    # --------------------------------------------------
    def classify_performance_level(self, data):
        """
        Multi-stage DVFS classifier:

        Stage 1: Update PMC windows (IPC, L3 miss, CPU usage).
        Stage 2: Use averages (when enough samples) to produce a raw label.
        Stage 3: Push raw label into label window and take majority vote.
        Stage 4: Apply hysteresis on final level (require 2 consecutive majority labels
                 different from last_final_label).
        """
        ipc = data['ipc']
        l3_miss = data['l3_miss_rate']
        cpu_usage = data['cpu_usage_overall']

        # --- Stage 1: Update PMC windows ---
        self.update_pmc_windows(ipc, l3_miss, cpu_usage)

        # --- Stage 2: Classification using averages (once we have enough samples) ---
        if len(self.ipc_window) < 3:
            # Not enough history yet → use current sample as "average"
            avg_ipc, avg_l3, avg_cpu = ipc, l3_miss, cpu_usage
            raw_label = self.classify_single_sample(ipc, l3_miss, cpu_usage)
        else:
            avg_ipc, avg_l3, avg_cpu = self.compute_window_averages()
            raw_label = self.classify_single_sample(avg_ipc, avg_l3, avg_cpu)

        # --- Stage 3: Majority vote over recent raw labels ---
        self.label_window.append(raw_label)
        if len(self.label_window) > LABEL_WINDOW_SIZE:
            self.label_window.pop(0)

        majority_label = self.majority_vote()

        # --- Stage 4: Hysteresis on final applied state ---
        if majority_label == self.last_final_label:
            self.final_streak = 0
            final_label = self.last_final_label
        else:
            self.final_streak += 1
            if self.final_streak >= 2:
                self.last_final_label = majority_label
                self.final_streak = 0
            final_label = self.last_final_label

        # Store last decisions and averages for logging
        self.last_raw_label = raw_label
        self.last_majority_label = majority_label
        self.last_avg_ipc = avg_ipc
        self.last_avg_l3 = avg_l3
        self.last_avg_cpu = avg_cpu

        # Print full window + averages + labels every decision
        self.debug_print_windows(avg_ipc, avg_l3, avg_cpu,
                                 raw_label, majority_label, final_label)

        return final_label

    # --------------------------------------------------
    # DATA COLLECTION
    # --------------------------------------------------
    def collect_data(self):
        """Collect all metrics and apply DVFS logic."""
        timestamp = datetime.now()
        pcm_data = self.read_pcm_data()
        power_data = self.read_power_gadget_data()

        total_l2 = pcm_data['l2_cache_hits'] + pcm_data['l2_cache_misses']
        total_l3 = pcm_data['l3_cache_hits'] + pcm_data['l3_cache_misses']
        l2_miss_rate = pcm_data['l2_cache_misses'] / total_l2 if total_l2 > 0 else 0
        l3_miss_rate = pcm_data['l3_cache_misses'] / total_l3 if total_l3 > 0 else 0

        cpu_percent_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        memory = psutil.virtual_memory()
        cpu_percent_overall = psutil.cpu_percent(interval=0.1)

        data = {
            'timestamp': timestamp.isoformat(),
            'cpu_usage_overall': cpu_percent_overall,
            'cpu_usage_per_core': cpu_percent_per_core,
            'memory_usage': memory.percent,
            'cpu_count': len(cpu_percent_per_core),
            'ipc': pcm_data['ipc'],
            'l2_cache_hits': pcm_data['l2_cache_hits'],
            'l2_cache_misses': pcm_data['l2_cache_misses'],
            'l3_cache_hits': pcm_data['l3_cache_hits'],
            'l3_cache_misses': pcm_data['l3_cache_misses'],
            'l2_miss_rate': round(l2_miss_rate, 3),
            'l3_miss_rate': round(l3_miss_rate, 3),
            'memory_bandwidth': round(pcm_data['memory_bandwidth'], 1),
            'cpu_power': round(power_data['cpu_power'], 1),
            'cpu_temperature': round(power_data['cpu_temperature'], 1),
            'cpu_frequency': round(power_data['cpu_frequency'], 1)
        }

        # Apply DVFS logic
        if USE_CUSTOM_DVFS:
            level = self.classify_performance_level(data)
            applied = self.set_power_plan(level)
        else:
            # Baseline: Fixed "Medium" (Balanced mode)
            level = "Medium"
            applied = self.set_power_plan("Medium")

        # Attach DVFS labels and windows to JSON log
        data['dvfs_level'] = level
        data['dvfs_applied'] = applied
        data['raw_label'] = self.last_raw_label
        data['majority_label'] = self.last_majority_label
        data['final_label'] = level
        data['ipc_window'] = self.ipc_window.copy()
        data['l3_window'] = self.l3_window.copy()
        data['cpu_window'] = self.cpu_window.copy()
        data['avg_ipc'] = self.last_avg_ipc
        data['avg_l3_miss'] = self.last_avg_l3
        data['avg_cpu_usage'] = self.last_avg_cpu

        return data

    # --------------------------------------------------
    # WORKLOAD LAUNCH
    # --------------------------------------------------
    def start_workload(self, script_name, num_cores):
        """Start selected workload with the requested OpenMP/BLAS thread count."""
        script_path = Path(script_name)
        if not script_path.is_absolute():
            script_path = ROOT / script_path
        script_path = script_path.resolve()

        if not script_path.exists():
            print(f"❌ Workload not found: {script_path}")
            return False

        env = os.environ.copy()
        thread_count = str(num_cores)
        env["OMP_NUM_THREADS"] = thread_count
        env["MKL_NUM_THREADS"] = thread_count
        env["OPENBLAS_NUM_THREADS"] = thread_count
        env["NUMEXPR_NUM_THREADS"] = thread_count

        try:
            print(f"🚀 Starting {script_path.relative_to(ROOT)} on {num_cores} core(s).")
            creationflags = 0
            if os.name == "nt":
                creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            self.workload_process = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(ROOT),
                env=env,
                creationflags=creationflags,
            )
            print(f"✅ Workload started (PID {self.workload_process.pid}).")
            return True
        except Exception as e:
            print(f"❌ Failed to start workload: {e}")
            self.workload_process = None
            return False

    def stop_workload(self):
        """Stop the launched workload so benchmark runs cannot overlap."""
        proc = self.workload_process
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True, text=True, check=False,
                )
            else:
                proc.terminate()
                proc.wait(timeout=5)
        except Exception as e:
            print(f"⚠️ Could not stop workload cleanly: {e}")
            try:
                proc.kill()
            except Exception:
                pass
        finally:
            self.workload_process = None

    # --------------------------------------------------
    # MAIN MONITORING LOOP
    # --------------------------------------------------
    def start_monitoring(self, workload_script, num_cores, duration):
        """Main monitoring loop."""
        print(f"🚀 Starting monitoring for {duration} seconds.")
        print(f"📊 Log: {LOG_FILE}")
        print(f"🔧 PCM: {'Yes' if self.pcm_path else 'No (simulated)'}")
        print(f"🔧 Power Gadget: {'Yes' if self.power_gadget_path else 'No (simulated)'}")
        print(f"🔧 Workload: {workload_script} on {num_cores} core(s)")
        print(f"🔧 DVFS Mode: {'Custom Sliding Window' if USE_CUSTOM_DVFS else 'Baseline (Fixed Medium)'}")
        print("=" * 80)

        start_time = time.time()
        with open(LOG_FILE, 'w') as f:
            iteration = 0
            while self.running and (time.time() - start_time < duration):
                try:
                    iteration += 1
                    data = self.collect_data()
                    f.write(json.dumps(data) + '\n')
                    f.flush()
                    self.data_buffer.append(data)

                    if iteration % 3 == 1:
                        level = data['dvfs_level']
                        ipc = data['ipc']
                        power = data['cpu_power']
                        temp = data['cpu_temperature']
                        cpu = data['cpu_usage_overall']
                        l3_miss = data['l3_miss_rate']
                        print(f"[{iteration:3}] {datetime.now().strftime('%H:%M:%S')} | "
                              f"Level: {level:6} | IPC: {ipc:4.2f} | Power: {power:5.1f}W | "
                              f"Temp: {temp:4.1f}°C | CPU: {cpu:5.1f}% | L3 Miss: {l3_miss:5.3f}")

                    time.sleep(3)

                except Exception as e:
                    print(f"❌ Monitoring error: {e}")
                    continue

        self.stop_workload()
        print("⏹️ Workload stopped.")
        print("✅ Experiment completed. Results logged.")

    # --------------------------------------------------
    # REAL-TIME PLOTTING
    # --------------------------------------------------
    def update_plots(self, frame):
        """Update real-time plots."""
        if not self.data_buffer:
            return

        df = pd.DataFrame(self.data_buffer)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['time_min'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 60

        # Clear all subplots
        for ax_row in self.axs:
            for ax in ax_row:
                ax.clear()

        df['color'] = df['dvfs_level'].map({'Low': 'red', 'Medium': 'orange', 'High': 'green'})

        plots = [
            ('cpu_power', 'CPU Power (W)', 'CPU Power vs Time', 'blue'),
            ('ipc', 'IPC', 'IPC vs Time', 'green'),
            ('cpu_frequency', 'Frequency (MHz)', 'CPU Frequency vs Time', 'magenta'),
            ('cpu_temperature', 'Temp (°C)', 'CPU Temperature vs Time', 'red'),
            ('l2_miss_rate', 'L2 Miss Rate', 'L2 Cache Miss Rate', 'purple'),
            ('l3_miss_rate', 'L3 Miss Rate', 'L3 Cache Miss Rate', 'brown'),
            ('memory_usage', 'Memory Usage (%)', 'Memory Usage vs Time', 'gray'),
            ('cpu_usage_overall', 'CPU Usage (%)', 'CPU Usage vs Time', 'olive'),
        ]

        for idx, (col, ylabel, title, color) in enumerate(plots):
            if col in df.columns:
                ax = self.axs[idx // 2, idx % 2]
                ax.plot(df['time_min'], df[col], color=color, alpha=0.8, linewidth=1.5)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

    def start_realtime_plot(self):
        """Start visualization."""
        print("📊 Starting real-time visualization.")
        ani = FuncAnimation(self.fig, self.update_plots, interval=3000, blit=False, cache_frame_data=False)
        plt.show()


def main():
    print("Integrated Workload & Real-Time Monitor")
    print("=" * 60)

    # # List workloads
    # scripts = []
    # if os.path.exists("cpu_intensive.py"):
    #     print(" 1. cpu_intensive.py")
    #     scripts.append("cpu_intensive.py")
    # if os.path.exists("memory_intensive.py"):
    #     print(" 2. memory_intensive.py")
    #     scripts.append("memory_intensive.py")

    # if not scripts:
    #     print("❌ No workload scripts found. Please place cpu_intensive.py or memory_intensive.py in this folder.")
    #     return
    
    # List workloads
    # scripts = []
    # if os.path.exists("mibench_qsort_cpuintensive.py"):
    #     print(" 1. mibench_qsort_cpuintensive.py (MiBench - CPU)")
    #     scripts.append("mibench_qsort_cpuintensive.py")
    # if os.path.exists("mibench_susan_mixed.py"):
    #     print(" 2. mibench_susan_mixed.py (MiBench - Mixed)")
    #     scripts.append("mibench_susan_mixed.py")
    # if os.path.exists("scimark_cpu.py"):
    #     print(" 3. scimark_cpu.py (SciMark 2 - FPU/CPU)")
    #     scripts.append("scimark_cpu.py")
    # if os.path.exists("scimark_memory.py"):
    #     print(" 4. scimark_memory.py (SciMark 2 - Memory)")
    #     scripts.append("scimark_memory.py")

    # List workloads
    configured_scripts = [
        "workload_mem2_stream_huge.py",              # Saturated Memory (High/Medium)
        "NPB3.0-omp-C/workload_cpu1_nas_ep.py",      # Saturated CPU (High)
        "NPB3.0-omp-C/workload_mem1_bursty_is.py",   # Bursty Memory (Low)
        "NPB3.0-omp-C/workload_cpu2_bursty_ep.py",   # Bursty CPU (Low)
        "NPB3.0-omp-C/workload_mixed1_bursty_lu.py", # Bursty Mixed 1 (Low)
        "NPB3.0-omp-C/workload_mixed2_bursty_mg.py"  # Bursty Mixed 2 (Low)
    ]
    scripts = []
    for script in configured_scripts:
        if (ROOT / script).exists():
            scripts.append(script)
        else:
            print(f"⚠️ Missing workload: {script}")
    
    print("Available Workloads:")
    for i, script in enumerate(scripts):
        print(f" {i+1}. {script}")

    if not scripts:
        print("❌ No workload scripts found.")
        return

    # Get user input
    try:
        # Workload
        while True:
            choice = input(f"Enter workload choice (1-{len(scripts)}): ").strip()
            if choice in [str(i + 1) for i in range(len(scripts))]:
                workload_script = scripts[int(choice) - 1]
                break
            print(f"Please enter 1 to {len(scripts)}.")

        # Cores
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

    except KeyboardInterrupt:
        print("❌ Setup cancelled by user.")
        return

    # Run experiment
    monitor = RobustSlidingWindowDVFS()

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
