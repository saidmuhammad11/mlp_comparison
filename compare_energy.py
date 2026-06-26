# dvfs_enhanced_analysis.py
"""
Enhanced DVFS Comparison Tool
Now includes memory-bound detection and robust adaptive analysis

COLOR SCHEME:
- Pink (#FF69B4): With HistGBDTController
- Blue (#4169E1): Baseline (Medium) Controller (Baseline)
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from collections import Counter

# =================== CONFIGURATION ===================
BASELINE_FILE = 'histgbdtdisable_memoryintensive3.json'  # Baseline (Medium)
DVFS_FILE = 'histgbdtenable_memoryintensive3.json'          # With HistGBDT
OUTPUT_PLOT_COMPREHENSIVE = 'dvfs_comprehensive_analysis.png'
OUTPUT_PLOT_BOX = 'dvfs_box_plots.png'
OUTPUT_PLOT_LEVELS = 'dvfs_performance_levels.png'
OUTPUT_PLOT_CACHE = 'dvfs_cache_performance.png'
OUTPUT_TABLE_PNG = 'dvfs_comparison_table.png'
OUTPUT_TABLE_CSV = 'dvfs_comparison_results.csv'
OUTPUT_REPORT = 'dvfs_analysis_report.txt'

SAMPLING_INTERVAL_SEC = 3
WORKLOAD_NAME = "Mixed"

# Colors
COLOR_WITH_DVFS = '#FF69B4'      # Pink: With HistGBDT
COLOR_default_DVFS = '#4169E1'   # Blue: Baseline (Medium)

# Memory bandwidth threshold (adjust based on your system)
MAX_MEMORY_BANDWIDTH = 50.0  # GB/s (typical DDR4 dual-channel)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
# =====================================================


def load_json_log(filename):
    """Load and parse JSON log file"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    # Ensure dvfs_level exists
                    if 'dvfs_level' not in entry:
                        entry['dvfs_level'] = "Fixed"  # Default for baseline
                    data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue
    if not data:
        raise ValueError(f"No valid data found in {filename}")
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)


def compute_energy(df, power_col, dt):
    """Compute total energy in Joules"""
    return np.sum(df[power_col].dropna().values) * dt


class RobustDVFSComparator:
    def __init__(self, df_baseline, df_dvfs, workload_name="Unknown"):
        self.df_baseline = df_baseline
        self.df_dvfs = df_dvfs
        self.workload_name = workload_name
        self.color_baseline = COLOR_default_DVFS
        self.color_dvfs = COLOR_WITH_DVFS
        self.color_baseline_alpha = COLOR_default_DVFS + '80'
        self.color_dvfs_alpha = COLOR_WITH_DVFS + '80'

        # Detect workload type
        self.workload_type = self.detect_workload_type()

    def detect_workload_type(self):
        """Detect if workload is CPU or memory intensive using DVFS log"""
        df = self.df_dvfs
        avg_l3_miss = df['l3_miss_rate'].mean()
        avg_mem_bw = df['memory_bandwidth'].mean() if 'memory_bandwidth' in df.columns else 0

        if avg_l3_miss > 0.05 and avg_mem_bw > 0.6 * MAX_MEMORY_BANDWIDTH:
            return "memory_intensive"
        return "cpu_intensive"

    def create_comprehensive_plots(self):
        """Create 4x2 grid of time-series plots (ONE legend only on the top-right subplot)"""
        fig = plt.figure(figsize=(18, 24))
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.08, right=0.95, hspace=0.6, wspace=0.4)

        time_baseline = (self.df_baseline['timestamp'] - self.df_baseline['timestamp'].iloc[0]).dt.total_seconds() / 60
        time_dvfs = (self.df_dvfs['timestamp'] - self.df_dvfs['timestamp'].iloc[0]).dt.total_seconds() / 60

        metrics = [
            ('cpu_power', 'CPU Power (W)', 'CPU Power'),
            ('ipc', 'IPC', 'IPC'),
            ('cpu_frequency', 'CPU Frequency (MHz)', 'CPU Frequency'),
            ('memory_usage', 'Memory Usage (%)', 'Memory Usage'),
            ('l2_miss_rate', 'L2 Miss Rate (%)', 'L2 Cache Miss Rate'),
            ('l3_miss_rate', 'L3 Miss Rate (%)', 'L3 Cache Miss Rate'),
            ('cpu_temperature', 'Temperature (°C)', 'CPU Temperature'),
            ('cpu_usage_overall', 'CPU Usage (%)', 'CPU Usage')
        ]

        # Collect legend handles once, and draw legend only on subplot #2 (top-right)
        legend_handles, legend_labels = None, None
        legend_ax = None

        for idx, (col, ylabel, title) in enumerate(metrics, 1):
            ax = plt.subplot(4, 2, idx)

            if idx == 2:
                legend_ax = ax

            if col in self.df_baseline.columns and col in self.df_dvfs.columns:
                ax.plot(
                    time_baseline, self.df_baseline[col],
                    label='Baseline (Medium)', color=self.color_baseline, linewidth=2, alpha=0.8
                )
                ax.plot(
                    time_dvfs, self.df_dvfs[col],
                    label='With HistGBDT', color=self.color_dvfs, linewidth=2, alpha=0.8
                )

                # Grab legend entries once (first subplot that actually has lines)
                if legend_handles is None:
                    legend_handles, legend_labels = ax.get_legend_handles_labels()

            ax.set_xlabel('Time (minutes)', fontsize=18)
            ax.set_ylabel(ylabel, fontsize=17)
            ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            # IMPORTANT: do NOT call ax.legend() here (that’s what was spamming it everywhere)

        # Draw ONE legend (top-right subplot)
        if legend_ax is not None and legend_handles is not None:
            legend_ax.legend(legend_handles, legend_labels, fontsize=20, loc='upper right')

        plt.tight_layout(pad=4.0, h_pad=3.0, w_pad=2.0)
        plt.savefig(OUTPUT_PLOT_COMPREHENSIVE, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Comprehensive plots saved: {OUTPUT_PLOT_COMPREHENSIVE}")

    def create_box_plots(self):
        """Create 2x2 box plots for key metrics"""
        fig = plt.figure(figsize=(18, 12))
        plt.subplots_adjust(top=0.98, bottom=0.05, left=0.08, right=0.95, hspace=0.6, wspace=0.4)

        metrics = [
            ('cpu_power', 'CPU Power (W)', 'Power'),
            ('ipc', 'IPC', 'IPC'),
            ('l2_miss_rate', 'L2 Miss Rate (%)', 'L2 Miss Rate'),
            ('l3_miss_rate', 'L3 Miss Rate (%)', 'L3 Miss Rate')
        ]

        data_pairs = [
            (self.df_baseline['cpu_power'], self.df_dvfs['cpu_power']),
            (self.df_baseline['ipc'], self.df_dvfs['ipc']),
            (self.df_baseline['l2_miss_rate'], self.df_dvfs['l2_miss_rate']),
            (self.df_baseline['l3_miss_rate'], self.df_dvfs['l3_miss_rate'])
        ]

        labels = ['Baseline (Medium)', 'With HistGBDT']
        colors = [self.color_baseline_alpha, self.color_dvfs_alpha]
        edge_colors = [self.color_baseline, self.color_dvfs]

        for idx, ((col, ylabel, title), (data_baseline, data_dvfs)) in enumerate(zip(metrics, data_pairs), 1):
            ax = plt.subplot(2, 2, idx)
            bp = ax.boxplot([data_baseline, data_dvfs], labels=labels, patch_artist=True)
            for patch, color, edge in zip(bp['boxes'], colors, edge_colors):
                patch.set_facecolor(color)
                patch.set_edgecolor(edge)
            ax.set_ylabel(ylabel, fontsize=18)
            ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
            ax.tick_params(axis='both', labelsize=20)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout(pad=4.0, h_pad=3.0, w_pad=2.0)
        plt.savefig(OUTPUT_PLOT_BOX, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Box plots saved: {OUTPUT_PLOT_BOX}")

    def create_performance_level_plots(self):
        """Plot DVFS level distribution for both runs"""
        fig = plt.figure(figsize=(18, 10))
        plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.95, hspace=0.4, wspace=0.4)

        ax1 = plt.subplot(1, 2, 1)
        self._plot_distribution(ax1, self.df_baseline, 'dvfs_level', 'Baseline (Medium)', self.color_baseline)

        ax2 = plt.subplot(1, 2, 2)
        self._plot_distribution(ax2, self.df_dvfs, 'dvfs_level', 'With HistGBDT', self.color_dvfs)

        plt.tight_layout(pad=4.0, h_pad=3.0, w_pad=2.0)
        plt.savefig(OUTPUT_PLOT_LEVELS, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Performance level plots saved: {OUTPUT_PLOT_LEVELS}")

    def _plot_distribution(self, ax, df, col, title, color):
        """Helper to plot bar distribution"""
        if col not in df.columns:
            ax.set_visible(False)
            return
        counts = Counter(df[col])
        categories = sorted(
            counts.keys(),
            key=lambda x: ["Low", "Medium", "High"].index(x) if x in ["Low", "Medium", "High"] else 3
        )
        values = [counts[c] for c in categories]
        bars = ax.bar(categories, values, color=color, alpha=0.7, edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.5, f'{int(height)}', ha='center', fontsize=18)
        ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
        ax.set_xlabel('Performance Level', fontsize=18)
        ax.set_ylabel('Count', fontsize=18)
        ax.tick_params(axis='both', labelsize=20)
        ax.grid(True, alpha=0.3, axis='y')

    def create_cache_performance_plot(self):
        """Plot L2/L3 cache hit rate comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        l2_hit_with = 1 - self.df_dvfs['l2_miss_rate'].mean()
        l2_hit_default = 1 - self.df_baseline['l2_miss_rate'].mean()
        l3_hit_with = 1 - self.df_dvfs['l3_miss_rate'].mean()
        l3_hit_default = 1 - self.df_baseline['l3_miss_rate'].mean()

        categories = ['L2 Cache', 'L3 Cache']
        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width / 2, [l2_hit_default, l3_hit_default], width,
               label='Baseline (Medium)', color=self.color_baseline, alpha=0.8)
        ax.bar(x + width / 2, [l2_hit_with, l3_hit_with], width,
               label='With HistGBDT', color=self.color_dvfs, alpha=0.8)

        ax.set_ylabel('Hit Rate', fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=18)
        ax.tick_params(axis='y', which='major', labelsize=18)
        ax.legend(fontsize=20)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT_CACHE, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Cache performance plot saved: {OUTPUT_PLOT_CACHE}")

    def create_comparison_table(self):
        """Generate a detailed comparison table with color-coded improvements"""
        E_baseline = compute_energy(self.df_baseline, 'cpu_power', SAMPLING_INTERVAL_SEC)
        E_dvfs = compute_energy(self.df_dvfs, 'cpu_power', SAMPLING_INTERVAL_SEC)

        metrics = [
            ('Avg CPU Power (W)', self.df_baseline['cpu_power'].mean(), self.df_dvfs['cpu_power'].mean(), 'lower'),
            ('Avg IPC', self.df_baseline['ipc'].mean(), self.df_dvfs['ipc'].mean(), 'higher'),
            ('Avg CPU Usage (%)', self.df_baseline['cpu_usage_overall'].mean(), self.df_dvfs['cpu_usage_overall'].mean(), 'lower'),
            ('Avg L2 Miss Rate', self.df_baseline['l2_miss_rate'].mean(), self.df_dvfs['l2_miss_rate'].mean(), 'lower'),
            ('Avg L3 Miss Rate', self.df_baseline['l3_miss_rate'].mean(), self.df_dvfs['l3_miss_rate'].mean(), 'lower'),
            ('Avg Temp (°C)', self.df_baseline['cpu_temperature'].mean(), self.df_dvfs['cpu_temperature'].mean(), 'lower'),
            ('Avg Freq (GHz)', self.df_baseline['cpu_frequency'].mean() / 1000, self.df_dvfs['cpu_frequency'].mean() / 1000, 'higher'),
            ('Total Energy (J)', E_baseline, E_dvfs, 'lower'),
        ]

        data = {
            'Metric': [],
            'Baseline (Medium)': [],
            'With HistGBDT': [],
            'Improvement': [],
            'Improvement %': []
        }

        for name, base_val, dvfs_val, direction in metrics:
            imp = dvfs_val - base_val
            imp_pct = (abs(imp) / base_val) * 100 if base_val != 0 else 0.0
            sign = '+' if ((imp < 0 and direction == 'lower') or (imp > 0 and direction == 'higher')) else '-'
            imp_str = f"{imp:+.3f}"
            imp_pct_str = f"{sign}{imp_pct:.1f}%"

            data['Metric'].append(name)
            data['Baseline (Medium)'].append(f"{base_val:.3f}")
            data['With HistGBDT'].append(f"{dvfs_val:.3f}")
            data['Improvement'].append(imp_str)
            data['Improvement %'].append(imp_pct_str)

        df_table = pd.DataFrame(data)
        df_table.to_csv(OUTPUT_TABLE_CSV, index=False)

        # Plot table
        fig, ax = plt.subplots(figsize=(14, len(df_table) * 0.8))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(
            cellText=df_table.values,
            colLabels=df_table.columns,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(18)
        table.scale(1.2, 2)

        # Color improvements column
        for i in range(len(df_table)):
            pct = df_table.iloc[i]['Improvement %']
            is_positive = (pct.startswith('+') and float(pct.strip('%+')) > 0)
            metric = df_table.iloc[i]['Metric']
            if 'Energy' in metric and is_positive:
                table[(i + 1, 4)].set_facecolor('#90EE90')  # Green
            elif not is_positive:
                table[(i + 1, 4)].set_facecolor('#FFB6C1')  # Red

        plt.savefig(OUTPUT_TABLE_PNG, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📋 Comparison table saved: {OUTPUT_TABLE_CSV}, {OUTPUT_TABLE_PNG}")

    def generate_report(self):
        """Generate a detailed text report with workload-aware insight"""
        E_baseline = compute_energy(self.df_baseline, 'cpu_power', SAMPLING_INTERVAL_SEC)
        E_dvfs = compute_energy(self.df_dvfs, 'cpu_power', SAMPLING_INTERVAL_SEC)
        energy_savings = ((E_baseline - E_dvfs) / E_baseline) * 100 if E_baseline != 0 else 0.0
        ipc_change = ((self.df_dvfs['ipc'].mean() - self.df_baseline['ipc'].mean()) / self.df_baseline['ipc'].mean()) * 100
        l3_miss_change = ((self.df_baseline['l3_miss_rate'].mean() - self.df_dvfs['l3_miss_rate'].mean()) / self.df_baseline['l3_miss_rate'].mean()) * 100

        report = f"""
DVFS Controller Performance Analysis Report
==========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Workload: {self.workload_name}
Type: {self.workload_type.upper()}
Duration: {len(self.df_baseline) * SAMPLING_INTERVAL_SEC} seconds
Total Samples: {len(self.df_baseline)} (baseline), {len(self.df_dvfs)} (DVFS)

WORKLOAD CHARACTERIZATION
-------------------------
This is a {self.workload_type.replace('_', '-')} workload:
- High L3 miss rate and memory bandwidth → CPU stalls on memory
- Frequency scaling has limited impact on IPC
- Ideal strategy: Moderate frequency to save energy without hurting performance

EXECUTIVE SUMMARY
-----------------
- Energy Savings: {energy_savings:+.1f}% {'✅' if energy_savings > 0 else '❌'}
- IPC Change: {ipc_change:+.1f}% {'✅' if ipc_change > 0 else '❌'}
- L3 Miss Rate Reduction: {l3_miss_change:+.1f}% {'✅' if l3_miss_change > 0 else '❌'}

KEY METRICS
-----------
Baseline (Medium):
  Avg Power:     {self.df_baseline['cpu_power'].mean():.2f} W
  Avg IPC:       {self.df_baseline['ipc'].mean():.3f}
  Avg Freq:      {self.df_baseline['cpu_frequency'].mean()/1000:.2f} GHz
  Avg Temp:      {self.df_baseline['cpu_temperature'].mean():.1f} °C
  L3 Miss Rate:  {self.df_baseline['l3_miss_rate'].mean():.3f}

With HistGBDT:
  Avg Power:     {self.df_dvfs['cpu_power'].mean():.2f} W
  Avg IPC:       {self.df_dvfs['ipc'].mean():.3f}
  Avg Freq:      {self.df_dvfs['cpu_frequency'].mean()/1000:.2f} GHz
  Avg Temp:      {self.df_dvfs['cpu_temperature'].mean():.1f} °C
  L3 Miss Rate:  {self.df_dvfs['l3_miss_rate'].mean():.3f}

DVFS Level Distribution (With HistGBDT):
"""
        if 'dvfs_level' in self.df_dvfs.columns:
            level_counts = self.df_dvfs['dvfs_level'].value_counts()
            for level, count in level_counts.items():
                report += f"  {level}: {count} samples ({(count/len(self.df_dvfs))*100:.1f}%)\n"
        else:
            report += "  N/A (DVFS level not logged)\n"

        report += "\nCONCLUSION\n----------\n"
        if self.workload_type == "memory_intensive":
            if energy_savings > 5 and abs(ipc_change) < 3:
                report += "✅ The DVFS controller is effective: it reduces energy significantly with negligible performance loss (memory-bound workload)."
            elif energy_savings > 0 and ipc_change > -5:
                report += "🟡 The DVFS controller saves energy with acceptable IPC trade-off."
            else:
                report += "❌ The DVFS controller is too aggressive (or not targeted): it spends power without clear performance gain."
        else:
            if energy_savings > 5 and ipc_change > -2:
                report += "✅ The DVFS controller is highly effective: significant energy savings with minimal performance impact."
            elif energy_savings > 0 and ipc_change > -5:
                report += "🟡 The DVFS controller reduces energy with acceptable performance trade-off."
            else:
                report += "❌ The DVFS controller does not provide clear benefits under this workload."

        with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
            f.write(report)

        print(report)
        print(f"📝 Full report saved to: {OUTPUT_REPORT}")


def main():
    print("🚀 Starting Enhanced DVFS Analysis")
    print("=" * 60)

    try:
        df_baseline = load_json_log(BASELINE_FILE)
        df_dvfs = load_json_log(DVFS_FILE)
        print(f"Loaded {len(df_baseline)} baseline samples and {len(df_dvfs)} DVFS samples.")

        comparator = RobustDVFSComparator(df_baseline, df_dvfs, WORKLOAD_NAME)

        print("\n1. Generating comprehensive time-series plots...")
        comparator.create_comprehensive_plots()

        print("\n2. Generating box plots...")
        comparator.create_box_plots()

        print("\n3. Generating performance level plots...")
        comparator.create_performance_level_plots()

        print("\n4. Generating cache performance plot...")
        comparator.create_cache_performance_plot()

        print("\n5. Creating comparison table...")
        comparator.create_comparison_table()

        print("\n6. Generating analysis report...")
        comparator.generate_report()

        print("\n🎉 All analysis complete!")
        print(f"📊 Plots: {OUTPUT_PLOT_COMPREHENSIVE}, {OUTPUT_PLOT_BOX}, {OUTPUT_PLOT_LEVELS}, {OUTPUT_PLOT_CACHE}")
        print(f"📋 Data: {OUTPUT_TABLE_CSV}, {OUTPUT_TABLE_PNG}")
        print(f"📄 Report: {OUTPUT_REPORT}")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
