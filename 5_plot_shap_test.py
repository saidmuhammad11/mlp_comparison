#!/usr/bin/env python3
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Define the exact features used in your model
FEATURE_ORDER = [
    "ipc", "l2_miss_rate", "l3_miss_rate", "memory_bandwidth", 
    "cpu_usage_overall", "cpu_temperature", "cpu_power", "cpu_frequency", 
    "ipc_change", "ipc_avg_5", "l3_miss_rate_avg_5", "bw_util"
]

def main():
    # --- GLOBAL FONT CONFIGURATION ---
    plt.rcParams.update({
        'font.size': 14,          
        'axes.titlesize': 18,     
        'axes.labelsize': 16,     
        'xtick.labelsize': 14,    
        'ytick.labelsize': 14,    
        'legend.fontsize': 14     
    })

    print("Loading model and data...")
    model_path = Path("models/histgbdt.joblib")
    data_path = Path("dataset/X_test.npy")

    if not model_path.exists() or not data_path.exists():
        print("Error: Could not find model or dataset. Check your paths!")
        return

    model = joblib.load(model_path)
    X_test = np.load(data_path)
    X_sample = X_test[:500]

    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    shap_values.feature_names = FEATURE_ORDER

    # ---------------------------------------------------------
    # PLOT 1: The Summary Plot (Global View)
    # ---------------------------------------------------------
    print("Generating Summary Plot...")
    plt.figure(figsize=(9, 8))
    
    if isinstance(shap_values, list) or len(shap_values.shape) == 3:
        shap.summary_plot(shap_values[:, :, 2], X_sample, feature_names=FEATURE_ORDER, show=False)
    else:
        shap.summary_plot(shap_values, X_sample, feature_names=FEATURE_ORDER, show=False)
    
    # Customizing Axes
    ax = plt.gca()
    for label in ax.get_yticklabels():
        label.set_fontsize(16)
    
    ax.tick_params(axis='x', labelsize=14)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=16)

    # Customize colorbar
    fig = plt.gcf()
    if len(fig.axes) > 1:
        cbar = fig.axes[-1]
        cbar.tick_params(labelsize=14)
        cbar.set_ylabel("Feature value", fontsize=16)
    
    plt.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.15)
    plt.show()

    # ---------------------------------------------------------
    # PLOT 2: The Waterfall/Cascade Plot
    # ---------------------------------------------------------
    print("Generating Waterfall Plot for Prediction #0...")
    plt.figure(figsize=(9, 8))
    
    if isinstance(shap_values, list) or len(shap_values.shape) == 3:
        shap.plots.waterfall(shap_values[:, :, 2][0], show=False)
    else:
        shap.plots.waterfall(shap_values[0], show=False)
        
    # --- FIX: FORCE WATERFALL FONT RESIZING ---
    ax = plt.gca()
    
    # 1. Increase Feature Name labels (Y-axis)
    for label in ax.get_yticklabels():
        label.set_fontsize(16)
        
    # 2. Increase Number labels inside the bars
    # SHAP creates these as 'texts' objects on the axes
    for text in ax.texts:
        text.set_fontsize(16)
        
    # 3. Increase X-axis tick labels
    ax.tick_params(axis='x', labelsize=14)
    
    # Add a title if desired
    # plt.title("SHAP Waterfall Plot - Explaining a single decision", fontsize=18)
    
    plt.subplots_adjust(left=0.35, right=0.95, top=0.9, bottom=0.1)
    plt.show()

if __name__ == "__main__":
    main()