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
    # This sets the default font size for most text elements
    plt.rcParams.update({
        'font.size': 14,          
        'axes.titlesize': 16,     
        'axes.labelsize': 16,     
        'xtick.labelsize': 14,    
        'ytick.labelsize': 14,    
        'legend.fontsize': 14     
    })

    print("Loading model and data...")
    # Adjust these paths if your folders are named differently
    model_path = Path("models/histgbdt.joblib")
    data_path = Path("dataset/X_test.npy")

    if not model_path.exists() or not data_path.exists():
        print("Error: Could not find model or dataset. Check your paths!")
        return

    # Load the trained HistGBDT model and the test data
    model = joblib.load(model_path)
    X_test = np.load(data_path)

    # SHAP can take a long time on huge datasets. 
    # We will just use the first 500 samples to keep it fast.
    X_sample = X_test[:500]

    print("Calculating SHAP values...")
    # TreeExplainer is highly optimized for HistGBDT and Random Forests
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    # Tell the SHAP values what the feature names are
    shap_values.feature_names = FEATURE_ORDER

    # ---------------------------------------------------------
    # PLOT 1: The Summary Plot (Global View)
    # ---------------------------------------------------------
    print("Generating Summary Plot...")
    # Set a large figure size to accommodate larger fonts without crowding
    plt.figure(figsize=(14, 10))
    
    # Generate the SHAP summary plot
    if isinstance(shap_values, list) or len(shap_values.shape) == 3:
        # Plot for Class 2 ("High")
        shap.summary_plot(shap_values[:, :, 2], X_sample, feature_names=FEATURE_ORDER, show=False)
    else:
        shap.summary_plot(shap_values, X_sample, feature_names=FEATURE_ORDER, show=False)
    
    # --- CUSTOMIZE COLORBAR AND LABELS ---
    # Retrieve the figure to access its axes
    fig = plt.gcf()
    
    # The summary plot adds the colorbar as the last axis (index -1)
    if len(fig.axes) > 1:
        cbar = fig.axes[-1]
        cbar.tick_params(labelsize=14)           # Increase "High" / "Low" label size
        cbar.set_ylabel("Feature value", fontsize=16) # Increase "Feature value" text size

    # Update axis labels and ticks explicitly
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=14) # Feature name labels
    ax.tick_params(axis='x', labelsize=14) # X-axis numbers
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=16)
    
    plt.title("SHAP Summary Plot - Feature Importance", fontsize=16)
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # PLOT 2: The Waterfall/Cascade Plot (Local View)
    # ---------------------------------------------------------
    print("Generating Waterfall Plot for Prediction #0...")
    # Using a slightly different layout for waterfall plots
    plt.figure(figsize=(12, 8))
    
    # Plot the cascade for the first sample
    if isinstance(shap_values, list) or len(shap_values.shape) == 3:
        shap.plots.waterfall(shap_values[:, :, 2][0], show=False)
    else:
        shap.plots.waterfall(shap_values[0], show=False)
        
    plt.title("SHAP Waterfall Plot - Explaining a single decision", fontsize=18)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()