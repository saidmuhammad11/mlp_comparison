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

    print("Calculating SHAP values (this might take a few seconds)...")
    # TreeExplainer is highly optimized for HistGBDT and Random Forests
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    # Tell the SHAP values what the feature names are
    shap_values.feature_names = FEATURE_ORDER

    # ---------------------------------------------------------
    # PLOT 1: The Summary Plot (Global View)
    # ---------------------------------------------------------
    print("Generating Summary Plot...")
    plt.figure(figsize=(10, 6))
    plt.title("SHAP Summary Plot - Feature Importance")
    
    # Because this is a multi-class model (Low, Medium, High), 
    # we tell SHAP to plot the impact on Class 2 (High Power) as an example.
    # (If shap_values is a list, it handles classes differently depending on the version)
    if isinstance(shap_values, list) or len(shap_values.shape) == 3:
        # For multi-class, plot the summary for Class 2 ("High")
        shap.summary_plot(shap_values[:, :, 2], X_sample, feature_names=FEATURE_ORDER, show=False)
    else:
        shap.summary_plot(shap_values, X_sample, feature_names=FEATURE_ORDER, show=False)
    
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # PLOT 2: The Waterfall/Cascade Plot (Local View)
    # ---------------------------------------------------------
    print("Generating Waterfall Plot for Prediction #0...")
    plt.figure(figsize=(10, 6))
    plt.title("SHAP Waterfall Plot - Explaining a single decision")
    
    # Plot the cascade for the very first sample in our test data for Class 2
    if isinstance(shap_values, list) or len(shap_values.shape) == 3:
        shap.plots.waterfall(shap_values[:, :, 2][0], show=False)
    else:
        shap.plots.waterfall(shap_values[0], show=False)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()