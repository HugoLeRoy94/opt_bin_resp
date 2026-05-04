#!/usr/bin/env python3

# %% Imports and Setup
import os
import pandas as pd
import matplotlib.pyplot as plt
import json

# Define the output directory of the run you wish to analyze
run_dir = "/app/data/single_run"

# %% 1. Load the configuration
if not os.path.exists(run_dir):
    print(f"Error: Directory {run_dir} does not exist.")
else:
    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("=== Configuration ===")
        for k, v in config.items():
            print(f"{k}: {v}")
        print("=====================\n")
    else:
        print("Warning: config.json not found. Proceeding with CSV parsing.")

# %% 2. Load the metrics and Plot
if os.path.exists(run_dir):
    csv_files = [f for f in os.listdir(run_dir) if f.endswith('.csv')]
        
    if not csv_files:
        print(f"No CSV data files found in {run_dir}.")
    else:
        for csv_file in csv_files:
            csv_path = os.path.join(run_dir, csv_file)
            df = pd.read_csv(csv_path)
            print(f"Loaded {csv_file} (showing first 5 rows):")
            print(df.head())
            print("-" * 40)
            
            # 3. Plot standard metrics automatically
            plt.figure(figsize=(10, 6))
            
            # Extract common metrics for visualization
            plot_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ["loss", "entropy", "mutual_information", "mi", "distance"])]
            
            if plot_cols:
                x_axis = df['epoch'] if 'epoch' in df.columns else (df['step'] if 'step' in df.columns else df.index)
                for col in plot_cols:
                    plt.plot(x_axis, df[col], label=col, alpha=0.8)
                
                plt.title(f"Optimization Metrics ({csv_file})")
                plt.xlabel("Epochs / Steps")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True)
                
                out_plot = os.path.join(run_dir, f"plot_{os.path.splitext(csv_file)[0]}.png")
                plt.savefig(out_plot)
                plt.show() # Renders inline in an interactive window
                print(f"✅ Saved plot overview to {out_plot}")
            else:
                print(f"No standard metrics (loss, entropy, mi) found to plot automatically in {csv_file}.")