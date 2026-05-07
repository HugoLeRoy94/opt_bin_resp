#!/usr/bin/env python3

# %% Imports and Setup
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch
import numpy as np

# Ensure src is in the path whether run locally or in Docker
sys.path.append('../')
sys.path.append('/app')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.IO import SingleRunLoader
from src.environment import LigandEnvironment, SymmetricLigandEnvironment, LogNormalConcentration, NormalConcentration
from src.analysis_helper import plot_latent_umap

# Define the output directory of the run you wish to analyze
run_dir = "/app/data/single_run"

# %% 1. Load the configuration
if not os.path.exists(run_dir):
    print(f"Error: Directory {run_dir} does not exist.")
    sys.exit(1)

config_path = os.path.join(run_dir, "config.json")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    print("=== Configuration ===")
    for k, v in config_dict.items():
        print(f"{k}: {v}")
    print("=====================\n")
else:
    print("Warning: config.json not found. Proceeding...")

# %% 2. Plot Training Data
csv_path = os.path.join(run_dir, "stats.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print("Loaded stats.csv")
    
    x_col = 'epoch' if 'epoch' in df.columns else (df['step'] if 'step' in df.columns else df.index)
    
    for col in df.columns:
        if col in ['epoch', 'step']:
            continue
            
        plt.figure(figsize=(8, 5))
        plt.plot(x_col if isinstance(x_col, pd.Series) else df[x_col], df[col], linewidth=2, color='tab:blue')
        plt.title(f"Training: {col}", fontweight='bold')
        plt.xlabel("Epochs")
        plt.ylabel(col)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_plot = os.path.join(run_dir, f"train_plot_{col}.png")
        plt.savefig(out_plot)
        plt.show()
        print(f"✅ Saved training plot to {out_plot}")
else:
    print(f"No stats.csv found in {run_dir}.")

# %% 3. Plot Test Data Histograms
test_path = os.path.join(run_dir, "test_results.json")
if os.path.exists(test_path):
    with open(test_path, 'r') as f:
        test_results = json.load(f)
        
    print(f"\nLoaded test_results.json")
    for metric, values in test_results.items():
        if isinstance(values, list) and len(values) > 0:
            flat_values = np.array(values).flatten()
            plt.figure(figsize=(8, 5))
            plt.hist(flat_values, bins=30, edgecolor='black', alpha=0.7, color='tab:orange')
            plt.title(f"Test Distribution: {metric}", fontweight='bold')
            plt.xlabel(metric)
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            out_plot = os.path.join(run_dir, f"test_hist_{metric}.png")
            plt.savefig(out_plot)
            plt.show()
            print(f"✅ Saved test histogram to {out_plot}")
else:
    print(f"No test_results.json found in {run_dir}.")

# %% 4. Plot Latent Space UMAP
try:
    print("\nAttempting to reconstruct environment for UMAP plot...")
    loader = SingleRunLoader(run_dir)
    config = loader.load_config()
    checkpoint = loader.load_checkpoint(filename="best_model.pt", map_location="cpu")
    
    conc_type = getattr(config, 'conc_model_type', 'lognormal')
    if conc_type == 'normal':
        conc_model = NormalConcentration(n_ligands=config.n_ligands, init_mean=config.conc_mean, init_scale=config.conc_std)
    else:
        conc_model = LogNormalConcentration(n_ligands=config.n_ligands, init_mean=config.conc_mean, init_scale=config.conc_std)
        
    env_class = SymmetricLigandEnvironment if config.env_type == "symmetric" else LigandEnvironment
    env = env_class(n_units=config.n_units, n_families=config.n_families, conc_model=conc_model, n_ligands=config.n_ligands, p_presence=config.p_presence, noise_sigma=config.noise_sigma, latent_dim=config.latent_dim, shape_sigma=config.shape_sigma, avg_family_distance=config.average_family_distance, use_sensitivity=config.use_sensitivity)
    env.load_state_dict(checkpoint['env_state'])
    env.eval()
    
    receptor_indices = checkpoint['receptor_indices']
    
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_latent_umap(env, receptor_indices, ax=ax)
    out_plot = os.path.join(run_dir, "umap_latent_space.png")
    plt.savefig(out_plot, dpi=300)
    plt.show()
    print(f"✅ Saved UMAP latent space plot to {out_plot}")
except Exception as e:
    print(f"Could not generate UMAP plot. Ensure umap-learn and seaborn are installed. Error: {e}")
