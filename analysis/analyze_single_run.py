# %%
"""
Analysis of a single training run.
Loads the most recent run fro
m ../data/single_run/ and plots:
  - Training curves (stats.csv)
  - Test result histograms (test_results.json)
  - Latent space UMAP and radar chart (if env can be reconstructed)
"""
import os
import sys
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append('../')
sys.path.append('/app')

from src.IO import SweepLoader, SingleRunLoader, find_latest_sweep
from src.environment import LigandEnvironment, SymmetricLigandEnvironment, LogNormalConcentration, NormalConcentration

# %% 1. Locate the run directory
# single_run.py writes to: ../data/single_run_{timestamp}/sample_0/
base_dir = Path("../data")

sweep_root = find_latest_sweep(str(base_dir), prefix="single_run")[0]
run_dir    = Path(sweep_root) / "sample_0"

print(f"Analysing run: {run_dir}")
if not run_dir.exists():
    print(f"Error: {run_dir} does not exist.")
    sys.exit(1)

# %% 2. Load and print configuration
config_path = run_dir / "config.json"
if config_path.exists():
    with open(config_path) as f:
        config_dict = json.load(f)
    print("\n=== Configuration ===")
    for k, v in config_dict.items():
        print(f"  {k}: {v}")
    print("=====================\n")
    
# %% 3. Plot training curves
stats_path = run_dir / "stats.csv"
if stats_path.exists():
    df = pd.read_csv(stats_path)
    print(f"Loaded stats.csv  ({len(df)} rows)")

    x_col = "epoch" if "epoch" in df.columns else df.index
    x_vals = df[x_col] if isinstance(x_col, str) else x_col
    

    # Plot full array entropy over optimization
    if "full_array_entropy" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_vals, df["full_array_entropy_blocked"], linewidth=2)
        ax.plot(x_vals, df["full_array_entropy"], linewidth=2, color="tab:blue")
        ax.set_title("Full Array Entropy over Optimization", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Full Array Entropy")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = run_dir / "train_plot_full_array_entropy.png"
        plt.savefig(out); plt.show()
        print(f"Saved {out}")

    # Plot largest dist_rank (last dist_rank_N column) over optimization
    dist_cols = sorted([c for c in df.columns if c.startswith("dist_rank_")],
                       key=lambda c: int(c.split("_")[-1]))
    if dist_cols:
        largest_dist_col = dist_cols[-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_vals, df[largest_dist_col], linewidth=2, color="tab:orange")
        ax.set_title(f"Largest Distance ({largest_dist_col}) over Optimization", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Energy Gap (largest rank)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = run_dir / f"train_plot_{largest_dist_col}.png"
        plt.savefig(out); plt.show()
        print(f"Saved {out}")
else:
    print("No stats.csv found.")

# %% 4. Plot test result histograms
test_path = run_dir / "test_results.json"
if test_path.exists():
    with open(test_path) as f:
        test_results = json.load(f)
    for metric, values in test_results.items():
        if isinstance(values, list) and len(values) > 0:
            flat = np.array(values).flatten()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(flat, bins="auto", edgecolor="black", alpha=0.7, color="tab:orange")
            ax.set_title(f"Test distribution: {metric}", fontweight="bold")
            ax.set_xlabel(metric); ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out = run_dir / f"test_hist_{metric}.png"
            plt.savefig(out); plt.show()
            print(f"Saved {out}")
else:
    print("No test_results.json found.")

# %% 5. Reconstruct environment for latent space plots
env = None
receptor_indices = None

try:
    print("\nReconstructing environment for latent space plots...")
    loader     = SingleRunLoader(str(run_dir))
    config     = loader.load_config()
    checkpoint = loader.load_checkpoint(filename="best_model.pt", map_location="cpu")

    conc_cls = NormalConcentration if config.conc_model_type == "normal" else LogNormalConcentration
    conc_model = conc_cls(
        n_ligands=config.n_ligands,
        init_mean=config.conc_mean,
        init_scale=config.conc_std,
    )

    env_class = (SymmetricLigandEnvironment
                 if config.environment_geometry == "symmetric"
                 else LigandEnvironment)
    env = env_class(
        config.n_genes,
        config.n_families,
        conc_model=conc_model,
        n_ligands=config.n_ligands,
        p_presence=config.p_presence,
        observation_noise_sigma=config.observation_noise_sigma,
        latent_dim=config.latent_dim,
        family_spread=config.family_spread,
        avg_family_distance=config.average_family_distance,
        affinity_kernel=config.affinity_kernel,
        kernel_params=config.kernel_params,
        distribution_type=config.distribution_type,
    )
    env.load_state_dict(checkpoint["env_state"])
    env.eval()

    receptor_indices = checkpoint["receptor_indices"]
    print("Environment reconstructed successfully.")
except Exception as e:
    print(f"Could not reconstruct environment: {e}")

# %% 6. UMAP and radar chart
if env is not None:
    try:
        from src.analysis_helper import plot_latent_umap
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_latent_umap(env, receptor_indices, ax=ax)
        out = run_dir / "umap_latent_space.png"
        plt.savefig(out, dpi=300); plt.show()
        print(f"Saved {out}")
    except Exception as e:
        print(f"Could not generate UMAP: {e}")

    try:
        from src.analysis_helper import plot_latent_radar_chart
        fig, ax = plot_latent_radar_chart(env, receptor_indices)
        out = run_dir / "radar_chart.png"
        plt.savefig(out, dpi=300, bbox_inches="tight"); plt.show()
        print(f"Saved {out}")
    except Exception as e:
        print(f"Could not generate radar chart: {e}")

# %%
