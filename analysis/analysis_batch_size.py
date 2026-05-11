# %%
"""
Analysis of batch size sweep results.
Plots how the entropy of the array evolves with different batch sizes.

Directory layout (produced by batch_size_sweep.py + SweepRunner):
  ../data/batch_size_sweep/batch_size_sweep_{timestamp}/
    batch_size_{val}/sample_{id}/    ← one step (n_genes is scalar)
"""
import sys
sys.path.append('../')
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.IO import SweepLoader, find_latest_sweep

# %% Configuration
base_dir   = Path("../data/batch_size_sweep")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# %% Load data
print(f"Loading data from {base_dir}...")

if not base_dir.exists():
    print(f"Sweep directory {base_dir} not found.")
    df = None
else:
    sweep_root = find_latest_sweep(str(base_dir), prefix="batch_size_sweep")
    loader     = SweepLoader(sweep_root)
    df         = loader.load_all_histories()
    print(f"Loaded {len(df)} logged steps across {df.groupby(['batch_size', 'sample_id']).ngroups} runs")

# %% Derive final entropy and full trajectories per run
if df is not None and not df.empty and "full_array_entropy" in df.columns:

    # Final entropy = last logged value per run
    final = (
        df.sort_values("epoch")
          .groupby(["batch_size", "sample_id"])["full_array_entropy"]
          .last()
          .reset_index()
    )

    unique_batch_sizes = sorted(df["batch_size"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_batch_sizes)))

    # %% Plot 1: mean ± std entropy vs epoch for each batch size
    # Build per-(batch_size) mean trajectory from the logged epochs
    fig, ax = plt.subplots(figsize=(12, 7))

    for bs, color in zip(unique_batch_sizes, colors):
        sub = df[df["batch_size"] == bs].sort_values("epoch")
        grp = sub.groupby("epoch")["full_array_entropy"]
        epochs     = np.array(sorted(sub["epoch"].unique()))
        mean_traj  = grp.mean().values
        std_traj   = grp.std().fillna(0).values

        ax.plot(epochs, mean_traj, color=color, label=f"BS={bs}", linewidth=2)
        ax.fill_between(epochs, mean_traj - std_traj, mean_traj + std_traj,
                        color=color, alpha=0.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Entropy vs Epoch for Each Batch Size")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # %% Plot 2: histogram of all entropy values per batch size
    fig, ax = plt.subplots(figsize=(12, 7))

    for bs, color in zip(unique_batch_sizes, colors):
        vals = df.loc[df["batch_size"] == bs, "full_array_entropy"].dropna().values
        if len(vals) > 0:
            ax.hist(vals, bins=20, histtype="step", color=color,
                    label=f"BS={bs}", linewidth=2, density=True)

    ax.set_xlabel("Entropy (bits)")
    ax.set_ylabel("Density")
    ax.set_title("Full Entropy Trajectory Distribution for Each Batch Size")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # %% Plot 3: final entropy scatter vs batch size, coloured by sample
    fig, ax = plt.subplots(figsize=(12, 7))

    unique_samples = sorted(final["sample_id"].unique())
    sample_colors  = plt.cm.tab10(np.linspace(0, 1, len(unique_samples)))

    for sid, color in zip(unique_samples, sample_colors):
        sub = final[final["sample_id"] == sid]
        ax.scatter(sub["batch_size"], sub["full_array_entropy"],
                   color=color, label=f"Sample {sid}", alpha=0.7, s=80)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch Size (log₂ scale)")
    ax.set_ylabel("Final Entropy (bits)")
    ax.set_title("Final Entropy vs Batch Size (one point per sample)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()

else:
    print("No data to plot.")
