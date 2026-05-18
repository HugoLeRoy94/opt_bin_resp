# %%
"""
Analysis of batch-size sweep results.
Plots final array entropy vs batch_size, overlaid with the log2(batch_size)
ceiling and the coding/vocabulary bounds.

Setup (fixed in batch_size_sweep.py): N=10, D=20, M=500.
  Cover bound:      sum_{k=0}^{22} C(10,k) = 2^10 = 1024  →  10 bits (non-limiting)
  Vocabulary bound: M*(N+1) = 500*11 = 5500 > 1024         →  ~12.4 bits (non-limiting)
  Coding ceiling:   N = 10 bits
The only ceiling that varies across the sweep is log2(batch_size).

Directory layout (produced by batch_size_sweep.py + SweepRunner):
  ../data/batch_size_sweep/batch_size_sweep_{timestamp}/
    batch_size_{val}/sample_{id}/
"""
import sys
sys.path.append('../')
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt

from src.IO import SweepLoader, find_latest_sweep

# %% Configuration
base_dir   = Path("../data/batch_size_sweep")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

N = 20    # n_genes fixed in the sweep
M = 100   # n_ligands fixed in the sweep

# %% Load data
print(f"Loading data from {base_dir}...")

if not base_dir.exists():
    print(f"Sweep directory {base_dir} not found.")
    df = None
else:
    sweep_root = find_latest_sweep(str(base_dir), prefix="batch_size_sweep")[0]
    print(sweep_root)
    loader   = SweepLoader(sweep_root)
    df_test  = loader.load_all_test_results()
    df_train = loader.load_all_histories()
    print(f"Loaded test results for {len(df_test)} runs across "
          f"{df_test['batch_size'].nunique()} batch sizes")

# %% 2. Load and print configuration
config_path = sweep_root + "/sweep_config.json"
with open(config_path) as f:
    config_dict = json.load(f)
print("\n=== Configuration ===")
for k, v in config_dict.items():
    print(f"  {k}: {v}")
print("=====================\n")
# %%
if df_test is not None and not df_test.empty and "full_array_entropy" in df_test.columns:

    batch_sizes = np.array(sorted(df_test["batch_size"].unique()))

    metrics = {
        "full_array_entropy":         ("steelblue",  "solid",  "unblocked"),
        "full_array_entropy_blocked":  ("darkorange", "dashed", "blocked"),
    }

    # Plot 1: test entropy vs batch_size — one curve per metric
    fig, ax = plt.subplots(figsize=(10, 6))

    for col, (color, ls, label) in metrics.items():
        if col not in df_test.columns:
            continue
        grp  = df_test.groupby("batch_size")[col]
        mean = np.array([grp.get_group(bs).mean() for bs in batch_sizes])
        std  = np.array([grp.get_group(bs).std(ddof=0) for bs in batch_sizes])
        ax.plot(batch_sizes, mean, marker="o", color=color, linestyle=ls,
                linewidth=2, label=label)
        ax.fill_between(batch_sizes, mean - std, mean + std, color=color, alpha=0.15)

    # Batch-size ceiling: log2(batch_size)
    bs_ref = np.geomspace(batch_sizes[0], batch_sizes[-1], 200)
    ax.plot(bs_ref, np.log2(bs_ref), "--", color="tomato", linewidth=1.5,
            label=r"$\log_2(\mathrm{batch\_size})$ ceiling")

    # Coding ceiling: N bits
    ax.axhline(N, color="green", linestyle=":", linewidth=1.5,
               label=f"Coding ceiling  N = {N} bits")

    # Vocabulary bound: log2(M*(N+1))
    vb = np.log2(M * (N + 1))
    ax.axhline(vb, color="orange", linestyle="-.", linewidth=1.5,
               label=rf"Vocabulary bound  $\log_2({M}\times{N+1})$ $\approx$ {vb:.1f} bits")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size (log₂ scale)")
    ax.set_ylabel("Test entropy (bits)")
    ax.set_title(
        f"Test array entropy vs training batch size  (N={N}, D=20, M={M})\n"
        "transition expected near batch_size ≈ 2^N = 1024"
    )
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(output_dir / "test_entropy_vs_batch_size.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Plot 2: training entropy vs epoch — one curve per batch_size
    if df_train is not None and not df_train.empty:
        bs_colors = plt.cm.viridis(np.linspace(0, 1, len(batch_sizes)))
        fig, ax = plt.subplots(figsize=(10, 6))

        for bs, color in zip(batch_sizes, bs_colors):
            sub = df_train[df_train["batch_size"] == bs].sort_values("epoch")
            grp = sub.groupby("epoch")["full_array_entropy"]
            epochs    = np.array(sorted(sub["epoch"].unique()))
            mean_traj = grp.mean().values
            std_traj  = grp.std().fillna(0).values

            ax.plot(epochs, mean_traj, color=color, label=f"BS={bs}", linewidth=1.5)
            ax.fill_between(epochs, mean_traj - std_traj, mean_traj + std_traj,
                            color=color, alpha=0.15)

        ax.axhline(N, color="green", linestyle=":", linewidth=1.5, label=f"N={N} bits")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Entropy (bits)")
        ax.set_title(f"Training entropy vs Epoch per Batch Size  (N={N}, D=20, M={M})")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="x-small")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "entropy_vs_epoch_per_batch_size.png", dpi=150, bbox_inches="tight")
        plt.show()

else:
    print("No data to plot.")

# %%
