# %%
"""
Analysis of family distance sweep results.
Plots final entropy vs average_family_distance, one curve per family_spread,
in a grid of (n_families × n_genes) panels.

Directory layout (produced by fam_distances.py + SweepRunner):
  ../data/fam_distance_sweep/fam_distances_{timestamp}/
    average_family_distance_{val}/family_spread_{val}/latent_dim_{val}/
    n_families_{val}/sample_{id}/n_genes_{val}/
"""
import sys
sys.path.append('../')
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.IO import SweepLoader, find_latest_sweep

# %% Configuration
base_dir   = Path("../data/fam_distance_sweep")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# %% Load data
print(f"Loading data from {base_dir}...")

if not base_dir.exists():
    print(f"Directory {base_dir} not found.")
    df = None
else:
    sweep_root = find_latest_sweep(str(base_dir), prefix="fam_distances")
    loader     = SweepLoader(sweep_root)
    df         = loader.load_all_histories()
    print(f"Loaded {len(df)} logged steps")

# %%
if df is not None and not df.empty and "full_array_entropy" in df.columns:

    # Final entropy = last logged value per run
    group_keys = ["n_families", "family_spread", "average_family_distance",
                  "n_genes", "sample_id"]
    final = (
        df.sort_values("epoch")
          .groupby(group_keys)["full_array_entropy"]
          .last()
          .reset_index()
    )

    unique_n_families   = sorted(final["n_families"].unique())
    unique_n_genes      = sorted(final["n_genes"].unique())
    unique_family_spreads = sorted(final["family_spread"].unique())
    unique_fam_distances  = sorted(final["average_family_distance"].unique())

    n_rows = len(unique_n_families)
    n_cols = len(unique_n_genes)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_family_spreads)))

    # %% Grid: rows = n_families, cols = n_genes
    # Each cell: entropy vs average_family_distance, one curve per family_spread
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # Normalise axes to always be 2-D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, nf in enumerate(unique_n_families):
        for col_idx, ng in enumerate(unique_n_genes):
            ax = axes[row_idx, col_idx]

            for ss, color in zip(unique_family_spreads, colors):
                sub = final[
                    (final["n_families"] == nf) &
                    (final["n_genes"]    == ng)  &
                    (final["family_spread"] == ss)
                ]
                if sub.empty:
                    continue

                fd_means, fd_stds, fd_list = [], [], []
                for fd in unique_fam_distances:
                    vals = sub.loc[sub["average_family_distance"] == fd,
                                   "full_array_entropy"].dropna()
                    if len(vals):
                        fd_list.append(fd)
                        fd_means.append(vals.mean())
                        fd_stds.append(vals.std())

                if fd_list:
                    ax.errorbar(fd_list, fd_means, yerr=fd_stds,
                                marker="o", capsize=5, color=color,
                                label=f"spread={ss}", linewidth=2, capthick=2)

            ax.set_xlabel("Average family distance")
            ax.set_ylabel("Final entropy (bits)")
            ax.set_title(f"n_families={nf},  n_genes={ng}")
            if col_idx == n_cols - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

else:
    print("No data to plot.")
