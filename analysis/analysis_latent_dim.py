# %%
"""
Analysis of latent dimension sweep results.
Plots how entropy evolves with latent dimension and n_genes.

Directory layout (produced by latent_dim_sweep.py + SweepRunner):
  ../data/latent_dim_sweep/latent_dim_sweep_{timestamp}/
    latent_dim_{val}/sample_{id}/n_genes_{val}/
"""
import sys
sys.path.append('../')
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.IO import SweepLoader, find_latest_sweep

# %% Configuration
base_dir   = Path("../data/latent_dim_sweep")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# %% Load data
print(f"Loading data from {base_dir}...")

if not base_dir.exists():
    print(f"Directory {base_dir} not found.")
    df = None
else:
    sweep_root = find_latest_sweep(str(base_dir), prefix="latent_dim_sweep")
    loader     = SweepLoader(sweep_root)
    df         = loader.load_all_histories()
    print(f"Loaded {len(df)} logged steps across "
          f"{df.groupby(['latent_dim', 'n_genes', 'sample_id']).ngroups} runs")

# %%
if df is not None and not df.empty and "full_array_entropy" in df.columns:

    # Final entropy per run
    final = (
        df.sort_values("epoch")
          .groupby(["latent_dim", "n_genes", "sample_id"])["full_array_entropy"]
          .last()
          .reset_index()
    )

    latent_dims   = sorted(final["latent_dim"].unique())
    n_genes_vals  = sorted(final["n_genes"].unique())
    colors_ld     = plt.cm.viridis(np.linspace(0, 1, len(latent_dims)))
    colors_ng     = plt.cm.viridis(np.linspace(0, 1, len(n_genes_vals)))

    # %% 2×2 grid: mean / std of final entropy vs n_genes and vs latent_dim
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def _agg(group_col, x_col, values_col, agg_fn):
        out = {}
        for gval in sorted(final[group_col].unique()):
            xs, ys = [], []
            for xval in sorted(final[x_col].unique()):
                sub = final[(final[group_col] == gval) & (final[x_col] == xval)][values_col].dropna()
                if len(sub):
                    xs.append(xval); ys.append(agg_fn(sub))
            if xs:
                out[gval] = (np.array(xs), np.array(ys))
        return out

    mean_by_ld_ng = _agg("latent_dim", "n_genes",  "full_array_entropy", np.mean)
    mean_by_ng_ld = _agg("n_genes",    "latent_dim","full_array_entropy", np.mean)
    std_by_ld_ng  = _agg("latent_dim", "n_genes",  "full_array_entropy", np.std)
    std_by_ng_ld  = _agg("n_genes",    "latent_dim","full_array_entropy", np.std)

    for (ax, data, colors, xlabel, title) in [
        (axes[0,0], mean_by_ld_ng, colors_ld, "Number of genes",   "Mean entropy vs n_genes"),
        (axes[0,1], mean_by_ng_ld, colors_ng, "Latent dimension",  "Mean entropy vs latent_dim"),
        (axes[1,0], std_by_ld_ng,  colors_ld, "Number of genes",   "Std entropy vs n_genes"),
        (axes[1,1], std_by_ng_ld,  colors_ng, "Latent dimension",  "Std entropy vs latent_dim"),
    ]:
        for (gval, (xs, ys)), color in zip(sorted(data.items()), colors):
            label = (f"latent_dim={gval}" if ax in (axes[0,0], axes[1,0])
                     else f"n_genes={gval}")
            ax.plot(xs, ys, marker="o", color=color, label=label, linewidth=2)
        ax.set_xlabel(xlabel); ax.set_ylabel("Entropy (bits)")
        ax.set_title(title); ax.legend(fontsize="small"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # %% Entropy vs epoch — one subplot per latent_dim, one curve per n_genes
    max_epoch = df["epoch"].max()

    fig, axes = plt.subplots(len(latent_dims), 1, figsize=(12, 4 * len(latent_dims)))
    if len(latent_dims) == 1:
        axes = [axes]

    for ax, ld in zip(axes, latent_dims):
        sub_ld = df[df["latent_dim"] == ld]
        for ng, color in zip(n_genes_vals, colors_ng):
            sub = sub_ld[sub_ld["n_genes"] == ng].sort_values("epoch")
            grp = sub.groupby("epoch")["full_array_entropy"]
            epochs    = np.array(sorted(sub["epoch"].unique()))
            mean_traj = grp.mean().values
            std_traj  = grp.std().fillna(0).values

            ax.plot(epochs, mean_traj, color=color, label=f"n_genes={ng}", linewidth=2)
            ax.fill_between(epochs, mean_traj - std_traj, mean_traj + std_traj,
                            color=color, alpha=0.2)

        ax.set_xlabel("Epoch"); ax.set_ylabel("Entropy (bits)")
        ax.set_title(f"Entropy vs Epoch  (latent_dim={ld})")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

else:
    print("No data to plot.")
