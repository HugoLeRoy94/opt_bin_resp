# %%
"""
Analysis of cover-bound sharpness sweep results.
Plots final array entropy vs latent_dim per n_genes, overlaid with the
theoretical Cover bound  sum_{k=0}^{D+2} C(N,k)  and the coding ceiling N bits.

Directory layout (produced by latent_dim_sweep.py + SweepRunner):
  ../data/cover_bound_sharpness/cover_bound_sharpness_{timestamp}/
    latent_dim_{val}/sample_{id}/n_genes_{val}/
"""
import sys
sys.path.append('../')
from math import comb as math_comb
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.IO import SweepLoader, find_latest_sweep

# %% Configuration
base_dir   = Path("../data/cover_bound_sharpness")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

M = 100  # n_ligands, for vocabulary bound

def cover_bound_bits(N: int, D: int) -> float:
    """log2 of Cover-bound cell count: min(N, log2(sum_{k=0}^{D+2} C(N,k)))."""
    cells = sum(math_comb(N, k) for k in range(D + 3))
    return min(float(N), np.log2(max(1, cells)))

def vocab_bound_bits(N: int, M: int = 100) -> float:
    """log2(M*(N+1)): vocabulary × concentration-axis bound."""
    return np.log2(M * (N + 1))

# %% Load data
print(f"Loading data from {base_dir}...")

if not base_dir.exists():
    print(f"Directory {base_dir} not found.")
    df = None
else:
    sweep_root = find_latest_sweep(str(base_dir), prefix="cover_bound_sharpness")
    loader     = SweepLoader(sweep_root)
    df         = loader.load_all_histories()
    print(f"Loaded {len(df)} logged steps across "
          f"{df.groupby(['latent_dim', 'n_genes', 'sample_id']).ngroups} runs")

# %%
if df is not None and not df.empty and "full_array_entropy" in df.columns:

    final = (
        df.sort_values("epoch")
          .groupby(["latent_dim", "n_genes", "sample_id"])["full_array_entropy"]
          .last()
          .reset_index()
    )

    latent_dims  = sorted(final["latent_dim"].unique())
    n_genes_vals = sorted(final["n_genes"].unique())
    colors_ng    = plt.cm.viridis(np.linspace(0, 1, len(n_genes_vals)))

    # %% Plot 1: final entropy vs latent_dim per n_genes + theoretical bounds
    fig, ax = plt.subplots(figsize=(10, 6))

    for ng, color in zip(n_genes_vals, colors_ng):
        sub = final[final["n_genes"] == ng]
        lds = np.array(sorted(sub["latent_dim"].unique()))
        grp = sub.groupby("latent_dim")["full_array_entropy"]
        mean = np.array([grp.get_group(d).mean() for d in lds])
        std  = np.array([grp.get_group(d).std(ddof=0) for d in lds])

        ax.plot(lds, mean, marker="o", color=color, linewidth=2, label=f"N={ng}")
        ax.fill_between(lds, mean - std, mean + std, color=color, alpha=0.15)

        # Cover bound (dashed, same color)
        cb = np.array([cover_bound_bits(ng, d) for d in lds])
        ax.plot(lds, cb, "--", color=color, alpha=0.55, linewidth=1.5)

    # Vocabulary bound per N (dotted, same color) — if below coding ceiling, it is the tighter cap
    for ng, color in zip(n_genes_vals, colors_ng):
        vb = vocab_bound_bits(ng, M)
        if vb < ng:  # only draw when vocab bound is tighter than the coding ceiling
            ax.axhline(vb, color=color, linestyle=":", alpha=0.4, linewidth=1)

    ax.set_xlabel("Latent dimension D")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title(
        "Final array entropy vs latent dimension\n"
        "solid = measured (mean ± std)  |  dashed = Cover bound  |  dotted = vocabulary bound (if tighter)"
    )
    ax.legend(fontsize="small", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "final_entropy_vs_latent_dim.png", dpi=150, bbox_inches="tight")
    plt.show()

    # %% Plot 2: entropy vs epoch — one subplot per latent_dim, curves per n_genes
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

            ax.plot(epochs, mean_traj, color=color, label=f"N={ng}", linewidth=2)
            ax.fill_between(epochs, mean_traj - std_traj, mean_traj + std_traj,
                            color=color, alpha=0.2)
            ax.axhline(cover_bound_bits(ng, ld), color=color, linestyle="--",
                       alpha=0.35, linewidth=1)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Entropy (bits)")
        ax.set_title(f"Entropy vs Epoch  (D={ld})")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "entropy_vs_epoch_per_latent_dim.png", dpi=150, bbox_inches="tight")
    plt.show()

else:
    print("No data to plot.")
