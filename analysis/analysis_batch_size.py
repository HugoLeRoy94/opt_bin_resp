# %%
"""
Analysis of batch-size sweep results.
Plots final array entropy vs batch_size, overlaid with the log2(batch_size)
ceiling and the coding/vocabulary bounds.  One curve per family_spread value.

Setup (fixed in batch_size_sweep.py): N=10, D=20, M=500.
  Cover bound:      sum_{k=0}^{22} C(10,k) = 2^10 = 1024  →  10 bits (non-limiting)
  Vocabulary bound: M*(N+1) = 500*11 = 5500 > 1024         →  ~12.4 bits (non-limiting)
  Coding ceiling:   N = 10 bits
The only ceiling that varies across the sweep is log2(batch_size).
family_spread controls the width of the receptor-family distribution.

Directory layout (produced by batch_size_sweep.py + SweepRunner):
  ../data/batch_size_sweep/batch_size_sweep_{timestamp}/
    batch_size_{val}_family_spread_{val}/sample_{id}/
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

N = 10    # n_genes fixed in the sweep
M = 500   # n_ligands fixed in the sweep

# %% Load data
print(f"Loading data from {base_dir}...")

if not base_dir.exists():
    print(f"Sweep directory {base_dir} not found.")
    df = None
else:
    sweep_root = find_latest_sweep(str(base_dir), prefix="batch_size_sweep")
    loader     = SweepLoader(sweep_root)
    df         = loader.load_all_histories()
    print(f"Loaded {len(df)} logged steps across "
          f"{df.groupby(['batch_size', 'family_spread', 'sample_id']).ngroups} runs")

# %%
if df is not None and not df.empty and "full_array_entropy" in df.columns:

    group_cols = ["batch_size", "family_spread", "sample_id"]
    final = (
        df.sort_values("epoch")
          .groupby(group_cols)["full_array_entropy"]
          .last()
          .reset_index()
    )

    batch_sizes    = np.array(sorted(final["batch_size"].unique()))
    family_spreads = sorted(final["family_spread"].unique())
    spread_colors  = plt.cm.plasma(np.linspace(0.1, 0.9, len(family_spreads)))

    # Plot 1: final entropy vs batch_size — one curve per family_spread
    fig, ax = plt.subplots(figsize=(10, 6))

    for fs, color in zip(family_spreads, spread_colors):
        sub  = final[final["family_spread"] == fs]
        grp  = sub.groupby("batch_size")["full_array_entropy"]
        mean = np.array([grp.get_group(bs).mean() for bs in batch_sizes])
        std  = np.array([grp.get_group(bs).std(ddof=0) for bs in batch_sizes])

        ax.plot(batch_sizes, mean, marker="o", color=color, linewidth=2,
                label=f"spread={fs}")
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
    ax.set_ylabel("Final entropy (bits)")
    ax.set_title(
        f"Final array entropy vs training batch size  (N={N}, D=20, M={M})\n"
        "transition expected near batch_size ≈ 2^N = 1024"
    )
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(output_dir / "final_entropy_vs_batch_size.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Plot 2: entropy vs epoch — one subplot per family_spread, one curve per batch_size
    bs_colors = plt.cm.viridis(np.linspace(0, 1, len(batch_sizes)))

    fig, axes = plt.subplots(1, len(family_spreads),
                             figsize=(6 * len(family_spreads), 6),
                             sharey=True)
    if len(family_spreads) == 1:
        axes = [axes]

    for ax, fs in zip(axes, family_spreads):
        sub_fs = df[df["family_spread"] == fs]
        for bs, color in zip(batch_sizes, bs_colors):
            sub = sub_fs[sub_fs["batch_size"] == bs].sort_values("epoch")
            grp = sub.groupby("epoch")["full_array_entropy"]
            epochs    = np.array(sorted(sub["epoch"].unique()))
            mean_traj = grp.mean().values
            std_traj  = grp.std().fillna(0).values

            ax.plot(epochs, mean_traj, color=color, label=f"BS={bs}", linewidth=1.5)
            ax.fill_between(epochs, mean_traj - std_traj, mean_traj + std_traj,
                            color=color, alpha=0.15)

        ax.axhline(N, color="green", linestyle=":", linewidth=1.5, label=f"N={N} bits")
        ax.set_xlabel("Epoch")
        ax.set_title(f"family_spread = {fs}")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Entropy (bits)")
    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="x-small")
    fig.suptitle("Entropy vs Epoch per Batch Size", y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "entropy_vs_epoch_per_batch_size.png", dpi=150, bbox_inches="tight")
    plt.show()

else:
    print("No data to plot.")

# %%
