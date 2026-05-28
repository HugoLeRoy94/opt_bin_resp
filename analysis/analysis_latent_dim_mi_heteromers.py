# %%
"""
Analysis of latent_dim_mi_heteromers sweep.
Plots final mutual information vs latent_dim per n_receptors (heteromerization level).

Directory layout (produced by latent_dim_mi_heteromers.py + SweepRunner):
  ../data/latent_dim_mi_heteromers/latent_dim_mi_heteromers_{timestamp}/
    latent_dim_{val}/n_receptors_{val}/n_ligands_{val}/average_family_distance_{val}/sample_{id}/
"""
import sys
sys.path.append('../')
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

from src.IO import SweepLoader, find_latest_sweep

# %% Configuration
base_dir   = Path("../data/latent_dim_mi_heteromers")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

MI_COL = "full_array_entropy"

# %% Load data
print(f"Loading data from {base_dir}...")

if not base_dir.exists():
    print(f"Directory {base_dir} not found.")
    df = None
else:
    sweep_root = find_latest_sweep(str(base_dir), prefix="latent_dim_mi_heteromers")[0]
    print(sweep_root)
    loader = SweepLoader(sweep_root)
    df     = loader.load_all_histories()
    print(f"Loaded {len(df)} logged steps across "
          f"{df.groupby(['latent_dim', 'n_receptors', 'sample_id']).ngroups} runs")

    config_path = sweep_root + "/sweep_config.json"
    with open(config_path) as f:
        config_dict = json.load(f)
    print("\n=== Configuration ===")
    for k, v in config_dict.items():
        print(f"  {k}: {v}")
    print("=====================\n")

# %%
if df is not None and not df.empty and MI_COL in df.columns:

    final = (
        df.sort_values("epoch")
          .groupby(["latent_dim", "n_receptors", "sample_id"])[MI_COL]
          .last()
          .reset_index()
    )

    latent_dims   = sorted(final["latent_dim"].unique())
    n_recept_vals = sorted(final["n_receptors"].unique())
    colors_nr     = plt.cm.viridis(np.linspace(0, 1, len(n_recept_vals)))

    # Plot 1: final MI vs latent_dim per n_receptors
    fig, ax = plt.subplots(figsize=(10, 6))

    for nr, color in zip(n_recept_vals, colors_nr):
        sub  = final[final["n_receptors"] == nr]
        lds  = np.array(sorted(sub["latent_dim"].unique()))
        grp  = sub.groupby("latent_dim")[MI_COL]
        mean = np.array([grp.get_group(d).mean() for d in lds])
        std  = np.array([grp.get_group(d).std(ddof=0) for d in lds])

        ax.plot(lds, mean, marker="o", color=color, linewidth=2, label=f"N_r={nr}")
        ax.fill_between(lds, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Latent dimension D")
    ax.set_ylabel("Mutual information (bits)")
    ax.set_title(
        "Final mutual information vs latent dimension\n"
        "solid = measured (mean ± std),  curves per n_receptors"
    )
    ax.legend(fontsize="small", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "final_mi_vs_latent_dim.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Plot 2: MI vs epoch — one subplot per latent_dim, curves per n_receptors
    fig, axes = plt.subplots(len(latent_dims), 1, figsize=(12, 4 * len(latent_dims)))
    if len(latent_dims) == 1:
        axes = [axes]

    for ax, ld in zip(axes, latent_dims):
        sub_ld = df[df["latent_dim"] == ld]
        for nr, color in zip(n_recept_vals, colors_nr):
            sub  = sub_ld[sub_ld["n_receptors"] == nr].sort_values("epoch")
            grp  = sub.groupby("epoch")[MI_COL]
            epochs    = np.array(sorted(sub["epoch"].unique()))
            mean_traj = grp.mean().values
            std_traj  = grp.std().fillna(0).values

            ax.plot(epochs, mean_traj, color=color, label=f"N_r={nr}", linewidth=2)
            ax.fill_between(epochs, mean_traj - std_traj, mean_traj + std_traj,
                            color=color, alpha=0.2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mutual information (bits)")
        ax.set_title(f"MI vs Epoch  (D={ld})")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "mi_vs_epoch_per_latent_dim.png", dpi=150, bbox_inches="tight")
    plt.show()

else:
    print("No data to plot.")

# %%
