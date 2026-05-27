# %%
"""
Analysis of the first-shot simulation campaign (§6 of narrative_and_next_steps.md).

Directory layout (produced by first_shot_*.py + SweepRunner):
  ../../../data/first_shot/{arm}_{timestamp}/
    {average_family_distance}_{val}/{family_spread}_{val}/{n_ligands}_{val}/sample_{id}/{warm_axis}_{val}/
        test_results.json   — metric dict, each key is a list of 10 eval-batch estimates
        stats.csv           — per-epoch training curve

Panels:
  A  — H(A) [bits] vs R        (absolute capacity, all arms)
  A2 — Heteromer/Homomer ratio  (I_hetero/I_homo per estimator, ng=5 and ng=8)
  C  — MI decomposition          (one subplot per channel: identity / concentration / family)
  D — Estimator cross-check     (Rényi H2 vs blocked Shannon vs Miller-Madow)
  E — Training convergence      (entropy vs epoch)
  G — I(A; family) per distance  (1×3: all arms at d=0.5, 1.0, 1.5)
  G2— I(A; family) homomers     (3×3: distance × spread, homomers only)
  F — Latent space UMAP         (4 contrasting environments: 2 family distances × 2 spreads)
"""
import sys
from pathlib import Path
import torch

exec_dir = "/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp"
sys.path.append(exec_dir)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json

from src.IO import SweepLoader, SingleRunLoader, find_latest_sweep
from src.environment import (LigandEnvironment, SymmetricLigandEnvironment,
                              LogNormalConcentration, NormalConcentration)

# %% Configuration

base_dir   = Path(exec_dir + "/data/first_shot")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Metric used for the main H(A) signal.
METRIC = "full_array_entropy_blocked"

# (prefix, r_col, n_genes_fixed, label, color, linestyle)
ARMS = [
    ("homomers", "n_genes",     None, "Homomers",             "#222222", "-" ),
    ("casc_ng5", "n_receptors", 5,    "Cascading  $n_g=5$",   "#1f77b4", "-" ),
    ("rand_ng5", "n_receptors", 5,    "Unif. rand $n_g=5$",   "#1f77b4", "--"),
    ("casc_ng3", "n_receptors", 3,    "Cascading  $n_g=3$",   "#d62728", "-" ),    
    ("rand_ng3", "n_receptors", 3,    "Unif. rand $n_g=3$",   "#d62728", "--"),
]

# One tuple per MI channel — each gets its own subplot in Panel C.
MI_CHANNELS = [
    ("mutual_information_ligand",        "I(A ; identity)"),
    ("mutual_information_concentration", "I(A ; concentration)"),
    ("mutual_information_family",        "I(A ; family)"),
    ("mutual_information_block",         "I(A ; block)"),
]

EST_COLS = [
    ("full_array_entropy",         "Rényi H2",        "#1f77b4", "-" ),
    ("full_array_entropy_blocked", "Blocked Shannon", "#ff7f0e", "-" ),
    ("codeword_entropy_mm",        "Miller-Madow",    "#2ca02c", "--"),
]


# %% Load test results

dfs = {}
for prefix, r_col, n_genes_fixed, label, color, ls in ARMS:
    if not base_dir.exists():
        print(f"  [skip] base_dir not found: {base_dir}")
        break
    try:
        sweep_dir = find_latest_sweep(str(base_dir), prefix=prefix)[0]
        df = SweepLoader(sweep_dir).load_all_test_results()
        df["R"] = df[r_col]
        if n_genes_fixed is not None:
            df["n_genes"] = n_genes_fixed
        dfs[prefix] = df
        print(f"  {prefix:12s} {sweep_dir} {len(df):3d} runs  "
              f"R ∈ {sorted(df['R'].unique())}  "
              f"n_lig ∈ {sorted(df['n_ligands'].unique())}")
    except FileNotFoundError as e:
        print(f"  [skip] {prefix}: {e}")

print(f"\nLoaded {len(dfs)}/{len(ARMS)} arms.")


# %% Panel A — H(A) vs R  (absolute capacity)

fig, ax = plt.subplots(figsize=(8, 5))

for prefix, r_col, n_genes_fixed, label, color, ls in ARMS:
    if prefix not in dfs or METRIC not in dfs[prefix].columns:
        continue
    df = dfs[prefix]
    grp = df.groupby("R")[METRIC]
    r_vals = sorted(df["R"].unique())
    med = np.array([grp.get_group(r).median() for r in r_vals])
    p10 = np.array([grp.get_group(r).quantile(0.10) for r in r_vals])
    p90 = np.array([grp.get_group(r).quantile(0.90) for r in r_vals])

    ax.plot(r_vals, med, color=color, ls=ls, lw=2, label=label)
    ax.fill_between(r_vals, p10, p90, color=color, alpha=0.5)

ax.plot(np.arange(1, 16, 1), np.arange(1, 16, 1),
        color='black', linestyle='--', label='perfect array', linewidth=1)

ax.set_xlabel("R  (number of receptors)", fontsize=11)
ax.set_ylabel("H(A)  [bits]", fontsize=11)
ax.set_title("Panel A — Array entropy vs receptor count\n"
             "solid = cascading / homomers  |  dashed = uniform random  |  band = 10–90 pct")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
#plt.savefig(output_dir / "panelA_entropy_vs_R.png", dpi=150, bbox_inches="tight")
plt.show()


# %% Panel A2 — Heteromer / Homomer MI ratio at matched R
# For each entropy estimator and sampling strategy, plots I_hetero(R) / I_homo(R).
# Homomer reference = median over all environments at each R.
# Values < 1 mean heteromers are sub-optimal at matched receptor count.
# Two subplots: ng=5 (left) and ng=8 (right).
# Color = entropy estimator  |  linestyle = sampling strategy (solid=cascading, dashed=random)

RATIO_METRICS = [
    ("full_array_entropy_blocked", "Blocked Shannon", "#ff7f0e"),
    ("full_array_entropy",         "Rényi H2",        "#1f77b4"),
    ("codeword_entropy_mm",        "Miller-Madow",    "#2ca02c"),
]

homo_df = dfs.get("homomers")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, ng_target in zip(axes, [3,5]):
    for metric_col, metric_label, metric_color in RATIO_METRICS:
        if homo_df is None or metric_col not in homo_df.columns:
            continue
        homo_grp = homo_df.groupby("R")[metric_col]
        homo_med = {int(r): homo_grp.get_group(r).median()
                    for r in homo_df["R"].unique()}

        for prefix, r_col, n_genes_fixed, arm_label, arm_color, ls in ARMS:
            if n_genes_fixed != ng_target:
                continue
            if prefix not in dfs:
                continue
            df = dfs[prefix]
            if metric_col not in df.columns:
                continue

            grp = df.groupby("R")[metric_col]
            r_vals = sorted(r for r in df["R"].unique()
                            if int(r) in homo_med and homo_med[int(r)] > 0)
            if not r_vals:
                continue

            med = np.array([grp.get_group(r).median() / homo_med[int(r)] for r in r_vals])
            p10 = np.array([grp.get_group(r).quantile(0.10) / homo_med[int(r)] for r in r_vals])
            p90 = np.array([grp.get_group(r).quantile(0.90) / homo_med[int(r)] for r in r_vals])

            strategy = "casc" if "casc" in prefix else "rand"
            ax.plot(r_vals, med, color=metric_color, ls=ls, lw=2,
                    label=f"{metric_label}  ({strategy})")
            ax.fill_between(r_vals, p10, p90, color=metric_color, alpha=0.12)

    ax.axhline(1.0, color="black", lw=1, ls="--", label="Homomer reference")
    ax.set_xlabel("R  (number of receptors)", fontsize=11)
    ax.set_ylabel("I(heteromer) / I(homomer)", fontsize=11)
    ax.set_title(f"$n_g = {ng_target}$", fontsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

fig.suptitle("Panel A2 — Heteromer / Homomer MI ratio at matched R\n"
             "solid = cascading  |  dashed = uniform random  |  band = 10–90 pct of heteromers / homomer median",
             fontsize=11)
plt.tight_layout()
#plt.savefig(output_dir / "panelA2_ratio_hetero_homo.png", dpi=150, bbox_inches="tight")
plt.show()


# %% Panel C — MI decomposition  (one subplot per channel, all arms overlaid)
# Each subplot shows how a single information channel (identity / concentration / family)
# scales with R for all arms — color = arm, linestyle = sampling strategy.

n_ch = len(MI_CHANNELS)
fig, axes = plt.subplots(1, n_ch, figsize=(5 * n_ch, 5), sharey=False)

for ax, (mi_col, mi_title) in zip(axes, MI_CHANNELS):
    for prefix, r_col, n_genes_fixed, label, color, ls in ARMS:
        if prefix not in dfs:
            continue
        df = dfs[prefix]
        if mi_col not in df.columns:
            continue
        grp = df.groupby("R")[mi_col]
        r_vals = sorted(df["R"].unique())
        med = np.array([grp.get_group(r).median() for r in r_vals])
        p10 = np.array([grp.get_group(r).quantile(0.10) for r in r_vals])
        p90 = np.array([grp.get_group(r).quantile(0.90) for r in r_vals])
        ax.plot(r_vals, med, color=color, ls=ls, lw=2, label=label)
        ax.fill_between(r_vals, p10, p90, color=color, alpha=0.5)

    ax.set_xlabel("R  (number of receptors)", fontsize=11)
    ax.set_ylabel("bits", fontsize=11)
    ax.set_title(mi_title, fontsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

fig.suptitle("Panel C — MI decomposition per channel\n"
             "solid = cascading / homomers  |  dashed = uniform random  |  band = 10–90 pct",
             fontsize=11)
plt.tight_layout()
#plt.savefig(output_dir / "panelC_mi_decomposition.png", dpi=150, bbox_inches="tight")
plt.show()



# %% Panel D — Estimator cross-check  (Rényi vs blocked Shannon vs Miller-Madow)
# §6.1 point 3: verify Rényi ≈ Shannon in the calibrated zone (R ≤ 15).

n_arms = len(dfs)
fig, axes = plt.subplots(1, n_arms, figsize=(4 * n_arms, 4), sharey=False)
if n_arms == 1:
    axes = [axes]

for ax, (prefix, r_col, n_genes_fixed, label, color, ls) in zip(
        axes, [arm for arm in ARMS if arm[0] in dfs]):
    df = dfs[prefix]
    for col, est_label, est_color, est_ls in EST_COLS:
        if col not in df.columns:
            continue
        grp = df.groupby("R")[col]
        r_vals = sorted(df["R"].unique())
        med = np.array([grp.get_group(r).median() for r in r_vals])
        p10 = np.array([grp.get_group(r).quantile(0.10) for r in r_vals])
        p90 = np.array([grp.get_group(r).quantile(0.90) for r in r_vals])
        ax.plot(r_vals, med, color=est_color, ls=est_ls, lw=2, label=est_label)
        ax.fill_between(r_vals, p10, p90, color=est_color, alpha=0.5)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("R")
    ax.set_ylabel("H  [bits]")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.suptitle("Panel D — Estimator cross-check  (should agree for R ≤ 15)", fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / "panelD_estimator_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


# %% Panel E — Training convergence  (entropy vs epoch per arm)

fig, axes = plt.subplots(1, n_arms, figsize=(4 * n_arms, 4), sharey=False)
if n_arms == 1:
    axes = [axes]

for ax, (prefix, r_col, n_genes_fixed, label, color, ls) in zip(
        axes, [arm for arm in ARMS if arm[0] in dfs]):
    try:
        sweep_dir = find_latest_sweep(str(base_dir), prefix=prefix)[0]
        hist = SweepLoader(sweep_dir).load_all_histories()
    except Exception as e:
        ax.set_title(f"{label}\n(no history: {e})")
        continue

    warm_col = r_col
    r_vals_sorted = sorted(hist[warm_col].unique()) if warm_col in hist.columns else [None]
    colors_r = plt.cm.viridis(np.linspace(0, 1, len(r_vals_sorted)))

    loss_col = "loss" if "loss" in hist.columns else (
        "full_array_entropy" if "full_array_entropy" in hist.columns else None
    )
    if loss_col is None:
        ax.set_title(f"{label}\n(no loss column)")
        continue

    for r_val, c_r in zip(r_vals_sorted, colors_r):
        sub = (hist[hist[warm_col] == r_val].sort_values("epoch")
               if r_val is not None else hist.sort_values("epoch"))
        grp = sub.groupby("epoch")[loss_col]
        epochs = sorted(sub["epoch"].unique())
        mean_v = grp.mean().values
        std_v  = grp.std().fillna(0).values
        ax.plot(epochs, mean_v, color=c_r, lw=1.5, label=f"R={r_val}")
        ax.fill_between(epochs, mean_v - std_v, mean_v + std_v, color=c_r, alpha=0.15)

    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(loss_col)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

plt.suptitle("Panel E — Training convergence per arm and R", fontsize=12)
plt.tight_layout()
#plt.savefig(output_dir / "panelE_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()


# %% Panel F — Latent space UMAP for contrasting environments
# 4 conditions: 2 average_family_distance values × 2 family_spread values.
# Loaded from the homomers sweep (n_genes=8, n_ligands=100, sample_0).

from src.analysis_helper import plot_latent_umap

# (average_family_distance, family_spread, short label)
UMAP_CONDITIONS = [
    (0.5, 0.10, "d=0.5  σ=0.10  (close, tight)"),
    (0.5, 0.20, "d=0.5  σ=0.20  (close, spread)"),
    (1.5, 0.10, "d=1.5  σ=0.10  (far, tight)"),
    (1.5, 0.20, "d=1.5  σ=0.20  (far, spread)"),
]

try:
    homomers_sweep = find_latest_sweep(str(base_dir), prefix="homomers")[0]
except FileNotFoundError:
    homomers_sweep = None
    print("[skip] homomers sweep not found — skipping Panel F")

if homomers_sweep:
    for avg_d, spread, title in UMAP_CONDITIONS:
        run_dir = (Path(homomers_sweep)
                   / f"average_family_distance_{avg_d}"
                   / f"family_spread_{spread}"
                   / "n_ligands_100"
                   / "sample_0"
                   / "n_genes_8")

        if not run_dir.exists():
            print(f"  [skip] {title}: directory not found at {run_dir}")
            continue

        try:
            loader     = SingleRunLoader(str(run_dir))
            config     = loader.load_config()
            checkpoint = loader.load_checkpoint(filename="best_model.pt", map_location="cpu")

            conc_cls   = (NormalConcentration if config.conc_model_type == "normal"
                          else LogNormalConcentration)
            conc_model = conc_cls(
                n_ligands=config.n_ligands,
                init_mean=config.conc_mean,
                init_scale=config.conc_std,
            )
            env_class = (SymmetricLigandEnvironment
                         if config.environment_geometry == "symmetric"
                         else LigandEnvironment)
            env = env_class(
                config.n_genes, config.n_families,
                conc_model=conc_model,
                n_ligands=config.n_ligands,
                p_presence=config.p_presence,
                observation_noise_sigma=config.observation_noise_sigma,
                latent_dim=config.latent_dim,
                family_spread=config.family_spread,
                avg_family_distance=config.average_family_distance,
                n_presence_blocks=config.n_presence_blocks,
                rho_block=config.rho_block,
                block_shared_conc_mean=config.block_shared_conc_mean,
                affinity_kernel=config.affinity_kernel,
                kernel_params=config.kernel_params,
                distribution_type=config.distribution_type,
            )
            env.load_state_dict(checkpoint["env_state"])
            env.eval()
            receptor_indices = checkpoint["receptor_indices"]

            fig, ax = plot_latent_umap(env, receptor_indices,
                                       n_samples_per_family=300)
            ax.set_title(f"UMAP — {title}", fontsize=10, fontweight='bold')
            safe = title.replace(" ", "_").replace("=", "").replace(".", "p")
            fig.savefig(output_dir / f"panelF_umap_{safe}.png",
                        dpi=150, bbox_inches="tight")
            plt.show()
            print(f"  Saved UMAP: {title}")
        except Exception as e:
            print(f"  [error] {title}: {e}")

# %%
