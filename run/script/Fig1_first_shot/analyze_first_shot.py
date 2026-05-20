# %%
"""
Analysis of the first-shot simulation campaign (§6 of narrative_and_next_steps.md).

Directory layout (produced by first_shot_*.py + SweepRunner):
  ../../../data/first_shot/{arm}_{timestamp}/
    {average_family_distance}_{val}/{n_ligands}_{val}/sample_{id}/{warm_axis}_{val}/
        test_results.json   — metric dict, each key is a list of 10 eval-batch estimates
        stats.csv           — per-epoch training curve

Panels:
  A — H(A) [bits] vs R         (absolute capacity)
  B — H(A) / n_genes vs R      (efficiency per gene — the headline)
  C — MI decomposition          (I(A;identity), I(A;conc.), I(A;family) per arm)
  D — Estimator cross-check     (Rényi H2 vs blocked Shannon vs Miller-Madow)
  E — Training convergence      (entropy vs epoch)
"""
import sys
sys.path.append('../../../')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json

from src.IO import SweepLoader, find_latest_sweep

# %% Configuration

base_dir   = Path("../../../data/first_shot")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Metric used for the main H(A) signal.
# full_array_entropy_blocked = blocked Shannon on the test batch (mean over 10 passes).
METRIC = "full_array_entropy_blocked"

# (prefix, r_col, n_genes_fixed, label, color, linestyle)
ARMS = [
    ("homomers", "n_genes",     None, "Homomers",             "#222222", "-" ),
    ("casc_ng5", "n_receptors", 5,    "Cascading  $n_g=5$",   "#1f77b4", "-" ),
    ("casc_ng8", "n_receptors", 8,    "Cascading  $n_g=8$",   "#d62728", "-" ),
    ("rand_ng5", "n_receptors", 5,    "Unif. rand $n_g=5$",   "#1f77b4", "--"),
    ("rand_ng8", "n_receptors", 8,    "Unif. rand $n_g=8$",   "#d62728", "--"),
]

MI_COLS = [
    ("mutual_information_ligand",        "I(A; identity)",      "#2ca02c"),
    ("mutual_information_concentration", "I(A; concentration)", "#ff7f0e"),
    ("mutual_information_family",        "I(A; family)",        "#9467bd"),
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
        print(f"  {prefix:12s}  {len(df):3d} runs  "
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
    ax.fill_between(r_vals, p10, p90, color=color, alpha=0.15)

ax.set_xlabel("R  (number of receptors)", fontsize=11)
ax.set_ylabel("H(A)  [bits]", fontsize=11)
ax.set_title("Panel A — Array entropy vs receptor count\n"
             "solid = cascading / homomers  |  dashed = uniform random  |  band = 10–90 pct")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "panelA_entropy_vs_R.png", dpi=150, bbox_inches="tight")
plt.show()


# %% Panel B — H(A) / n_genes vs R  (efficiency per gene — the headline)

fig, ax = plt.subplots(figsize=(8, 5))

for prefix, r_col, n_genes_fixed, label, color, ls in ARMS:
    if prefix not in dfs or METRIC not in dfs[prefix].columns:
        continue
    df = dfs[prefix]

    # Homomers: n_genes = R per row → divide before aggregating.
    # Heteromers: n_genes fixed → divide after aggregating.
    if n_genes_fixed is None:
        df = df.copy()
        df["_hpg"] = df[METRIC] / df["n_genes"]
        metric_b = "_hpg"
    else:
        df = df.copy()
        df["_hpg"] = df[METRIC] / n_genes_fixed
        metric_b = "_hpg"

    grp = df.groupby("R")[metric_b]
    r_vals = sorted(df["R"].unique())
    med = np.array([grp.get_group(r).median() for r in r_vals])
    p10 = np.array([grp.get_group(r).quantile(0.10) for r in r_vals])
    p90 = np.array([grp.get_group(r).quantile(0.90) for r in r_vals])

    ax.plot(r_vals, med, color=color, ls=ls, lw=2, label=label)
    ax.fill_between(r_vals, p10, p90, color=color, alpha=0.15)

ax.set_xlabel("R  (number of receptors)", fontsize=11)
ax.set_ylabel("H(A) / n_genes  [bits / gene]", fontsize=11)
ax.set_title("Panel B — Efficiency per gene\n"
             "Heteromers cross above homomers when reusing genes pays off")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "panelB_efficiency_per_gene.png", dpi=150, bbox_inches="tight")
plt.show()


# %% Panel C — MI decomposition  (identity / concentration / family channels)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

for ax, ng_target in zip(axes, [5, 8]):
    for prefix, r_col, n_genes_fixed, label, color, ls in ARMS:
        if n_genes_fixed != ng_target:
            continue
        if prefix not in dfs:
            continue
        df = dfs[prefix]

        for mi_col, mi_label, mi_color in MI_COLS:
            if mi_col not in df.columns:
                continue
            grp = df.groupby("R")[mi_col]
            r_vals = sorted(df["R"].unique())
            med = np.array([grp.get_group(r).median() for r in r_vals])
            p10 = np.array([grp.get_group(r).quantile(0.10) for r in r_vals])
            p90 = np.array([grp.get_group(r).quantile(0.90) for r in r_vals])
            full_label = f"{mi_label}  ({label})"
            ax.plot(r_vals, med, color=mi_color, ls=ls, lw=2, label=full_label)
            ax.fill_between(r_vals, p10, p90, color=mi_color, alpha=0.12)

    ax.set_xlabel("R  (number of receptors)", fontsize=11)
    ax.set_ylabel("bits", fontsize=11)
    ax.set_title(f"MI decomposition  $n_g = {ng_target}$\n"
                 "solid = cascading  |  dashed = uniform random")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "panelC_mi_decomposition.png", dpi=150, bbox_inches="tight")
plt.show()


# %% Panel D — Estimator cross-check  (Rényi vs blocked Shannon vs Miller-Madow)
# §6.1 point 3: verify Rényi ≈ Shannon in the calibrated zone (R ≤ 15).

n_arms = len(dfs)
fig, axes = plt.subplots(1, n_arms, figsize=(4 * n_arms, 4), sharey=False)
if n_arms == 1:
    axes = [axes]

for ax, (prefix, r_col, n_genes_fixed, label, color, ls) in zip(axes, [arm for arm in ARMS if arm[0] in dfs]):
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
        ax.fill_between(r_vals, p10, p90, color=est_color, alpha=0.15)
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

for ax, (prefix, r_col, n_genes_fixed, label, color, ls) in zip(axes, [arm for arm in ARMS if arm[0] in dfs]):
    try:
        sweep_dir = find_latest_sweep(str(base_dir), prefix=prefix)[0]
        hist = SweepLoader(sweep_dir).load_all_histories()
    except Exception as e:
        ax.set_title(f"{label}\n(no history: {e})")
        continue

    warm_col = r_col   # n_genes for homomers, n_receptors for heteromers
    r_vals_sorted = sorted(hist[warm_col].unique()) if warm_col in hist.columns else [None]
    colors_r = plt.cm.viridis(np.linspace(0, 1, len(r_vals_sorted)))

    loss_col = "loss" if "loss" in hist.columns else (
        "full_array_entropy" if "full_array_entropy" in hist.columns else None
    )
    if loss_col is None:
        ax.set_title(f"{label}\n(no loss column)")
        continue

    for r_val, c_r in zip(r_vals_sorted, colors_r):
        sub = hist[hist[warm_col] == r_val].sort_values("epoch") if r_val is not None else hist.sort_values("epoch")
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
plt.savefig(output_dir / "panelE_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
