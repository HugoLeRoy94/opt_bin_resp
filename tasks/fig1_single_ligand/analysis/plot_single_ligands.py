# %%
"""Fig 1 plots — entropy/MI of receptor arrays, built on src.plotlib.

Edit METRIC to switch what is plotted; everything else is generic.
"""
import sys
from pathlib import Path
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")  # exec dir

import numpy as np
import matplotlib.pyplot as plt

from src.plotlib import load_runs, load_epochs, plot_metric, latest_sweep, load_model
from src.analysis_helper import plot_latent_umap

GOAL      = "fig1_single_ligand"
METRIC    = "full_array_entropy_blocked_mean"   # upper bound on MI
METRIC_LO = "full_array_entropy_mean"           # lower bound on MI
GENES     = [3]

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)

homo = load_runs(GOAL, receptor_type="homomer",   entropy="annealed")
hete = load_runs(GOAL, receptor_type="heteromer", entropy="annealed")

r_ref = np.arange(1, 10)

# %%
hete = latest_sweep(hete)
homo = latest_sweep(homo)
print(hete)

# %%
# ── MI vs R by family_spread, hom (solid) vs het (dashed) ────────────────────
ax = plot_metric(homo,y = METRIC,x = 'R',group='family_spread',cmap="viridis")
plot_metric(hete,y = METRIC,x = 'R',ax=ax,group='family_spread',cmap="viridis",ls='--')
ax.plot(r_ref, r_ref, "k--", lw=.8)
plt.savefig(FIGURES / "mi_vs_r_by_family_spread.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# ── Ratio of full_array_entropy (homomer / heteromer) vs R ────────────────────
RATIO_METRIC = METRIC

def _agg(df, y):
    return df.groupby(["family_spread", "R"])[y].agg(["mean", "std", "count"])

H = _agg(homo, RATIO_METRIC)
E = _agg(hete, RATIO_METRIC)

fs_vals = sorted(set(H.index.get_level_values("family_spread"))
                 & set(E.index.get_level_values("family_spread")))
norm = plt.Normalize(min(fs_vals), max(fs_vals))
cmap = plt.colormaps["viridis"]

fig, ax = plt.subplots()
for fs in fs_vals:
    h, e = H.loc[fs], E.loc[fs]
    R = h.index.intersection(e.index)
    if len(R) == 0:
        continue
    h, e = h.loc[R].sort_index(), e.loc[R].sort_index()
    ratio = h["mean"] / e["mean"]
    sem = ratio * np.sqrt((h["std"] / h["mean"]) ** 2 / h["count"]
                          + (e["std"] / e["mean"]) ** 2 / e["count"])
    c = cmap(norm(fs))
    ax.plot(ratio.index.values, ratio.values, marker="o", color=c)
    ax.fill_between(ratio.index.values,
                    (ratio - sem).values, (ratio + sem).values,
                    color=c, alpha=0.2)
ax.axhline(1.0, color="k", ls="--", lw=.8)
ax.set_xlabel("R")
ax.set_ylabel(f"{RATIO_METRIC}\nhomomer / heteromer")
fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=norm),
             ax=ax, label="family_spread")
plt.savefig(FIGURES / "mi_ratio_hom_het.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
ep = load_epochs(hete)
ax = plot_metric(ep, y="full_array_entropy_blocked", x="epoch",
                 group="R", cmap="viridis", err=None)
plt.savefig(FIGURES / "convergence_hete.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
ep = load_epochs(homo)
ax = plot_metric(ep, y="full_array_entropy_blocked", x="epoch",
                 group="R", cmap="viridis", err=None)
plt.savefig(FIGURES / "convergence_homo.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# ── UMAP of the latent environment for the largest homomer array ───────────────
max_R = int(homo["R"].max())
env, physics, ri = load_model(homo[homo["R"] == max_R])
fig, ax = plt.subplots(figsize=(7, 6))
plot_latent_umap(env, ri, ax=ax)
ax.set_title(f"Latent space UMAP — Homomers (R={max_R})")
plt.savefig(FIGURES / "umap_single_ligand.png", dpi=150, bbox_inches="tight")
plt.show()
# %%
