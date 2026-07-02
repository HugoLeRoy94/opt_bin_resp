# %%
"""Fig 1 plots — entropy/MI of receptor arrays, built on analysis.plotlib.

Edit METRIC to switch what is plotted; everything else is generic.
"""
import sys
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")  # exec dir

import numpy as np
import matplotlib.pyplot as plt

from analysis.plotlib import load_runs, load_epochs, plot_metric, latest_sweep, load_model
from src.analysis_helper import plot_latent_umap

GOAL      = "fig1"
METRIC    = "full_array_entropy_blocked_mean"   # upper bound on MI
#METRIC = "full_array_entropy_blocked_corrected_mean"
METRIC_LO = "full_array_entropy_mean"           # lower bound on MI
GENES     = [2,3, 5, 7, 10, 15,20, 25]

homo = load_runs(GOAL, receptor_type="homomer",   entropy="annealed")
hete = load_runs(GOAL, receptor_type="heteromer", entropy="annealed")

r_ref = np.arange(1, 50)

# %%
print(set(hete[hete['n_genes']==10]['sweep_folder']))
hete[hete['n_genes']==10] = hete[hete['n_genes']==10][hete[hete['n_genes']==10]['sweep_folder'] == "ng10_20260623_102715"]
print(set(hete[hete['n_genes']==10]['sweep_folder']))
#for date in set(hete[hete['n_genes']==10]['sweep_folder']):
#    print(date)
#    print(hete[hete['sweep_folder']==date].__len__())
#    print(hete[hete['sweep_folder']==date]['mu_ligands_per_source'])

# %% 

hete = latest_sweep(hete)
homo = latest_sweep(homo)
#print(homo.columns)
print(set(hete['sweep_folder']))
print(set(homo['sweep_folder']))

for mus in hete.groupby('n_genes')['mu_ligands_per_source']:
    print("n genes : "+str(mus[0]))
    print(mus[1].__len__())
    print(np.mean(mus[1]))


# %%
# ── UMAP landscape — optimized homomers, 5 families ───────────────────────────
#   Shows the chemical latent space (family regions + ligands) and where the
#   optimized receptors sit. R picks the receptor count (homomer => R == n_genes).
N_FAMILIES = 5
R_SHOW     = 12

homo5 = latest_sweep(homo[homo["n_families"] == N_FAMILIES])
# Most-separated families: largest inter-family distance / spread ratio.
homo5 = homo5.assign(_sep=homo5["average_family_distance"] / homo5["family_spread"])
best_combo = homo5.loc[homo5["_sep"].idxmax(), ["family_spread", "average_family_distance"]]
homo5 = homo5[(homo5["family_spread"] == best_combo["family_spread"]) &
              (homo5["average_family_distance"] == best_combo["average_family_distance"])]
run   = homo5[homo5["R"] == R_SHOW]
env, physics, ri = load_model(run)
print(f"spread={best_combo['family_spread']:.3f}  "
      f"avg_dist={best_combo['average_family_distance']:.3f}")

fig, ax = plt.subplots(figsize=(7, 6))
plot_latent_umap(env, ri, ax=ax)
ax.set_title(f"Latent space UMAP — Homomers (R={R_SHOW}, {N_FAMILIES} families)")
ax.legend(fontsize=8, loc="best")
plt.tight_layout()
#plt.savefig('umap_pres.svg')
plt.show()

# %%
# ── Plot 1 — env-condition spread for one arm (each sweep folder = one curve) ──
# %%
# ── Plot 2 — summary: H(A) vs R, homomers + one heteromer curve per gene count ─
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'monospace'
fig,ax = plt.subplots(figsize=(6,4))
plot_metric(latest_sweep(homo), y=METRIC, x="R", color="k", lw=2, label="Homomers",ax=ax)
plot_metric(latest_sweep(hete[hete["n_genes"] == 3]), y=METRIC, x="R", cmap="viridis", ax=ax,label='3 genes heteromers')
#df = latest_sweep(hete[hete["n_genes"] == 3])
#ax.plot(df['R'],df[METRIC])
ax.plot(r_ref, r_ref, "k--", lw=1)
ax.set_ylim(0, 50)

ax.set_xlim(1,16)
ax.set_ylim(1,16)
ax.set_ylabel("mutual information [bits]")
ax.set_xlabel('# of receptors')
ax.tick_params(axis = 'both',direction='in')
ax.set_xticks([1,3,5,10,15])
ax.set_yticks([1,3,5,10,15])
ax.hlines(y=3, xmin=0, xmax=3, color='black', linestyle='--', linewidth=1)
ax.vlines(x=3, ymin=0, ymax=3, color='black', linestyle='--', linewidth=1)
plt.legend()
plt.show()

# %%
# ── Plot 2 — summary: H(A) vs R, homomers + one heteromer curve per gene count ─
ax = plot_metric(latest_sweep(homo), y=METRIC, x="R", color="k", lw=2, label="Homomers")
plot_metric(latest_sweep(hete), y=METRIC, x="R", group="n_genes", cmap="viridis", ax=ax)
ax.plot(r_ref, r_ref, "k--", lw=1)
ax.set_ylim(0, 30)
ax.set_xlim(0, 30)

ax.set_title("Array entropy vs number of receptors")
#ax.set_xlim(0,10)
#ax.set_ylim(0,10)
plt.show()

# %%
# ── Plot 3 — convergence: metric vs epoch for one arm, one curve per R ─────────
#ep = load_epochs(hete[hete["n_genes"] == 35])
ep = load_epochs(hete[hete['n_genes']==15])
#ep = load_epochs(hom)
ax = plot_metric(ep, y="full_array_entropy_blocked_corrected", x="epoch",
                 group="R", cmap="viridis", err=None)
ax.set_title("Convergence — Homomers")
ax.set_ylabel("H(A)  [bits]")
plt.ylim(0,50)
plt.show()

# %%
# ── Plot 4 — MI vs n_genes: homomer line + heteromer error bars by n_receptors ─
fig, ax = plt.subplots()
plot_metric(homo, y=METRIC, x="R", color="k", lw=2, label="Homomers", ax=ax)
for g in GENES:
    sub = hete[hete["n_genes"] == g]
    stat = sub.groupby("R")[METRIC].agg(["mean", "std"])
    cmap = plt.colormaps["plasma"]
    norm = plt.Normalize(hete["R"].min(), hete["R"].max())
    for R, row in stat.iterrows():
        ax.errorbar(g, row["mean"], yerr=row["std"], fmt="o", ms=4,
                    color=cmap(norm(R)), capsize=2, alpha=.8)
fig.colorbar(plt.cm.ScalarMappable(cmap="plasma", norm=norm), ax=ax,
             label="n_receptors")
ax.set_xlabel("n_genes"); ax.set_ylabel("MI  [bits]")
ax.set_xticks([1] + GENES); ax.legend(fontsize=8)
plt.show()

# %%
# ── Plot 5 — MI bounds: upper (solid) vs lower (dashed) per arm ────────────────
ax = plot_metric(homo, y=METRIC,    x="R", color="k", lw=2, label="Homomers")
plot_metric(homo, y=METRIC_LO, x="R", color="k", lw=1.5, ls="--",ax=ax)
plot_metric(hete, y=METRIC,    x="R", group="n_genes", cmap="viridis", ax=ax)
plot_metric(hete, y="full_array_entropy_blocked_corrected_mean",    x="R", group="n_genes", cmap="viridis", ax=ax,ls=':')
#plot_metric(hete, y=METRIC_LO, x="R", group="n_genes", cmap="viridis",ax=ax, ls="--")
ax.plot(r_ref, r_ref, "k--", lw=1); ax.set_ylim(0, 50)
ax.set_ylabel("MI  [bits]"); ax.set_title("MI bounds (solid=upper, dashed=lower)")
#plt.savefig('MI_R.png')

# %%
# ── Plot 7 — MI vs n_genes, one curve per heteromerization ratio R/G ─────────

homo_ls = latest_sweep(homo)
hete_ls = latest_sweep(hete).copy()
hete_ls["het_ratio"] = (hete_ls["R"] / hete_ls["n_genes"]).round().astype(int)

RATIO_LEVELS = sorted(r for r in hete_ls["het_ratio"].unique() if 1 <= r <= 5)

fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

cmap_het = plt.colormaps["viridis"]
norm_het = plt.Normalize(min(RATIO_LEVELS), max(RATIO_LEVELS))

h_up = homo_ls.groupby("n_genes")[METRIC].agg(["mean", "std"]).sort_index()
#ax.plot(h_up.index, h_up["mean"], color="k", lw=2, label="Homomers")
ax.fill_between(h_up.index, h_up["mean"] - h_up["std"],
                h_up["mean"] + h_up["std"], color="k", alpha=0.10)
#h_lo = homo_ls.groupby("n_genes")[METRIC_LO].agg(["mean"]).sort_index()
#ax.plot(h_lo.index, h_lo["mean"], color="k", lw=1.2, ls="--")

for ratio in RATIO_LEVELS:
    sub = hete_ls[hete_ls["het_ratio"] == ratio]
    stat = sub.groupby("n_genes")[METRIC].agg(["mean", "std"]).sort_index()
    if len(stat) < 2:
        continue
    color = cmap_het(norm_het(ratio))
    ax.plot(stat.index, stat["mean"], color=color, lw=1.8,
        label=fr"$R/n_\text{{genes}} \approx {ratio}$")
    band = stat["std"].fillna(0)
    ax.fill_between(stat.index, stat["mean"] - band, stat["mean"] + band,
                    color=color, alpha=0.5)
    lo = sub.groupby("n_genes")[METRIC_LO].agg(["mean"]).sort_index()
    ax.plot(lo.index, lo["mean"], color=color, lw=1.0, ls="--")

ax.set_xlabel("$n_{\\mathrm{genes}}$")
ax.set_ylabel("MI  [bits]")
ax.plot(np.arange(0, 16, 1), np.arange(0, 16, 1),color='k',ls='--',label='perfect \nhomomers')
ax.legend(frameon=False, fontsize=8, loc="upper left",bbox_to_anchor=(1.05, 1.))
ax.set_xlim(2, 15)
ax.set_ylim(2, 30)
ax.tick_params(axis = 'both',direction='in')
ax.set_xticks([2,5,10,15])
ax.set_yticks([2,5,10,15,20,25])
fig.tight_layout()
plt.subplots_adjust(right=0.8)
#plt.savefig('impact_of_heteromerization.svg',bbox_inches='tight')

# %%


epochs = load_epochs