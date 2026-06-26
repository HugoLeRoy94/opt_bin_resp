# %%
"""Fig 1 plots — entropy/MI of receptor arrays, built on analysis.plotlib.

Edit METRIC to switch what is plotted; everything else is generic.
"""
import sys
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")  # exec dir

import numpy as np
import matplotlib.pyplot as plt

from analysis.plotlib import load_runs, load_epochs, plot_metric, latest_sweep

GOAL      = "fig1"
METRIC    = "full_array_entropy_blocked_mean"   # upper bound on MI
#METRIC = "full_array_entropy_blocked_corrected_mean"
METRIC_LO = "full_array_entropy_mean"           # lower bound on MI
GENES     = [3, 5, 7, 10, 15,20, 25]

homo = load_runs(GOAL, receptor_type="homomer",   entropy="annealed")
hete = load_runs(GOAL, receptor_type="heteromer", entropy="annealed")

r_ref = np.arange(1, 50)

# %%
for date in set(hete[hete['n_genes']==10]['sweep_folder']):
    print(hete[hete['sweep_folder']==date].__len__())
    print(hete[hete['sweep_folder']==date]['mu_ligands_per_source'])

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
# ── Plot 1 — env-condition spread for one arm (each sweep folder = one curve) ──
# %%
# ── Plot 2 — summary: H(A) vs R, homomers + one heteromer curve per gene count ─
ax = plot_metric(latest_sweep(homo), y=METRIC, x="R", color="k", lw=2, label="Homomers")
plot_metric(latest_sweep(hete[hete["n_genes"] == 3]), y=METRIC, x="R", cmap="viridis", ax=ax)
ax.plot(r_ref, r_ref, "k--", lw=1)
ax.set_ylim(0, 50)
ax.set_ylabel("H(A)  [bits]")
ax.set_title("Array entropy vs number of receptors")
ax.set_xlim(1,16)
ax.set_ylim(1,16)
plt.show()

# %%
# ── Plot 2 — summary: H(A) vs R, homomers + one heteromer curve per gene count ─
ax = plot_metric(latest_sweep(homo), y=METRIC, x="R", color="k", lw=2, label="Homomers")
plot_metric(latest_sweep(hete), y=METRIC, x="R", group="n_genes", cmap="viridis", ax=ax)
ax.plot(r_ref, r_ref, "k--", lw=1)
ax.set_ylim(0, 50)
ax.set_ylabel("H(A)  [bits]")
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
# Degree of heteromerization := round(n_receptors / n_genes).  This integer
# ratio counts how many receptors the array deploys per gene — the fold
# heteromerization.  Panel A: MI vs gene count, one curve per ratio.
# Panel B: MI vs R/G at fixed n_genes (diminishing returns).

homo_ls = latest_sweep(homo)
hete_ls = latest_sweep(hete).copy()
hete_ls["het_ratio"] = (hete_ls["R"] / hete_ls["n_genes"]).round().astype(int)

RATIO_LEVELS = sorted(r for r in hete_ls["het_ratio"].unique() if 1 <= r <= 5)

fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.5, 1]},
)
for ax in (axA, axB):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

cmap_het = plt.colormaps["viridis"]
norm_het = plt.Normalize(min(RATIO_LEVELS), max(RATIO_LEVELS))

# ---- Panel A: MI vs n_genes ------------------------------------------------

# Homomer reference — upper bound (solid) + lower bound (dashed)
h_up = homo_ls.groupby("n_genes")[METRIC].agg(["mean", "std"]).sort_index()
axA.plot(h_up.index, h_up["mean"], color="k", lw=2, label="Homomers")
axA.fill_between(h_up.index, h_up["mean"] - h_up["std"],
                 h_up["mean"] + h_up["std"], color="k", alpha=0.10)
h_lo = homo_ls.groupby("n_genes")[METRIC_LO].agg(["mean"]).sort_index()
axA.plot(h_lo.index, h_lo["mean"], color="k", lw=1.2, ls="--")

# Heteromer curves — one per ratio bin
for ratio in RATIO_LEVELS:
    sub = hete_ls[hete_ls["het_ratio"] == ratio]
    stat = sub.groupby("n_genes")[METRIC].agg(["mean", "std"]).sort_index()
    if len(stat) < 2:
        continue
    color = cmap_het(norm_het(ratio))
    axA.plot(stat.index, stat["mean"], color=color, lw=1.8,
             label=f"$R/G \\approx {ratio}$")
    band = stat["std"].fillna(0)
    axA.fill_between(stat.index, stat["mean"] - band, stat["mean"] + band,
                     color=color, alpha=0.15)
    lo = sub.groupby("n_genes")[METRIC_LO].agg(["mean"]).sort_index()
    axA.plot(lo.index, lo["mean"], color=color, lw=1.0, ls="--")

axA.set_xlabel("$n_{\\mathrm{genes}}$")
axA.set_ylabel("MI  [bits]")
axA.legend(frameon=False, fontsize=8, loc="lower right")
axA.set_xlim(2,20)
axA.set_ylim(2,40)
axA.plot(np.arange(0,21,1),np.arange(0,21,1))

# ---- Panel B: MI vs R/G, one curve per n_genes (diminishing returns) -------
genes_b = sorted(hete_ls["R"].unique())
cmap_b = plt.colormaps["viridis"]
norm_b = plt.Normalize(min(genes_b), max(genes_b))
for g in genes_b:
    sub_b = hete_ls[hete_ls["R"] == g]
    stat_b = sub_b.groupby("n_genes")[METRIC].agg(["mean", "std"]).sort_index()
    if stat_b.empty:
        continue
    rg = stat_b.index / g
    color = cmap_b(norm_b(g))
    mi_per_r = stat_b["mean"] / stat_b.index
    std_per_r = stat_b["std"].fillna(0) / stat_b.index
    axB.plot(rg, mi_per_r, "-o", color=color, lw=1.8, ms=3,
             label=f"$G={g}$")
    axB.fill_between(rg, mi_per_r - std_per_r, mi_per_r + std_per_r,
                     color=color, alpha=0.15)
axB.legend(frameon=False, fontsize=8)
axB.set_xlabel("$R \\,/\\, n_{\\mathrm{genes}}$")
axB.set_ylabel("MI / R  [bits per receptor]")

fig.tight_layout()
#fig.savefig(
#    "/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp/analysis/"
#    "heteromer_by_genes_real.svg",
#    bbox_inches="tight",
#)
plt.show()

# %%


epochs = load_epochs