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

hete = latest_sweep(hete)
#print(homo.columns)
print(set(hete['sweep_folder']))
print(set(homo['sweep_folder']))

#print(set(latest_sweep(hete[hete['n_genes']==10]['sweep_folder'])))
print(latest_sweep(hete[hete['n_genes']==10])['sweep_folder'])

# %%
# ── Plot 1 — env-condition spread for one arm (each sweep folder = one curve) ──
df_g = hete[hete["n_genes"] == 15]
#df_g = homo
ax = plot_metric(df_g, y=METRIC, x="R", group="sweep_folder")
ax.plot(r_ref, r_ref, "k--", lw=.8)
ax.set_title("Homomers — per sweep folder")
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
ep = load_epochs(hete[hete['n_genes']==25])
#ep = load_epochs(homo)
ax = plot_metric(ep, y="full_array_entropy", x="epoch",
                 group="R", cmap="viridis", err=None)
ax.set_title("Convergence — Homomers")
ax.set_ylabel("H(A)  [bits]")
#plt.ylim(0,50)
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


# ---- Panel B: MI vs R/G at fixed n_genes (diminishing returns) -------------
G_FIXED = 10
sub_b = hete_ls[hete_ls["n_genes"] == G_FIXED]
stat_b = sub_b.groupby("R")[METRIC].agg(["mean", "std"]).sort_index()
rg = stat_b.index / G_FIXED

axB.plot(rg, stat_b["mean"], "-o", color="#555", lw=1.8, ms=3)
band_b = stat_b["std"].fillna(0)
axB.fill_between(rg, stat_b["mean"] - band_b, stat_b["mean"] + band_b,
                 color="#555", alpha=0.15)
if G_FIXED in h_up.index:
    axB.axhline(h_up.loc[G_FIXED, "mean"], color="k", lw=1, ls="--",
                label=f"homomer ($G={G_FIXED}$)")
    axB.legend(frameon=False, fontsize=8)
axB.set_xlabel("$R \\,/\\, n_{\\mathrm{genes}}$")
axB.set_ylabel("MI  [bits]")

fig.tight_layout()
fig.savefig(
    "/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp/analysis/"
    "heteromer_by_genes_real.svg",
    bbox_inches="tight",
)
plt.show()

# %%


epochs = load_epochs