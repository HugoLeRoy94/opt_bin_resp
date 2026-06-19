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
METRIC_LO = "full_array_entropy_mean"           # lower bound on MI
GENES     = [3, 5, 10, 20, 25, 35]

homo = load_runs(GOAL, receptor_type="homomer",   entropy="blocked")
hete = load_runs(GOAL, receptor_type="heteromer", entropy="blocked")

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
#df_g = hete[hete["n_genes"] == 10]
df_g = homo
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
ep = load_epochs(hete[hete['n_genes']==35])
ax = plot_metric(ep, y="full_array_entropy", x="epoch",
                 group="R", cmap="viridis", err=None)
ax.set_title("Convergence — Homomers")
ax.set_ylabel("H(A)  [bits]")
plt.show()

# %%
print(hete[(hete['n_genes']==3) & (hete['R']==11)]['epochs'])

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
#ax = plot_metric(homo, y=METRIC,    x="R", color="k", lw=2, label="Homomers")
ax = plot_metric(homo, y=METRIC_LO, x="R", color="k", lw=1.5, ls="--")
#plot_metric(hete, y=METRIC,    x="R", group="n_genes", cmap="viridis", ax=ax)
plot_metric(hete, y=METRIC_LO, x="R", group="n_genes", cmap="viridis",
            ax=ax, ls="--")
ax.plot(r_ref, r_ref, "k--", lw=1); ax.set_ylim(0, 50)
ax.set_ylabel("MI  [bits]"); ax.set_title("MI bounds (solid=upper, dashed=lower)")
plt.show()

# %%
# ── Plot 6 — conditional gain: heteromer MI minus homomer baseline at R=n_genes ─
base = homo.groupby("R")[METRIC].mean()        # homomer mean per R(=n_genes)
fig, ax = plt.subplots()
for g in GENES:
    if g not in base.index:
        continue
    stat = hete[hete["n_genes"] == g].groupby("R")[METRIC].agg(["mean", "std"])
    ax.plot(stat.index, stat["mean"] - base[g], lw=2, label=f"{g} genes")
    ax.fill_between(stat.index, stat["mean"] - base[g] - stat["std"],
                    stat["mean"] - base[g] + stat["std"], alpha=.2)
ax.axhline(0, color="k", lw=1, ls="--")
ax.set_xlabel("n_receptors  (R)")
ax.set_ylabel("MI(het, R) − MI(hom, n_genes)  [bits]")
ax.legend(title="n_genes", fontsize=8)
plt.show()
# %%


epochs = load_epochs