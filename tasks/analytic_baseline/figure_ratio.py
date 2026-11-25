# %%
"""analytic_baseline, fig1-style: MI against the number of genes G, one curve per R / G.

R / G = 1 is the homomer array, read from the homomer_R sweep (train_sweep_R.py).
Higher ratios build more receptors from the same G genes (train_sweep_ratio.py).
One panel per nu, which conc_std sets.

Filled circles: exact entropy (test_exact.py --prefix ratio). Hollow
squares: counting, Miller-Madow (test_counting.py --prefix ratio). Dashed
grey: homomer closed form at R = G. Band: 1 sd across replicates, which average
the environment, the stoichiometry and the repertoire.

Second figure: the two channels, I(A; l) = H(A) - H(A | l) with l the ligand
position, and I(A; s) = H(A) - H(A | s) with s = ln c the log-concentration.
H(A) is entropy_exact, the conditionals come from test_channels.py.
"""
import matplotlib.pyplot as plt
import numpy as np

from analysis.load import across_replicates, runs
from tasks._lib import theory as T

GOAL = "analytic_baseline"
EXACT = "entropy_exact"
COUNT = "codeword_entropy_mm"
CHANNELS = {"I_ligand": ("entropy_given_ligand", "ligand identity, I(A; l)"),
            "I_conc": ("entropy_given_conc", "concentration, I(A; s)")}
RATIOS = (1, 2, 3, 4, 5)
COLOURS = dict(zip(RATIOS, plt.colormaps["Blues"](np.linspace(.4, 1., len(RATIOS)))))

frame = runs(GOAL, sweeps=["ratio", "homomer_R"], latest_only=True)
frame = frame.assign(
    ratio=(frame.n_receptors_resolved / frame.n_genes).round().astype(int),
    nu=[round(T.nu_from_columns(r.latent_dim, r.family_spread, r.kernel_lambda,
                                r.max_energy, r.conc_std), 4)
        for r in frame.itertuples()])
# homomer_R reaches G = 16; keep only the G range the heteromer sweeps cover.
G_max = frame.loc[frame.ratio > 1, "n_genes"].max()
frame = frame[frame.n_genes <= G_max]
for mi, (given, _) in CHANNELS.items():
    if given in frame and EXACT in frame:
        frame[mi] = frame[EXACT] - frame[given]
points = across_replicates(frame, keys=["nu", "ratio", "n_genes"])
nus = sorted(points.nu.unique())
D = int(frame.latent_dim.iloc[0])

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True, sharey=True)
for ax, nu in zip(axes.flat, nus):
    for ratio in sorted(points.ratio.unique()):
        curve = points[(points.nu == nu) & (points.ratio == ratio)].sort_values("n_genes")
        if curve.empty:
            continue
        G = curve.n_genes.to_numpy()
        colour = COLOURS[ratio]
        label = f"R/G = {ratio}" + ("  (homomers)" if ratio == 1 else "")
        if EXACT in curve:
            ax.plot(G, curve[EXACT], "o-", color=colour, lw=2, ms=6, label=label)
            ax.fill_between(G, curve[f"{EXACT} lo"], curve[f"{EXACT} hi"],
                            color=colour, alpha=.15, linewidth=0)
        if COUNT in curve:
            # Larger than the exact markers, which they would otherwise hide under.
            ax.plot(G, curve[COUNT], "s", color=colour, ms=11, mew=1.2,
                    markerfacecolor="none", zorder=3,
                    label=None if EXACT in curve else label)
    genes = np.sort(points.n_genes.unique())
    ax.plot(genes, [T.mi_predicted(nu, int(g)) for g in genes], "--",
            color="0.5", lw=1, label="closed form, homomers")
    ax.set_title(f"nu = {nu:.2f}", fontsize=10)
    ax.grid(alpha=.2)
for ax in axes[-1]:
    ax.set_xlabel("genes available, G")
for ax in axes[:, 0]:
    ax.set_ylabel("MI [bits]")
for ax in axes.flat[len(nus):]:
    ax.set_visible(False)
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=8,
           frameon=False)
fig.suptitle(f"filled circles: exact   hollow squares: counting   D = {D}\n"
             "band: 1 sd across replicates (environment, stoichiometry, repertoire)",
             fontsize=9, y=.94)
plt.tight_layout(rect=(0, 0, 1, .9))
plt.show()

# %%
# ── which channel carries the information ───────────────────────────────────
present = [mi for mi in CHANNELS if mi in points]
if present:
    fig, axes = plt.subplots(len(present), len(nus), figsize=(3.2 * len(nus), 3.2 * len(present)),
                             sharex=True, sharey="row", squeeze=False)
    for row, mi in zip(axes, present):
        for ax, nu in zip(row, nus):
            for ratio in sorted(points.ratio.unique()):
                curve = points[(points.nu == nu) & (points.ratio == ratio)].sort_values("n_genes")
                curve = curve.dropna(subset=[mi])
                if curve.empty:
                    continue
                colour = COLOURS[ratio]
                G = curve.n_genes.to_numpy()
                ax.plot(G, curve[mi], "o-", color=colour, lw=2, ms=6,
                        label=f"R/G = {ratio}" + ("  (homomers)" if ratio == 1 else ""))
                ax.fill_between(G, curve[f"{mi} lo"], curve[f"{mi} hi"],
                                color=colour, alpha=.15, linewidth=0)
            ax.grid(alpha=.2)
        row[0].set_ylabel(f"{CHANNELS[mi][1]} [bits]")
    for ax, nu in zip(axes[0], nus):
        ax.set_title(f"nu = {nu:.2f}", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("genes available, G")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=8,
               frameon=False)
    fig.suptitle("I(A; x) = H(A) - H(A | x), exact   band: 1 sd across replicates",
                 fontsize=9, y=.95)
    plt.tight_layout(rect=(0, 0, 1, .92))
    plt.show()

# %%
