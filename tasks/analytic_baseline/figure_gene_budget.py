# %%
"""analytic_baseline, gene budget arm: MI against the number of genes G, homomers vs heteromers.

m = 1: homomers, R = G (analytic_baseline homomer_R).
m >= 2: R = c G heteromers from the same G genes, exactly m distinct genes each.
Dashed black: homomers with c G genes, the unconstrained array with as many
receptors as the heteromer one, read from the same homomer_R sweep.

m, G and c are derived from each run (receptor_indices, n_genes), not from the
sweep name. nu uses the homomer definition for every m. One panel per nu.

Filled circles: exact entropy (test_exact.py --prefix gene_budget). Hollow
squares: counting, Miller-Madow (test_counting.py --prefix gene_budget).
Band: 1 sd across replicates, which average the environment, the
stoichiometry and the gene repertoire.
"""
import matplotlib.pyplot as plt
import numpy as np

from analysis.load import across_replicates, runs
from tasks._lib import theory as T
from train.io import load_config

EXACT = "entropy_exact"
COUNT = "codeword_entropy_mm"
COLOURS = dict(zip(range(1, 6), plt.colormaps["Blues"](np.linspace(.4, 1., 5))))


def index_of(run_dir):
    sizes = {len(set(r)) for r in load_config(run_dir).receptor_indices}
    assert len(sizes) == 1, f"{run_dir}: mixed heteromerization {sizes}"
    return sizes.pop()


def derive(frame):
    return frame.assign(
        m=[index_of(d) for d in frame.run_dir],
        c=(frame.n_receptors_resolved / frame.n_genes).round(3),
        nu=[round(T.nu_from_columns(r.latent_dim, r.family_spread, r.kernel_lambda,
                                    r.fixed_max_energy, r.conc_std), 4)
            for r in frame.itertuples()])


GOAL = "analytic_baseline"
hetero_runs = runs(GOAL, sweeps=["gene_budget"], latest_only=True)
hetero = across_replicates(derive(hetero_runs),
                           keys=["nu", "m", "c", "n_genes"])
homo = across_replicates(
    derive(runs(GOAL, sweeps=["homomer_R"], latest_only=True)),
    keys=["nu", "m", "c", "n_genes"])
nus = sorted(hetero.nu.unique())
cs = sorted(hetero.c.unique())
G_max = hetero.n_genes.max()
D = int(hetero_runs.latent_dim.iloc[0])


def draw(ax, curve, colour, label, line="o-"):
    G = curve.n_genes.to_numpy()
    if EXACT in curve:
        ax.plot(G, curve[EXACT], line, color=colour, lw=2, ms=6, label=label)
        ax.fill_between(G, curve[f"{EXACT} lo"], curve[f"{EXACT} hi"],
                        color=colour, alpha=.15, linewidth=0)
    if COUNT in curve:
        # Larger than the exact markers, which they would otherwise hide under.
        ax.plot(G, curve[COUNT], "s", color=colour, ms=11, mew=1.2,
                markerfacecolor="none", zorder=3,
                label=None if EXACT in curve else label)


# %%
for c in cs:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True, sharey=True)
    for ax, nu in zip(axes.flat, nus):
        h = homo[(homo.nu == nu) & (homo.n_genes <= G_max)].sort_values("n_genes")
        draw(ax, h, COLOURS[1], "m = 1 (homomers, R = G)")
        for m in sorted(hetero.m.unique()):
            curve = hetero[(hetero.nu == nu) & (hetero.m == m)
                           & (hetero.c == c)].sort_values("n_genes")
            if not curve.empty:
                draw(ax, curve, COLOURS[m], f"m = {m}, R = {c:g} G")
        # Homomers with c G genes, plotted at G: the same R without the constraint.
        ref = homo[homo.nu == nu].set_index("n_genes")
        G_ref = [G for G in sorted(hetero.n_genes.unique()) if round(c * G) in ref.index]
        if G_ref and EXACT in ref:
            ax.plot(G_ref, [ref.loc[round(c * G), EXACT] for G in G_ref], "--",
                    color="k", lw=1, label=f"homomers with {c:g} G genes")
        ax.set_title(f"nu = {nu:.2f}", fontsize=10)
        ax.grid(alpha=.2)
    for ax in axes[-1]:
        ax.set_xlabel("genes G")
    for ax in axes[:, 0]:
        ax.set_ylabel("MI [bits]")
    for ax in axes.flat[len(nus):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"filled circles: exact   hollow squares: counting   D = {D}\n"
                 "band: 1 sd across replicates (environment, stoichiometry, repertoire)",
                 fontsize=9, y=.92)
    plt.tight_layout(rect=(0, 0, 1, .88))
    plt.show()

# %%
