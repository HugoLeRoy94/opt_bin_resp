# %%
"""analytic_baseline: MI against R for each heteromerization index m.

m = distinct genes per receptor, n_genes = R throughout. m = 1 is the homomer
sweep (train_sweep_R.py), m >= 2 the heteromer arms
(train_sweep_R_heteromer.py). One panel per nu, which conc_std sets. nu is
computed from the config with the homomer definition for every m.

m is derived from each run's receptor_indices, not from the sweep name.

Solid: exact entropy (test_exact.py), continuous ligand cloud, no counting
ceiling. Hollow squares: counting, Miller-Madow (test_counting.py), capped at log2 B.
Hard threshold, so either entropy is the MI. Dashed grey: homomer closed form.
"""
import matplotlib.pyplot as plt
import numpy as np

from analysis.load import across_replicates, runs
from tasks._lib import theory as T
from train.io import load_config

GOAL = "analytic_baseline"
EXACT = "entropy_exact"
COUNT = "codeword_entropy_mm"

frame = runs(GOAL, sweeps=["homomer_R", "heteromer_m"], latest_only=True)


def index_of(run_dir):
    sizes = {len(set(r)) for r in load_config(run_dir).receptor_indices}
    assert len(sizes) == 1, f"{run_dir}: mixed heteromerization {sizes}"
    return sizes.pop()


frame = frame.assign(
    m=[index_of(d) for d in frame.run_dir],
    nu=[round(T.nu_from_columns(r.latent_dim, r.family_spread, r.kernel_lambda,
                                r.fixed_max_energy, r.conc_std), 4)
        for r in frame.itertuples()])
points = across_replicates(frame, keys=["nu", "m", "n_genes"])
nus = sorted(points.nu.unique())
ms = sorted(points.m.unique())
colours = dict(zip(ms, plt.colormaps["Blues"](np.linspace(.4, 1., len(ms)))))
D = int(frame.latent_dim.iloc[0])

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True, sharey=True)
for ax, nu in zip(axes.flat, nus):
    for m in ms:
        curve = points[(points.nu == nu) & (points.m == m)].sort_values("n_genes")
        if curve.empty:
            continue
        R = curve.n_genes.to_numpy()
        label = f"m = {m}" + (" (homomers)" if m == 1 else "")
        if EXACT in curve:
            ax.plot(R, curve[EXACT], "o-", color=colours[m], lw=2, ms=6, label=label)
            ax.fill_between(R, curve[f"{EXACT} lo"], curve[f"{EXACT} hi"],
                            color=colours[m], alpha=.15, linewidth=0)
        if COUNT in curve:
            # Larger than the exact markers: the two agree to ~0.1 bit, so a
            # same-size hollow marker would hide under the filled one.
            ax.plot(R, curve[COUNT], "s", color=colours[m], ms=11, mew=1.2,
                    markerfacecolor="none", zorder=3,
                    label=None if EXACT in curve else label)
    R_all = np.array(sorted(points.n_genes.unique()))
    ax.plot(R_all, [T.mi_predicted(nu, int(r)) for r in R_all], "--",
            color="0.5", lw=1, label="closed form, homomers")
    ax.axvline(D, color="0.3", lw=.8, ls=":")
    ax.set_title(f"nu = {nu:.2f}", fontsize=10)
    ax.grid(alpha=.2)
for ax in axes[-1]:
    ax.set_xlabel("receptors R (= genes)")
for ax in axes[:, 0]:
    ax.set_ylabel("MI [bits]")
for ax in axes.flat[len(nus):]:
    ax.set_visible(False)
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=8,
           frameon=False)
fig.suptitle(f"filled circles: exact   hollow squares: counting   dotted line: R = D = {D}\n"
             "band: 1 sd across worlds (absent with one replicate)",
             fontsize=9, y=.94)
plt.tight_layout(rect=(0, 0, 1, .9))
plt.show()

# %%
