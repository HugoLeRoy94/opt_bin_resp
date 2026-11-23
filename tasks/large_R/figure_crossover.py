# %%
"""large_R crossover: MI against R across R = D, with both asymptotic regimes.

MI is the exact codeword entropy against the continuous ligand cloud
(test_exact.py). Hard threshold, so H(Y|X) = 0 and the entropy is the MI.

    R < D    closed form MI = R eta(nu), dashed (theory.mi_predicted)
    R >> D   D log2 R + c, dotted. Only the offset c is fitted, per nu, on the
             points with R >= LARGE_R_FROM. The slope D is not fitted.
    R > D    large-R expansion theory.mi_large_R, dash-dot, nothing fitted:
             shell at each run's learned mean norm U and E_base, uniform
             directions, atoms |c| >= 1 (all on / all off) weighted apart.

Band: 1 sd across independently drawn worlds.
"""
import matplotlib.pyplot as plt
import numpy as np

from analysis.load import across_replicates, epochs, load_model, runs
from tasks._lib import energies as E
from tasks._lib import theory as T

GOAL = "large_R"
SWEEPS = ["crossover_R"]     # a full directory name pins one sweep
MI = "entropy_exact"

frame = runs(GOAL, sweeps=SWEEPS, latest_only=True)
D = int(frame.latent_dim.iloc[0])
LARGE_R_FROM = 2 * D
# nu is derived from the geometry and the concentration spread, not tagged.
frame = frame.assign(nu=[
    round(T.nu_from_columns(r.latent_dim, r.family_spread, r.kernel_lambda,
                            r.fixed_max_energy, r.conc_std), 3)
    for r in frame.itertuples()])


def expansion(r):
    """theory.mi_large_R at this run's learned mean norm and E_base."""
    if r.n_genes <= r.latent_dim:
        return np.nan
    env, *_ = load_model(r.run_dir)
    u = (env.unit_latent - env.family_latent[0]).numpy()
    m, sigma_c = T.from_log10(r.conc_mean, r.conc_std)
    p = T.Params(D=r.latent_dim, sigma=r.family_spread, lam=r.kernel_lambda,
                 Emax=r.fixed_max_energy, sigma_c=sigma_c, m=m)
    U = float(np.linalg.norm(u, axis=1).mean())
    return float(T.mi_large_R(p, U, float(E.ebase(env).mean()), [r.n_genes])[0])


frame = frame.assign(expansion=[expansion(r) for r in frame.itertuples()])
points = across_replicates(frame, keys=["nu", "n_genes"], metrics=[MI, "expansion"])
nus = sorted(points.nu.unique())
colours = plt.colormaps["viridis"](np.linspace(0, .85, len(nus)))

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, xscale in zip(axes, ["linear", "log"]):
    for colour, nu in zip(colours, nus):
        curve = points[points.nu == nu].sort_values("n_genes")
        R = curve.n_genes.to_numpy()
        ax.plot(R, curve[MI], "o-", color=colour, ms=4, label=f"nu = {nu:.2f}")
        ax.fill_between(R, curve[f"{MI} lo"], curve[f"{MI} hi"],
                        color=colour, alpha=.15, linewidth=0)
        ax.errorbar(R, curve[MI], yerr=[curve[MI] - curve[f"{MI} lo"],
                                        curve[f"{MI} hi"] - curve[MI]],
                    fmt="none", color=colour, capsize=3, lw=1)

        small = np.arange(1, D + 1)
        ax.plot(small, [T.mi_predicted(nu, int(r)) for r in small], "--",
                color=colour, lw=1.2)

        large = R >= LARGE_R_FROM
        c = np.mean(curve[MI].to_numpy()[large] - D * np.log2(R[large]))
        grid = np.geomspace(D, R.max(), 50)
        ax.plot(grid, (D) * np.log2(grid) + c, ":", color=colour, lw=1.5)

        big = R > D
        ax.plot(R[big], curve.expansion.to_numpy()[big], "-.", color=colour, lw=1.2)

    ax.axvline(D, color="k", lw=.8, ls=":")
    ax.set(xscale=xscale, xlabel="receptors R", ylabel="MI [bits]",
           title=f"D = {D},  R axis {xscale}")
    ax.grid(alpha=.2)
axes[0].plot([], [], "--", color="0.3", label="closed form, R <= D")
axes[0].plot([], [], ":", color="0.3", label=f"{D} log2 R + c, c fitted on R >= {LARGE_R_FROM}")
axes[0].plot([], [], "-.", color="0.3", label="large-R expansion, learned U and E_base")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%
# Convergence: training KT MI after each logged epoch, one line per run,
# coloured by R. A curve still rising at the end was stopped too early.
history = epochs(frame)
R_all = np.sort(frame.n_genes.unique())
norm = plt.matplotlib.colors.LogNorm(R_all[0], R_all[-1])
cmap = plt.get_cmap("plasma")
fig, axes = plt.subplots(1, len(nus), figsize=(5 * len(nus), 4), sharey=True,
                         squeeze=False)
for ax, nu in zip(axes[0], nus):
    for _, part in history[history.nu == nu].groupby("run_dir"):
        ax.plot(part.epoch, part.train_mutual_information, lw=.7,
                color=cmap(norm(part.n_genes.iloc[0])))
    ax.set(xlabel="epoch", title=f"nu = {nu:.2f}")
    ax.grid(alpha=.2)
axes[0][0].set_ylabel("training MI, KT bound [bits]")
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[0].tolist(),
             label="receptors R")
plt.show()

# %%
