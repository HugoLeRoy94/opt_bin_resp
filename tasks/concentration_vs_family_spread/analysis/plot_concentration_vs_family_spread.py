# %%
"""concentration_vs_family_spread plots — MI (full_array_entropy) of homomer vs
heteromer arrays (R = 10) as a function of the environmental spread, swept along
two axes:

  panel 1 : family_spread  (latent dispersion of ligands within a family)
  panel 2 : conc_std        (log-concentration dispersion)

Each panel overlays the two receptor strategies; solid = blocked estimator
(MI upper bound), dashed = plug-in estimator (MI lower bound).  plot_metric
aggregates over the N_RUNS samples at each x, drawing mean ± std.

Notes on filtering this goal:
  * Sweeps are NOT distinguishable by the `sweep_name` column — the shallow
    run layout makes it parse to "run" for every row.  Select instead by the
    `sweep_folder` prefix via load_runs(..., date="conc_hom").
  * `conc_std` is a tuple-typed config field (it can be per-ligand), so runs.db
    does NOT store it as a column.  attach_cfg() reads the scalar value back
    from each run's saved SingleRunConfig.
"""
import sys
from pathlib import Path
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")  # exec dir

import matplotlib.pyplot as plt

from src.plotlib import (load_runs, latest_sweep, plot_metric, load_run,
                         load_epochs, DATA_ROOT)

GOAL      = "concentration_vs_family_spread"
METRIC    = "full_array_entropy_blocked_mean"   # MI upper bound
METRIC_LO = "full_array_entropy_mean"           # MI lower bound

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)


def attach_cfg(df, field):
    """Add `field` as a column, read from each run's saved SingleRunConfig.

    Needed for tuple-typed config fields (conc_mean, conc_std) that the runs.db
    schema skips.  Returns a copy with the new column.
    """
    df = df.copy()
    df[field] = [getattr(load_run(run_dir=str(DATA_ROOT / GOAL / p))[0], field)
                 for p in df["path"]]
    return df


def _panel(hom, het, x, title, ax=None):
    """Overlay homomer (C0) vs heteromer (C1); upper bound solid, lower dashed."""
    ax = plot_metric(hom, y=METRIC,    x=x, ax=ax, label="homomer (upper)",   color="C0")
    plot_metric(het, y=METRIC,    x=x, ax=ax, label="heteromer (upper)", color="C1")
    ax.set_xscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel("full_array_entropy  [bits]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax

# %%
fig,ax = plt.subplots(ncols=3,figsize=(12,4))
sparse_hom = load_runs(GOAL,date='sparsity_hom')
print(sparse_hom.columns)
plot_metric(sparse_hom,y='full_array_entropy_mean',x='mu_ligands_per_source',ax=ax[0])
plot_metric(sparse_hom,y='mutual_information_ligand_mean',x='mu_ligands_per_source',ax=ax[1])
plot_metric(sparse_hom,y='mutual_information_concentration_mean',x='mu_ligands_per_source',ax=ax[2])
plt.savefig(FIGURES / "sparsity_channels.png", dpi=150, bbox_inches="tight")
plt.show()

# %% ── Panel 1: vary family_spread (concentration fixed) ───────────────────────
fam_hom = latest_sweep(load_runs(GOAL, date="family_hom"))
fam_het = latest_sweep(load_runs(GOAL, date="family_het"))
_panel(fam_hom, fam_het, x="family_spread",
       title="Homomers vs heteromers — varying family_spread")
plt.savefig(FIGURES / "mi_vs_family_spread.png", dpi=150, bbox_inches="tight")
plt.show()

# %% ── Panel 2: vary conc_std (family_spread fixed) ────────────────────────────
con_hom = attach_cfg(latest_sweep(load_runs(GOAL, date="conc_hom")), "conc_std")
con_het = attach_cfg(latest_sweep(load_runs(GOAL, date="conc_het")), "conc_std")
_panel(con_hom, con_het, x="conc_std",
       title="Homomers vs heteromers — varying conc_std")
plt.savefig(FIGURES / "mi_vs_conc_std.png", dpi=150, bbox_inches="tight")
plt.show()

# %% ── Both panels side by side ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
_panel(fam_hom, fam_het, x="family_spread",
       title="varying family_spread", ax=axes[0])
_panel(con_hom, con_het, x="conc_std",
       title="varying conc_std", ax=axes[1])
fig.tight_layout()
plt.savefig(FIGURES / "mi_both_panels.png", dpi=150, bbox_inches="tight")
plt.show()


# %% ── Convergence check ───────────────────────────────────────────────────────
def _conv(df, group, title):
    ep = load_epochs(df)
    ax = plot_metric(ep, y="full_array_entropy", x="epoch",
                     group=group, cmap="viridis", err=None)
    ax.set_xlabel("epoch")
    ax.set_ylabel("full_array_entropy_blocked  [bits]")
    ax.set_title(title)
    return ax


_conv(fam_hom, "family_spread", "Convergence — homomer, vary family_spread")
plt.savefig(FIGURES / "conv_hom_family_spread.png", dpi=150, bbox_inches="tight")
plt.show()

_conv(fam_het, "family_spread", "Convergence — heteromer, vary family_spread")
plt.savefig(FIGURES / "conv_het_family_spread.png", dpi=150, bbox_inches="tight")
plt.show()

_conv(con_hom, "conc_std", "Convergence — homomer, vary conc_std")
plt.savefig(FIGURES / "conv_hom_conc_std.png", dpi=150, bbox_inches="tight")
plt.show()

_conv(con_het, "conc_std", "Convergence — heteromer, vary conc_std")
plt.savefig(FIGURES / "conv_het_conc_std.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
