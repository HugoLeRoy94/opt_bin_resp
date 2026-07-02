# %%
"""Sparsity experiment (Experiment C) — identity vs concentration coding,
homomer vs heteromer, as a function of mixture density mu_ligands_per_source.

The total-comparable split (reliable at low mu, where the presence pattern M
repeats) is the exact chain rule
    H(A) = identity_channel + concentration_channel = I(A;M) + H(A|M).
Reliance fractions (which sum to 1):
    identity_frac      = identity_channel      / (identity_channel + concentration_channel)
    concentration_frac = concentration_channel / (identity_channel + concentration_channel)
The architecture with the larger concentration_frac leans more on level coding.

For the DENSE regime (where the joint split degenerates because every sniff has a
unique M) fall back to the per-ligand marginals mutual_information_ligand vs
mutual_information_concentration, compared to each other (not to the total).
"""
import sys
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")  # exec dir

import matplotlib.pyplot as plt

from analysis.plotlib import load_runs, latest_sweep, plot_metric

GOAL = "concentration_vs_family_spread"
X    = "mu_ligands_per_source"

ID   = "identity_channel_mean"            # I(A;M)
CONC = "concentration_channel_mean"       # H(A|M)
MIL  = "mutual_information_ligand_mean"          # per-ligand identity marginal
MIC  = "mutual_information_concentration_mean"   # per-ligand concentration marginal


def _load(tag):
    df = latest_sweep(load_runs(GOAL, date=tag)).copy()
    missing = [c for c in (ID, CONC) if c not in df.columns]
    if missing:
        raise KeyError(
            f"{tag}: runs.db is missing {missing}. These columns only exist for runs "
            f"executed AFTER identity_channel/concentration_channel were added to "
            f"measurement_fns. Re-run scripts/concentration_vs_family_spread/{tag}.py "
            f"(the old rows also used the buggy concentration metric), then reload."
        )
    total = df[ID] + df[CONC]                       # = H(A) on the Shannon estimator
    df["identity_frac"]      = df[ID]   / total
    df["concentration_frac"] = df[CONC] / total
    return df


hom = _load("sparsity_hom")
het = _load("sparsity_het")

# %% ── Panel 1: the two channels in bits (the crossover) ───────────────────────
fig, ax = plt.subplots()
plot_metric(hom, y=ID,   x=X, ax=ax, label="homomer  identity I(A;M)",      color="C0")
plot_metric(hom, y=CONC, x=X, ax=ax, label="homomer  concentration H(A|M)", color="C0", ls="--")
plot_metric(het, y=ID,   x=X, ax=ax, label="heteromer identity I(A;M)",      color="C1")
plot_metric(het, y=CONC, x=X, ax=ax, label="heteromer concentration H(A|M)", color="C1", ls="--")
ax.set_xscale("log")
ax.set_xlabel("mu_ligands_per_source  (dense → ; sparse ←)")
ax.set_ylabel("bits")
ax.set_title("Identity vs concentration channels (total-comparable, reliable at low mu)")
ax.legend(fontsize=8)
plt.show()

# %% ── Panel 2: reliance fractions (sum to 1) ──────────────────────────────────
fig, ax = plt.subplots()
plot_metric(hom, y="concentration_frac", x=X, ax=ax, label="homomer  conc. fraction",  color="C0")
plot_metric(het, y="concentration_frac", x=X, ax=ax, label="heteromer conc. fraction",  color="C1")
ax.axhline(0.5, color="k", ls=":", lw=.8)
ax.set_xscale("log")
ax.set_ylim(0, 1)
ax.set_xlabel("mu_ligands_per_source")
ax.set_ylabel("concentration fraction  H(A|M)/H(A)")
ax.set_title("How much each architecture leans on concentration coding")
ax.legend(fontsize=8)
plt.show()

# %% ── Panel 3: per-ligand marginals (dense-regime diagnostic) ─────────────────
# Compare these to EACH OTHER (same per-ligand normalization), not to H(A).
fig, ax = plt.subplots()
plot_metric(hom, y=MIL, x=X, ax=ax, label="homomer  identity (per ligand)",      color="C0")
#plot_metric(hom, y=MIC, x=X, ax=ax, label="homomer  concentration (per ligand)", color="C0", ls="--")
plot_metric(het, y=MIL, x=X, ax=ax, label="heteromer identity (per ligand)",      color="C1")
#plot_metric(het, y=MIC, x=X, ax=ax, label="heteromer concentration (per ligand)", color="C1", ls="--")
ax.set_xscale("log")
ax.set_xlabel("mu_ligands_per_source")
ax.set_ylabel("bits (per-ligand marginal)")
ax.set_title("Per-ligand marginals — corrected concentration should rise toward sparse")
ax.legend(fontsize=8)
plt.show()

# %%
