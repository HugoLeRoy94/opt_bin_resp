# %%
"""impact_of_heteromerization, KT-bracket version.

Standalone copy of Plot 7 from plots_hete_by_genes.py, adapted for the data produced
by the new fig1/het_casc.py (entropy="kt", bracket-only measurement): MI vs n_genes,
one curve per heteromerization ratio R/n_genes ∈ {1..5}. Each curve shows the KT
**lower** bound (solid, the reported MI = optimizer target) with its across-environment
std band, and the KT **upper** bound (dashed) — together the certified bracket on the
true joint entropy H(s). The y=x line marks perfect (lossless) homomers.

Reads GOAL="fig1" (where het_casc.py writes). The old annealed data lives in fig1_2 and
is untouched. A homomer reference band is drawn only if homomer KT runs exist in fig1.
"""
import sys
from pathlib import Path
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")  # exec dir

import numpy as np
import matplotlib.pyplot as plt

from src.plotlib import load_runs, latest_sweep

GOAL   = "fig1"                               # het_casc.py KT output
KT_LO  = "full_array_entropy_kt_mean"         # KT lower bound (reported MI)
KT_UP  = "full_array_entropy_kt_upper_mean"   # KT upper bound (bracket top)
RATIO_RANGE = range(1, 6)                     # R/n_genes levels to draw

FIGURES = Path("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp/tasks/fig1/figures")
FIGURES.mkdir(exist_ok=True)

hete = latest_sweep(load_runs(GOAL, receptor_type="heteromer"))
hete = hete.copy()
hete["het_ratio"] = (hete["R"] / hete["n_genes"]).round().astype(int)
RATIO_LEVELS = sorted(r for r in hete["het_ratio"].unique() if r in RATIO_RANGE)

cmap = plt.colormaps["viridis"]
norm = plt.Normalize(min(RATIO_LEVELS), max(RATIO_LEVELS)) if RATIO_LEVELS else None

# %%
fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# optional homomer reference band (only if homomer KT runs are present in this goal)
try:
    homo = latest_sweep(load_runs(GOAL, receptor_type="homomer"))
    if not homo.empty:
        h = homo.groupby("n_genes")[KT_LO].agg(["mean", "std"]).sort_index()
        ax.fill_between(h.index, h["mean"] - h["std"].fillna(0),
                        h["mean"] + h["std"].fillna(0), color="k", alpha=0.10)
except Exception:
    pass

for ratio in RATIO_LEVELS:
    sub = hete[hete["het_ratio"] == ratio]
    lo = sub.groupby("n_genes")[KT_LO].agg(["mean", "std"]).sort_index()
    up = sub.groupby("n_genes")[KT_UP].agg(["mean"]).sort_index().reindex(lo.index)
    if len(lo) < 2:
        continue
    color = cmap(norm(ratio))
    ax.plot(lo.index, lo["mean"], color=color, lw=1.8,
            label=fr"$R/n_\text{{genes}} \approx {ratio}$")            # KT lower (MI)
    band = lo["std"].fillna(0)
    ax.fill_between(lo.index, lo["mean"] - band, lo["mean"] + band, color=color, alpha=0.35)
    ax.plot(up.index, up["mean"], color=color, lw=1.0, ls="--")       # KT upper (bracket)

ax.plot(np.arange(0, 16), np.arange(0, 16), color="k", ls="--", label="perfect \nhomomers")
ax.set_xlabel("$n_{\\mathrm{genes}}$")
ax.set_ylabel("MI  [bits]")
ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.05, 1.0))
ax.set_xlim(2, 15)
ax.set_ylim(2, 30)
ax.tick_params(axis="both", direction="in")
ax.set_xticks([2, 5, 10, 15])
ax.set_yticks([2, 5, 10, 15, 20, 25])
fig.tight_layout()
plt.subplots_adjust(right=0.8)
plt.savefig(FIGURES / "impact_of_heteromerization_kt.svg", bbox_inches="tight")
plt.show()
# %%
