# %%
"""Prototype: two ways to draw the KT bracket + environmental std on the
impact_of_heteromerization figure (MI vs n_genes, one series per ratio R/n_genes).


  - KT bracket (lower→upper) = certified methodological bound  -> a filled band
  - std across environments   = statistical spread             -> caps / thin band

A) single axes: bracket fill + error-bar caps (±std) on the LOWER bound (reported MI).
B) small multiples: one panel per ratio, bracket fill + std bands on BOTH bounds.
"""
import sys, os
from pathlib import Path
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.plotlib import load_runs, latest_sweep

GOAL  = "fig1"
KT_LO = "full_array_entropy_kt_mean"
KT_UP = "full_array_entropy_kt_upper_mean"
RATIOS = range(1, 6)
SAVE = Path(os.environ.get("SAVE_DIR", "."))   # PNGs for inspection

hete = latest_sweep(load_runs(GOAL, receptor_type="heteromer")).copy()
hete["het_ratio"] = (hete["R"] / hete["n_genes"]).round().astype(int)
levels = sorted(r for r in hete["het_ratio"].unique() if r in RATIOS)
cmap = plt.colormaps["viridis"]


def col(ratio):
    """viridis, but keep off the extreme-yellow / extreme-purple ends so the top and
    bottom series stay legible on a white ground."""
    span = max(1, max(levels) - min(levels))
    return cmap(0.12 + 0.78 * (ratio - min(levels)) / span)


def agg(ratio):
    """Per-n_genes mean/std of lower and upper bound across the environment runs."""
    s = hete[hete["het_ratio"] == ratio]
    g = s.groupby("n_genes").agg(lo=(KT_LO, "mean"), lo_sd=(KT_LO, "std"),
                                 up=(KT_UP, "mean"), up_sd=(KT_UP, "std")).sort_index()
    return g.fillna(0.0)


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")


# %% ── Variant A: single axes — bracket fill + error caps on the lower bound ──
figA, ax = plt.subplots(figsize=(5.6, 4.3))
style(ax)
ax.plot(np.arange(0, 16), np.arange(0, 16), color="0.6", ls=(0, (4, 3)), lw=1.0, zorder=1)
ax.text(15, 15, " perfect\n homomers", fontsize=7, color="0.5", va="top")
for ratio in levels:
    g = agg(ratio)
    if len(g) < 2:
        continue
    c = col(ratio)
    ax.fill_between(g.index, g["lo"], g["up"], color=c, alpha=0.16, lw=0, zorder=2)  # bracket
    ax.plot(g.index, g["lo"], color=c, lw=2.0, zorder=4)                             # reported MI
    ax.plot(g.index, g["up"], color=c, lw=0.9, alpha=0.6, zorder=3)                  # ceiling
    ax.errorbar(g.index, g["lo"], yerr=g["lo_sd"], fmt="none", ecolor=c,             # env std
                elinewidth=1.1, capsize=2.5, capthick=1.1, zorder=5)
ax.set_xlim(2, 15); ax.set_ylim(2, 30)
ax.set_xlabel("$n_{\\mathrm{genes}}$"); ax.set_ylabel("MI  [bits]")
ax.set_xticks([2, 5, 10, 15]); ax.set_yticks([5, 10, 15, 20, 25, 30])
handles = [Line2D([0], [0], color=col(r), lw=2.4,
                  label=fr"$R/n_g\approx{r}$") for r in levels]
handles += [Line2D([0], [0], color="0.35", lw=2.0, label="KT lower (MI)"),
            Line2D([0], [0], color="0.35", lw=0.9, alpha=0.6, label="KT upper"),
            Line2D([0], [0], color="0.35", lw=0, marker="|", label="± std (envs)")]
ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper left",
          bbox_to_anchor=(1.02, 1.0))
figA.suptitle("A — single axes: bracket fill + std caps on the lower bound", fontsize=10)
figA.tight_layout()
figA.savefig(SAVE / "variantA.png", dpi=130, bbox_inches="tight")

# %% ── Variant B: small multiples — bracket fill + std bands on both bounds ───
figB, axs = plt.subplots(1, len(levels), figsize=(2.5 * len(levels), 3.2),
                         sharex=True, sharey=True, squeeze=False)
for j, ratio in enumerate(levels):
    ax = axs[0][j]
    style(ax)
    c = col(ratio)
    ax.plot(np.arange(0, 16), np.arange(0, 16), color="0.7", ls=(0, (4, 3)), lw=0.9, zorder=1)
    g = agg(ratio)
    if len(g) >= 2:
        ax.fill_between(g.index, g["lo"], g["up"], color=c, alpha=0.14, lw=0, zorder=2)   # bracket
        ax.fill_between(g.index, g["lo"] - g["lo_sd"], g["lo"] + g["lo_sd"],
                        color=c, alpha=0.35, lw=0, zorder=3)                              # std lower
        ax.fill_between(g.index, g["up"] - g["up_sd"], g["up"] + g["up_sd"],
                        color=c, alpha=0.18, lw=0, zorder=3)                              # std upper
        ax.plot(g.index, g["lo"], color=c, lw=2.0, zorder=4)
        ax.plot(g.index, g["up"], color=c, lw=1.0, ls="--", alpha=0.8, zorder=4)
    ax.set_title(fr"$R/n_g\approx{ratio}$", fontsize=9)
    ax.set_xlim(2, 15); ax.set_ylim(2, 30)
    ax.set_xticks([2, 5, 10, 15])
    ax.set_xlabel("$n_{\\mathrm{genes}}$", fontsize=8)
    if j == 0:
        ax.set_ylabel("MI  [bits]")
figB.suptitle("B — small multiples: bracket fill + std bands on both bounds", fontsize=10)
figB.tight_layout()
figB.savefig(SAVE / "variantB.png", dpi=130, bbox_inches="tight")

plt.show()
# %%
