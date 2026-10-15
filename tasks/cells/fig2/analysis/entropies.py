# %%
"""Where the MI comes from: total response entropy against the noise that eats it.

MI = H(Y) - H(Y|X). Plotting the two separately says whether a flat MI curve is a
flat H(Y) or a H(Y|X) that grew just as fast.

All three come from grouped_counting, so the identity holds exactly here: H(Y|X) is
analytic (a sum of binomial entropies over the groups), H(Y) is the sampled count
entropy plus the label term, and the label term cancels in the difference.
"""
import sys
from pathlib import Path

# Absolute, not Path(__file__): __file__ is undefined in an interactive window, so
# deriving the path from it makes the cells unrunnable in the IDE.
HERE = Path("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp/tasks/cells/fig2/analysis")
sys.path.insert(0, str(HERE))
import matplotlib.pyplot as plt
from _load import runs, STYLE

G, C = 5, 50

points = runs()
panel = points[(points["G"] == G) & (points["C"] == C)].sort_values("genes")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for arm, curve in panel.groupby("arm"):
    marker, colour = STYLE[arm]
    for ax, column in zip(axes, ("H(Y)", "H(Y|X)", "MI")):
        ax.plot(curve["genes"], curve[column], marker, color=colour, label=arm)
for ax, column in zip(axes, ("H(Y) response entropy", "H(Y|X) response noise",
                             "MI = H(Y) - H(Y|X)")):
    ax.set(xlabel="genes expressed per cell", ylabel="bits", title=column)
axes[0].legend(fontsize=8)
fig.suptitle(f"G = {G}, C = {C}")
plt.tight_layout()
plt.show()

# %%
# ── the dilution collapse test, across the WHOLE grid ───────────────────────
# Hypothesis: the fall of MI past its peak is dilution, i.e. a cell averaging more
# receptors drifts toward a coin. If so, H(Y|X) per cell is a function of RECEPTORS
# PER CELL alone, and every (G, C, strategy, condition) point collapses onto one
# curve. Scatter around that curve is whatever dilution does NOT explain.
#
# Receptors per cell is the x axis here, not genes per cell: under the interface
# model a cell expressing g genes assembles (g^5 + 4g)/5 receptors, so the two axes
# are wildly non-linear in each other and only one of them is the averaging width.
import numpy as np

whole = runs()

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for arm, curve in whole.groupby("arm"):
    marker, colour = STYLE[arm]
    axes[0].plot(curve["per_cell"], curve["H(Y|X)"] / curve["C"], marker[0],
                 color=colour, alpha=.7, label=arm)
    axes[1].plot(curve["per_cell"], curve["MI"], marker[0], color=colour, alpha=.7)
    axes[2].plot(curve["per_cell"], curve["H(Y)"] / curve["C"], marker[0],
                 color=colour, alpha=.7)
for ax, label in zip(axes, ("H(Y|X) per cell [bits]", "MI [bits]",
                            "H(Y) per cell [bits]")):
    ax.set(xlabel="receptors per cell", ylabel=label, xscale="log")
    ax.grid(alpha=.2)
axes[0].axhline(1.0, color="0.6", lw=.8, ls=":")      # a fair coin: 1 bit per cell
axes[0].legend(fontsize=7)
fig.suptitle("Every (G, C, strategy, condition) point. Collapse onto one curve in "
             "the left panel = dilution explains the noise.")
plt.tight_layout()
plt.show()

# How tight is the collapse? Spread of H(Y|X)/cell among points that share a
# receptors-per-cell value but differ in everything else.
binned = whole.assign(bin=np.log2(whole["per_cell"]).round()).groupby("bin").agg(
    points=("MI", "size"), per_cell=("per_cell", "median"),
    noise_mean=("H(Y|X)", lambda s: (s / whole.loc[s.index, "C"]).mean()),
    noise_std=("H(Y|X)", lambda s: (s / whole.loc[s.index, "C"]).std()))
print(binned.round(3).to_string())

# %%
