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

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
