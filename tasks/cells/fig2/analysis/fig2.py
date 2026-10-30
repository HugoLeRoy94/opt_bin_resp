# %%
"""fig2: MI against genes per cell, one curve per sampling strategy and condition.

Each point is the mean over independent worlds; the band is their min/max. A gap
between two curves that is no wider than the bands is not a result.
"""
import sys
from pathlib import Path

# Absolute, not Path(__file__): __file__ is undefined in an interactive window, so
# deriving the path from it makes the cells unrunnable in the IDE.
HERE = Path("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp/tasks/cells/fig2/analysis")
sys.path.insert(0, str(HERE))
import matplotlib.pyplot as plt
from _load import runs, across_replicates, STYLE

G, C = 5, 50          # which array the figure is for
KEY = "MI"            # from grouped_counting; see _load.METRICS

points = across_replicates(runs())
panel = points[(points["G"] == G) & (points["C"] == C)].sort_values("target")

fig, ax = plt.subplots(figsize=(7, 5))
for arm, curve in panel.groupby("arm"):
    marker, colour = STYLE[arm]
    ax.plot(curve["genes"], curve[KEY], marker, color=colour, label=arm)
    # band = min to max over the independent worlds, not a standard error
    ax.fill_between(curve["genes"], curve[f"{KEY} lo"], curve[f"{KEY} hi"],
                    color=colour, alpha=.15, linewidth=0)
ax.set(xlabel="genes expressed per cell", ylabel=f"{KEY} [bits]",
       title=f"G = {G} genes available, C = {C} cells")
ax.legend()
plt.tight_layout()
plt.show()

# %%
# ── every C available at this G, three columns ──────────────────────────────
# The headline figure fixes one array size. This says whether its shape survives
# the number of cells, or is a property of C = 50 alone.
COLUMNS = 3

grid = points[points["G"] == G].sort_values("target")
cell_counts = sorted(grid["C"].unique())
rows = -(-len(cell_counts) // COLUMNS)

fig, axes = plt.subplots(rows, COLUMNS, figsize=(5 * COLUMNS, 4 * rows),
                         sharey=True, squeeze=False)
for ax, n_cells in zip(axes.flat, cell_counts):
    for arm, curve in grid[grid["C"] == n_cells].groupby("arm"):
        marker, colour = STYLE[arm]
        ax.plot(curve["genes"], curve[KEY], marker, color=colour, label=arm)
        ax.fill_between(curve["genes"], curve[f"{KEY} lo"], curve[f"{KEY} hi"],
                        color=colour, alpha=.15, linewidth=0)
    ax.set(title=f"C = {n_cells} cells", xlabel="genes expressed per cell")
for ax in axes.flat[len(cell_counts):]:
    ax.set_visible(False)
for ax in axes[:, 0]:
    ax.set_ylabel(f"{KEY} [bits]")
axes[0, 0].legend(fontsize=8)
fig.suptitle(f"G = {G} genes available")
plt.tight_layout()
plt.show()

# %%
