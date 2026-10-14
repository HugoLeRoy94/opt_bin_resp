# %%
"""fig2: MI against genes per cell, one curve per sampling strategy and condition."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from _load import runs, STYLE

G, C = 5, 50          # which array the figure is for
KEY = "MI"            # from grouped_counting; see _load.METRICS

points = runs()
panel = points[(points["G"] == G) & (points["C"] == C)].sort_values("genes")

fig, ax = plt.subplots(figsize=(7, 5))
for arm, curve in panel.groupby("arm"):
    marker, colour = STYLE[arm]
    ax.plot(curve["genes"], curve[KEY], marker, color=colour, label=arm)
ax.set(xlabel="genes expressed per cell", ylabel=f"{KEY} [bits]",
       title=f"G = {G} genes available, C = {C} cells")
ax.legend()
plt.tight_layout()
plt.show()
