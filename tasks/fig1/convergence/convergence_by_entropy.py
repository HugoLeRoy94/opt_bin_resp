# %%
"""
Convergence comparison across entropy loss types.

Goals loaded: fig1_1
Entropy types: shannon | collision | blocked | annealed
  (blocked_to_corrected skipped — only n_genes>=15, no overlap with shannon)

Layout: rows = difficulty condition, cols = entropy type.
Each subplot shows per-run MI vs epoch, normalised by max(upper_bound) of that run:
  - solid blue  : full_array_entropy_blocked  (upper bound)
  - dashed orange: full_array_entropy          (lower bound / true-MI proxy)
Divergence between the two = loss is being gamed.
"""
import sys
from pathlib import Path
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.plotlib import load_runs, load_epochs, latest_sweep

GOAL = "fig1_1"

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)

ENTROPY_TYPES = ["shannon", "collision", "blocked", "annealed"]

# (n_genes, n_receptors) — easy / medium / hard
# shannon limited to ng<=10, nr<=14 → shows "no data" for medium/hard
CONDITIONS = [
    (5,  9,  "easy"),
    (10, 30, "medium"),
    (20, 30, "hard"),
]

UPPER = "full_array_entropy_blocked"
LOWER = "full_array_entropy"

# %%
fig, axes = plt.subplots(
    len(CONDITIONS), len(ENTROPY_TYPES),
    figsize=(4 * len(ENTROPY_TYPES), 3 * len(CONDITIONS)),
    sharex=False, sharey=False,
)

for row, (ng, nr, label) in enumerate(CONDITIONS):
    for col, ent in enumerate(ENTROPY_TYPES):
        ax = axes[row, col]

        runs = load_runs(GOAL, receptor_type="heteromer", entropy=ent, n_genes=ng)
        runs = runs[runs["n_receptors"] == nr]
        runs = latest_sweep(runs)

        if runs.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
        else:
            ep = load_epochs(runs)
            norm = ep[UPPER].max()
            if norm > 0:
                for path, grp in ep.groupby("path"):
                    grp = grp.sort_values("epoch")
                    ax.plot(grp["epoch"], grp[UPPER] / norm,
                            color="tab:blue", alpha=0.5, lw=0.9)
                    ax.plot(grp["epoch"], grp[LOWER] / norm,
                            color="tab:orange", alpha=0.5, lw=0.9, ls="--")

        ax.set_ylim(0, 1.05)
        if row == 0:
            ax.set_title(ent, fontsize=11)
        if col == 0:
            ax.set_ylabel(f"{label}\n(ng={ng}, nr={nr})\nMI / max(MI)")
        if row == len(CONDITIONS) - 1:
            ax.set_xlabel("epoch")

handles = [
    Line2D([0], [0], color="tab:blue",   lw=1.5, label="upper bound (blocked)"),
    Line2D([0], [0], color="tab:orange", lw=1.5, ls="--", label="lower bound"),
]
fig.legend(handles=handles, loc="upper right", fontsize=9)
fig.suptitle("Convergence by entropy loss type — fig1_1", fontsize=13)
fig.tight_layout()
plt.savefig(FIGURES / "convergence_by_entropy_type.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# ── Absolute MI comparison — one subplot per condition, all entropy types overlaid ──

ENT_COLORS = {
    "shannon":   "tab:green",
    "collision": "tab:red",
    "blocked":   "tab:purple",
    "annealed":  "tab:blue",
}

fig2, axes2 = plt.subplots(
    1, len(CONDITIONS),
    figsize=(5 * len(CONDITIONS), 4),
    sharey=False,
)

for col, (ng, nr, label) in enumerate(CONDITIONS):
    ax = axes2[col]
    for ent, color in ENT_COLORS.items():
        runs = load_runs(GOAL, receptor_type="heteromer", entropy=ent, n_genes=ng)
        runs = runs[runs["n_receptors"] == nr]
        runs = latest_sweep(runs)
        if runs.empty:
            continue
        ep = load_epochs(runs)
        for path, grp in ep.groupby("path"):
            grp = grp.sort_values("epoch")
            ax.plot(grp["epoch"], grp[UPPER],
                    color=color, alpha=0.5, lw=0.9)
            ax.plot(grp["epoch"], grp[LOWER],
                    color=color, alpha=0.5, lw=0.9, ls="--")

    ax.set_title(f"{label} (ng={ng}, nr={nr})", fontsize=10)
    ax.set_xlabel("epoch")
    if col == 0:
        ax.set_ylabel("MI [bits]")

handles2 = (
    [Line2D([0], [0], color=c, lw=1.5, label=e) for e, c in ENT_COLORS.items()]
    + [Line2D([0], [0], color="k", lw=1.5, label="upper"),
       Line2D([0], [0], color="k", lw=1.5, ls="--", label="lower")]
)
fig2.legend(handles=handles2, loc="upper right", fontsize=9)
fig2.suptitle("Absolute MI by entropy loss type — fig1_1", fontsize=13)
fig2.tight_layout()
plt.savefig(FIGURES / "absolute_mi_by_entropy_type.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
