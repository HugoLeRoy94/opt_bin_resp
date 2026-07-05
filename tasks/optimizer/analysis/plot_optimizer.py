# %%
"""Optimizer comparison — the measured entropy estimators vs epoch.

Grid layout:
  rows = system type   (the fixed (n_genes, n_receptors) conditions)
  cols = optimizer     (the loss the run was trained on: the `entropy` column)

Each subplot overlays the three per-epoch entropy measurements (all in bits):
  full_array_entropy                    — plug-in / Shannon  (MI lower bound)
  full_array_entropy_blocked            — blocked            (MI upper bound)
  full_array_entropy_blocked_corrected  — Miller-Madow-corrected blocked

Reading it: a run trained on optimizer X (its column) but read out with all three
estimators shows whether that loss actually raises the *honest* MI (lower/corrected
tracking the upper bound) or just games its own estimator (upper bound climbing
while the plug-in lower bound stalls). y is shared per row so optimizers are
directly comparable within a system type.
"""
import sys
from pathlib import Path
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")  # exec dir

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.plotlib import load_runs, load_epochs, latest_sweep

GOAL = "optimizer"
FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)

# metric -> (legend label, colour, linestyle)
METRICS = {
    "full_array_entropy":                   ("plug-in (lower)",   "tab:orange", "--"),
    "full_array_entropy_blocked":           ("blocked (upper)",   "tab:blue",   "-"),
    "full_array_entropy_blocked_corrected": ("blocked corrected", "tab:green",  "-."),
}

df = latest_sweep(load_runs(GOAL))     # complete runs, newest sweep per condition
ep = load_epochs(df)                    # per-epoch stats + config cols (entropy, R…)

conditions = sorted(set(zip(df["n_genes"], df["n_receptors"])))   # rows
optimizers = sorted(df["entropy"].unique())                        # cols


def _draw(ax, sub, metric, color, ls):
    """Mean over runs at each epoch (+ std band if >1 run)."""
    g = sub.groupby("epoch")[metric].agg(["mean", "std", "count"]).sort_index()
    ax.plot(g.index.values, g["mean"].values, color=color, ls=ls, lw=1.3)
    if (g["count"] > 1).any():
        ax.fill_between(g.index.values,
                        (g["mean"] - g["std"]).values,
                        (g["mean"] + g["std"]).values,
                        color=color, alpha=0.2)


# %%
fig, axes = plt.subplots(
    len(conditions), len(optimizers),
    figsize=(3.6 * len(optimizers), 2.8 * len(conditions)),
    sharex="row", sharey="row", squeeze=False,
)

for r, (ng, nr) in enumerate(conditions):
    for c, opt in enumerate(optimizers):
        ax = axes[r][c]
        sub = ep[(ep["n_genes"] == ng) & (ep["n_receptors"] == nr) &
                 (ep["entropy"] == opt)]
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
        else:
            for metric, (_, color, ls) in METRICS.items():
                _draw(ax, sub, metric, color, ls)
        ax.set_ylim(0, nr)                 # MI is capped at R = n_receptors bits
        if r == 0:
            ax.set_title(f"optimizer: {opt}", fontsize=10)
        if c == 0:
            ax.set_ylabel(f"G={ng}, R={nr}\nentropy [bits]")
        if r == len(conditions) - 1:
            ax.set_xlabel("epoch")

handles = [Line2D([0], [0], color=col, ls=ls, label=lab)
           for lab, col, ls in METRICS.values()]
fig.legend(handles=handles, loc="upper center", ncol=len(METRICS), fontsize=9,
           bbox_to_anchor=(0.5, 1.0))
fig.suptitle("Optimizer comparison — entropy estimators vs epoch", y=1.03)
fig.tight_layout()
#plt.savefig(FIGURES / "entropy_vs_epoch_by_optimizer_and_system.png",dpi=150, bbox_inches="tight")
plt.show()

# %%
