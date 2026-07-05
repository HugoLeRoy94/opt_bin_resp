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

# metric -> (legend label, colour, linestyle). Explicit per-estimator columns,
# logged via the opt-in measurement_fns in optimizer.py (not the loss-native
# `full_array_entropy`, which is redundant with whichever estimator matches the loss).
METRICS = {
    "full_array_entropy_collision":         ("collision H2 (lower)", "tab:orange", "--"),
    "full_array_entropy_blocked":           ("blocked (upper)",      "tab:blue",   "-"),
    "full_array_entropy_blocked_corrected": ("blocked corrected",    "tab:green",  "-."),
    "full_array_entropy_kt":                ("KT lower bound",       "tab:red",    ":"),
}

df = latest_sweep(load_runs(GOAL))     # complete runs, newest sweep per condition
ep = load_epochs(df)                    # per-epoch stats + config cols (entropy, R…)

# recompute_backward may be stored as bool / 0-1 / "True" — normalise to a bool column.
def _as_bool(x):
    return str(x).strip().lower() in ("1", "true", "t", "yes")

ep["_rb"] = (ep["recompute_backward"].map(_as_bool)
             if "recompute_backward" in ep.columns else False)

conditions = sorted(set(zip(df["n_genes"], df["n_receptors"])))   # rows
optimizers = sorted(df["entropy"].unique())                        # cols

# recompute_backward off vs on → distinguished by weight/alpha (estimator identity
# stays in colour+linestyle). Off = crisp thin, on = fat translucent.
RB_STYLE = {False: dict(lw=1.3, alpha=1.0), True: dict(lw=2.6, alpha=0.4)}


def _draw(ax, sub, metric, color, ls, **line_kw):
    """Mean over runs at each epoch (+ std band if >1 run). Skips absent metrics
    (e.g. full_array_entropy_kt on runs logged before it was measured)."""
    if metric not in sub.columns:
        return
    g = sub.groupby("epoch")[metric].agg(["mean", "std", "count"]).sort_index()
    g = g.dropna(subset=["mean"])
    if g.empty:
        return
    ax.plot(g.index.values, g["mean"].values, color=color, ls=ls, **line_kw)
    if (g["count"] > 1).any():
        ax.fill_between(g.index.values,
                        (g["mean"] - g["std"]).values,
                        (g["mean"] + g["std"]).values,
                        color=color, alpha=0.15)


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
            # one set of estimator lines per recompute_backward value (same env)
            for rb in sorted(sub["_rb"].unique()):
                ssub = sub[sub["_rb"] == rb]
                for metric, (_, color, ls) in METRICS.items():
                    _draw(ax, ssub, metric, color, ls, **RB_STYLE[bool(rb)])
        ax.set_ylim(0, nr)                 # MI is capped at R = n_receptors bits
        if r == 0:
            ax.set_title(f"optimizer: {opt}", fontsize=10)
        if c == 0:
            ax.set_ylabel(f"G={ng}, R={nr}\nentropy [bits]")
        if r == len(conditions) - 1:
            ax.set_xlabel("epoch")

# legend: estimators (colour+style), only those present (KT appears after a re-run)
handles = [Line2D([0], [0], color=col, ls=ls, label=lab)
           for metric, (lab, col, ls) in METRICS.items() if metric in ep.columns]
# + recompute_backward off/on (weight/alpha), only if both are in the data
if ep["_rb"].any():
    handles += [Line2D([0], [0], color="0.3", label="recompute off", **RB_STYLE[False]),
                Line2D([0], [0], color="0.3", label="recompute on",  **RB_STYLE[True])]
fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=9,
           bbox_to_anchor=(0.5, 1.0))
fig.suptitle("Optimizer comparison — entropy estimators vs epoch", y=1.03)
fig.tight_layout()
#plt.savefig(FIGURES / "entropy_vs_epoch_by_optimizer_and_system.png",dpi=150, bbox_inches="tight")
plt.show()

# %%
