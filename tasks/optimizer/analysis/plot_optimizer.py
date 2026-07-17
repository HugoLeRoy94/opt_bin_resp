# %%
"""Optimizer comparison — the measured entropy estimators vs epoch.

Grid layout:
  rows = system type   (the fixed (n_genes, n_receptors) conditions)
  cols = optimizer     (the loss the run was trained on: the `entropy` column)

Each subplot overlays every measured estimator (all in bits):
  full_array_entropy_collision           — collision H2   (lower)
  full_array_entropy_blocked             — blocked        (upper, loose)
  full_array_entropy_blocked_corrected   — blocked corrected
  full_array_entropy_kt                  — KT lower bound
  full_array_entropy_kt_upper            — KT upper bound

Decision (recorded here): we use the **KT lower bound** as the optimizer target /
reported MI, and the **KT upper bound** for bracketing the true entropy. The other
estimators are kept as diagnostics.

Data selection: `best_per_cell` keeps, for each (condition, optimizer), the runs
from the NEWEST sweep that holds that cell. So the newest kt sweep is used for kt,
while older collision / annealed runs still show (they were not re-run as recently).
"""
import sys
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")  # exec dir

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.plotlib import load_runs, load_epochs

GOAL = "optimizer"

# metric -> (legend label, colour, linestyle)
METRICS = {
    "full_array_entropy_collision":         ("collision H2 (lower)", "tab:orange", "--"),
    "full_array_entropy_blocked":           ("blocked (upper)",      "tab:blue",   "-"),
    "full_array_entropy_blocked_corrected": ("blocked corrected",    "tab:green",  "-."),
    "full_array_entropy_kt":                ("KT lower bound",       "tab:red",    ":"),
    "full_array_entropy_kt_upper":          ("KT upper bound",       "tab:purple", ":"),
}


def _as_bool(x):
    return str(x).strip().lower() in ("1", "true", "t", "yes")


def best_per_cell(df):
    """For each (condition, optimizer) keep only the runs from the NEWEST sweep that
    holds that cell — older collision/annealed survive even when kt is in a newer,
    kt-only sweep (which plain latest_sweep would otherwise collapse everything to)."""
    best = df.groupby(["n_genes", "n_receptors", "entropy"])["sweep_folder"].transform("max")
    return df[df["sweep_folder"] == best]


_all = load_runs(GOAL)
_all = _all[_all["sweep_folder"].str.startswith("optimizer_")]  # exclude sample_limit_*
df = best_per_cell(_all)
ep = load_epochs(df)

ep["_rb"] = ep["recompute_backward"].map(_as_bool) if "recompute_backward" in ep.columns else False
df["_rb"] = df["recompute_backward"].map(_as_bool) if "recompute_backward" in df.columns else False

conditions = sorted(set(zip(df["n_genes"], df["n_receptors"])))   # rows
optimizers = sorted(df["entropy"].unique())                        # cols

# recompute_backward off vs on → weight/alpha (estimator identity = colour+linestyle).
RB_STYLE = {False: dict(lw=1.3, alpha=1.0), True: dict(lw=2.6, alpha=0.4)}


def _draw(ax, sub, metric, color, ls, **line_kw):
    """Mean over runs at each epoch (+ std band if >1 run); skips absent metrics."""
    if metric not in sub.columns:
        return
    g = sub.groupby("epoch")[metric].agg(["mean", "std", "count"]).sort_index()
    g = g.dropna(subset=["mean"])
    if g.empty:
        return
    ax.plot(g.index.values, g["mean"].values, color=color, ls=ls, **line_kw)
    if (g["count"] > 1).any():
        ax.fill_between(g.index.values, (g["mean"] - g["std"]).values,
                        (g["mean"] + g["std"]).values, color=color, alpha=0.15)


def _test_marker(ax, dsub, metric, color, x, alpha):
    """Final test value (test_results.json → `<metric>_mean`) as a marker at the curve
    end. It is measured on the FINAL (max-memory) batch, so for KT it sits ABOVE the
    per-epoch curve (which is measured on the smaller per-epoch test batch)."""
    col = metric + "_mean"
    if col not in dsub.columns:
        return
    v = dsub[col].dropna()
    if v.empty:
        return
    ax.plot([x], [v.mean()], marker="o", color=color, ms=7, mec="k", mew=0.6,
            alpha=alpha, ls="none", zorder=5)


# %%
fig, axes = plt.subplots(
    len(conditions), len(optimizers),
    figsize=(3.6 * len(optimizers), 2.8 * len(conditions)),
    sharex="row", sharey="row", squeeze=False,
)

for r, (ng, nr) in enumerate(conditions):
    for c, opt in enumerate(optimizers):
        ax = axes[r][c]
        sub = ep[(ep["n_genes"] == ng) & (ep["n_receptors"] == nr) & (ep["entropy"] == opt)]
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
        else:
            for rb in sorted(sub["_rb"].unique()):
                ssub = sub[sub["_rb"] == rb]
                x_end = ssub["epoch"].max()
                dsub = df[(df["n_genes"] == ng) & (df["n_receptors"] == nr) &
                          (df["entropy"] == opt) & (df["_rb"] == rb)]
                for metric, (_, color, ls) in METRICS.items():
                    _draw(ax, ssub, metric, color, ls, **RB_STYLE[bool(rb)])
                    _test_marker(ax, dsub, metric, color, x_end, RB_STYLE[bool(rb)]["alpha"])
        ax.set_ylim(0, nr)                 # MI is capped at R = n_receptors bits
        if r == 0:
            ax.set_title(f"optimizer: {opt}", fontsize=10)
        if c == 0:
            ax.set_ylabel(f"G={ng}, R={nr}\nentropy [bits]")
        if r == len(conditions) - 1:
            ax.set_xlabel("epoch")

handles = [Line2D([0], [0], color=col, ls=ls, label=lab)
           for metric, (lab, col, ls) in METRICS.items() if metric in ep.columns]
if ep["_rb"].any():
    handles += [Line2D([0], [0], color="0.3", label="recompute off", **RB_STYLE[False]),
                Line2D([0], [0], color="0.3", label="recompute on",  **RB_STYLE[True])]
handles += [Line2D([0], [0], color="0.3", marker="o", mec="k", ls="none", label="final test value")]
fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=9,
           bbox_to_anchor=(0.5, 1.0))
fig.suptitle("Optimizer comparison — entropy estimators vs epoch", y=1.03)
fig.tight_layout()
plt.show()

# %%
