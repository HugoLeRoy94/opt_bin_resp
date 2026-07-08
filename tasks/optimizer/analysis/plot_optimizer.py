# %%
"""Optimizer comparison — the measured entropy estimators vs epoch.

Grid layout:
  rows = system type   (the fixed (n_genes, n_receptors) conditions)
  cols = optimizer     (the loss the run was trained on: the `entropy` column)

Each subplot overlays the three per-epoch entropy measurements (all in 
bits):
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
FIGURES = Path("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp/tasks/optimizer/figures")
FIGURES.mkdir(exist_ok=True)

# metric -> (legend label, colour, linestyle). Explicit per-estimator columns,
# logged via the opt-in measurement_fns in optimizer.py (not the loss-native
# `full_array_entropy`, which is redundant with whichever estimator matches the loss).
METRICS = {
    "full_array_entropy_collision":         ("collision H2 (lower)", "tab:orange", "--"),
    "full_array_entropy_blocked":           ("blocked (upper)",      "tab:blue",   "-"),
    "full_array_entropy_blocked_corrected": ("blocked corrected",    "tab:green",  "-."),
    "full_array_entropy_kt":                ("KT lower bound",       "tab:red",    ":"),
    "full_array_entropy_kt_upper":          ("KT upper bound",       "tab:purple", ":"),
}

df = latest_sweep(load_runs(GOAL))     # complete runs, newest sweep per condition
ep = load_epochs(df)                    # per-epoch stats + config cols (entropy, R…)

# recompute_backward may be stored as bool / 0-1 / "True" — normalise to a bool column.
def _as_bool(x):
    return str(x).strip().lower() in ("1", "true", "t", "yes")

ep["_rb"] = (ep["recompute_backward"].map(_as_bool)
             if "recompute_backward" in ep.columns else False)
df["_rb"] = (df["recompute_backward"].map(_as_bool)
             if "recompute_backward" in df.columns else False)

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


def _test_marker(ax, dsub, metric, color, x, alpha):
    """Final test value (test_results.json → `<metric>_mean` column) as a marker at
    the end of the training curve. The per-epoch curve is itself measured on
    test_batch_size, so this should sit at the curve's endpoint (10x-averaged)."""
    col = metric + "_mean"
    if col not in dsub.columns:
        return
    v = dsub[col].dropna()
    if v.empty:
        return
    ax.plot([x], [v.mean()], marker="o", color=color, ms=7, mec="k", mew=0.6,
            alpha=alpha, ls="none", zorder=5)


# %% ── FIRST PANEL: impact of the measurement sample count (few vs many) ────────
# Only KT changes with the test-batch size (blocked/collision are read on the
# unchanged eval chunk). The old sweeps measured KT on ~4096 samples (ceiling
# ~12 bits); the new large-test-batch sweeps measure it on min(2^R, memory). Here
# we overlay the KT bracket from a "few-sample" sweep vs a "many-sample" sweep.
# NOTE: the two sweeps are different runs (different random environments), so this
# shows the ceiling/resolution effect, not a same-environment delta.
FEW_SWEEP, MANY_SWEEP = None, None       # sweep_folder; None → earliest / latest with KT
CMP_OPT = "kt"                            # optimizer column to compare on
KT_METRICS = {"full_array_entropy_kt":       ("KT lower", "tab:red"),
              "full_array_entropy_kt_upper": ("KT upper", "tab:purple")}

cmp_ep = load_epochs(load_runs(GOAL, complete=False))   # include the running sweep
_kt = "full_array_entropy_kt"
_avail = (sorted(cmp_ep.dropna(subset=[_kt])["sweep_folder"].unique())
          if _kt in cmp_ep.columns else [])
print("sweeps with KT data:", _avail)
few  = FEW_SWEEP  or (_avail[0]  if _avail else None)
many = MANY_SWEEP or (_avail[-1] if _avail else None)
print(f"comparing  few={few}  vs  many={many}")

if few and many and few != many:
    fig0, ax0 = plt.subplots(1, len(conditions),
                             figsize=(4.0 * len(conditions), 3.2), squeeze=False)
    for c, (ng, nr) in enumerate(conditions):
        ax = ax0[0][c]
        for sweep, ls, lw in [(few, "--", 1.3), (many, "-", 2.4)]:
            s = cmp_ep[(cmp_ep["n_genes"] == ng) & (cmp_ep["n_receptors"] == nr) &
                       (cmp_ep["entropy"] == CMP_OPT) & (cmp_ep["sweep_folder"] == sweep)]
            for metric, (_, color) in KT_METRICS.items():
                if metric not in s.columns:
                    continue
                g = s.groupby("epoch")[metric].mean().dropna()
                if g.empty:
                    continue
                ax.plot(g.index.values, g.values, color=color, ls=ls, lw=lw)
        ax.set_ylim(0, nr)
        ax.set_title(f"G={ng}, R={nr}")
        ax.set_xlabel("epoch")
        if c == 0:
            ax.set_ylabel(f"KT entropy [bits]\noptimizer = {CMP_OPT}")
    h0 = [Line2D([0], [0], color=col, label=lab) for lab, col in KT_METRICS.values()]
    h0 += [Line2D([0], [0], color="0.3", ls="--", lw=1.3, label=f"few samples ({few[-6:]})"),
           Line2D([0], [0], color="0.3", ls="-",  lw=2.4, label=f"many samples ({many[-6:]})")]
    fig0.legend(handles=h0, loc="upper center", ncol=len(h0), fontsize=8,
                bbox_to_anchor=(0.5, 1.12))
    fig0.suptitle("Measurement sample count — KT bracket, few vs many samples", y=1.18)
    fig0.tight_layout()
    fig0.savefig(FIGURES / "kt_few_vs_many_samples.svg", bbox_inches="tight")
else:
    print("skipping few-vs-many panel: need two distinct sweeps with KT data")


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
                x_end = ssub["epoch"].max()
                dsub = df[(df["n_genes"] == ng) & (df["n_receptors"] == nr) &
                          (df["entropy"] == opt) & (df["_rb"] == rb)]
                for metric, (_, color, ls) in METRICS.items():
                    _draw(ax, ssub, metric, color, ls, **RB_STYLE[bool(rb)])
                    # final test value (o marker at the curve end)
                    _test_marker(ax, dsub, metric, color, x_end, RB_STYLE[bool(rb)]["alpha"])
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
handles += [Line2D([0], [0], color="0.3", marker="o", mec="k", ls="none",
                   label="final test value")]
fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=9,
           bbox_to_anchor=(0.5, 1.0))
fig.suptitle("Optimizer comparison — entropy estimators vs epoch", y=1.03)
fig.tight_layout()
plt.savefig(FIGURES / "entropy_vs_epoch_by_optimizer_and_system.svg", bbox_inches="tight")
#plt.show()

# %%
