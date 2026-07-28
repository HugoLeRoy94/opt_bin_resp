# %%
"""Per-epoch training-loss convergence for the het_casc KT runs — a smoothness check
for exactly the runs plotted by impact_of_heteromerization_kt.py (same selection:
latest_sweep of the fig1 heteromer runs).

het_casc runs with per_epoch_measure=False, so each logged step records the FREE
training objective (no extra sampling): `loss` = -(KT lower bound on the train batch)
and `train_entropy = -loss`. So this is the optimizer's own trace, not a measurement.

Grid: rows = n_genes, cols = heteromerization ratio R/n_genes ∈ {1..5}. Each panel
overlays the individual random-environment runs at that (n_genes, R) — individual
curves (not a mean) so any jaggedness is visible. x = training epoch (logged step ×
epochs//100, matching the run.py logging cadence). Non-monotonic early on is expected:
temperature anneals from soft→sharp over the first 80% of epochs.
"""
import sys
from pathlib import Path
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")  # exec dir

import numpy as np
import matplotlib.pyplot as plt

from src.plotlib import load_runs, load_epochs, latest_sweep

GOAL        = "fig1"
Y           = "train_entropy"   # +bits (= -loss); set to "loss" for the raw minimised loss
RATIO_RANGE = range(1, 6)

df = latest_sweep(load_runs(GOAL, receptor_type="heteromer"))
ep = load_epochs(df)
if Y not in ep.columns:
    raise SystemExit(f"no '{Y}' column in stats.csv — expected per_epoch_measure=False "
                     f"het_casc KT runs (got columns: {list(ep.columns)})")
ep = ep.copy()
ep["het_ratio"] = (ep["R"] / ep["n_genes"]).round().astype(int)
ep["train_epoch"] = ep["epoch"] * (ep["epochs"] // 100).clip(lower=1)   # logged step → true epoch

genes  = sorted(ep["n_genes"].unique())
ratios = sorted(r for r in ep["het_ratio"].unique() if r in RATIO_RANGE)

# %%
fig, axs = plt.subplots(len(genes), len(ratios),
                        figsize=(2.6 * len(ratios), 2.1 * len(genes)),
                        squeeze=False)
for r, g in enumerate(genes):
    for c, ratio in enumerate(ratios):
        ax = axs[r][c]
        sub = ep[(ep["n_genes"] == g) & (ep["het_ratio"] == ratio)]
        if sub.empty:
            ax.set_axis_off()
            continue
        for _, run in sub.groupby("path"):
            run = run.sort_values("train_epoch")
            ax.plot(run["train_epoch"], run[Y], lw=0.8, alpha=0.8)
        R = int(round(g * ratio))
        ax.set_title(f"G={g}, R={R}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_ylim(0, R if Y == "train_entropy" else None)   # MI ≤ R bits
        if c == 0:
            ax.set_ylabel(Y, fontsize=7)
        if r == len(genes) - 1:
            ax.set_xlabel("epoch", fontsize=7)

fig.suptitle("Per-epoch training loss — het_casc KT runs (convergence smoothness check)",
             y=1.005, fontsize=11)
fig.tight_layout()
plt.show()
# %%
