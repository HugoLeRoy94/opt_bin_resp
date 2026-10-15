# %%
"""Did training converge?

One panel per (sampling strategy, condition), each holding every run of that arm at
the chosen G and C. Curves are coloured by genes expressed per cell, so the question
"does the expression level change how the optimizer behaves" is readable, not just
"where did it land".

This is `train_mutual_information` = -loss, logged for free from the gradient step
itself, so it is the TRAINING objective on the training batch, not a held-out
measurement. Flat at the end means the optimizer finished; still climbing means
EPOCHS is too small.

A fall over the second half is EXPECTED wherever the receptor pool is small. The
receptor temperature anneals from 3.0 to 0.05 over the first 80% of training, so the
objective is first optimised against SOFT, graded receptors that carry analog
information, then converges onto the hard binary regime where the whole array can
read at most R_pool bits. Runs whose soft peak exceeds R_pool fall onto it; runs
already below it do not move. The decay stops exactly where annealing stops.
"""
import sys
from pathlib import Path

# Absolute, not Path(__file__): __file__ is undefined in an interactive window, so
# deriving the path from it makes the cells unrunnable in the IDE.
HERE = Path("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp/tasks/cells/fig2/analysis")
sys.path.insert(0, str(HERE))
from _load import runs                   # also puts the repo root on sys.path
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from src.IO import SingleRunLoader

G, C = 5, 50
CMAP = plt.cm.viridis

points = runs()
panel = points[(points["G"] == G) & (points["C"] == C)].sort_values("genes")
norm = Normalize(panel["genes"].min(), panel["genes"].max())

fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
cells = {("complete", "heteromers"): axes[0, 0], ("complete", "homomers"): axes[0, 1],
         ("bernoulli", "heteromers"): axes[1, 0], ("bernoulli", "homomers"): axes[1, 1]}
for (strategy, condition), ax in cells.items():
    arm = panel[(panel["strategy"] == strategy) & (panel["condition"] == condition)]
    for _, row in arm.iterrows():
        loader = SingleRunLoader(row["run"])
        history = loader.load_history()
        # The logged "epoch" column is the MEASUREMENT INDEX, not the epoch: the runner
        # records every epochs//100 steps. Rescale, or the anneal end lands at 80 on a
        # 0-5000 axis and the curve looks like it decays for most of training.
        steps = history["epoch"].to_numpy() * max(1, loader.load_config().epochs // 100)
        ax.plot(steps, history["train_mutual_information"],
                color=CMAP(norm(row["genes"])))
    ax.set_title(f"{strategy} / {condition}")
    ax.grid(alpha=.2)
for ax in axes[1]:
    ax.set_xlabel("epoch")
for ax in axes.flat:
    ax.axvline(0.8 * 5000, color="0.6", lw=.8, ls=":")   # receptor annealing ends
for ax in axes[:, 0]:
    ax.set_ylabel("train MI [bits]")
fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), ax=axes,
             label="genes expressed per cell")
fig.suptitle(f"G = {G}, C = {C}")
plt.show()
