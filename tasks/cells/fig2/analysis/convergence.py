# %%
"""Did training converge, and does the measurement depend on the evaluation budget?

Top row: the training objective against epoch, one line per run, with the last fifth
magnified on the right. Flat at the end means the optimizer finished; still climbing
means EPOCHS is too small. This is `train_mutual_information` = -loss, logged for
free from the gradient step itself, not a separate evaluation.

Bottom row: the MI against the number of input samples used to measure it, from
scripts/evaluation_budget.py, beside the Good-Turing missing mass. Sampled counting
is biased DOWN when the response alphabet is undersampled, so a curve still rising at
the largest budget is reporting the budget, not the array.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load import runs, STYLE          # also puts the repo root on sys.path
import pandas as pd
import matplotlib.pyplot as plt
from src.IO import SingleRunLoader

G, C = 5, 50

points = runs()
panel = points[(points["G"] == G) & (points["C"] == C)].sort_values("genes")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for arm, curve in panel.groupby("arm"):
    marker, colour = STYLE[arm]
    for _, row in curve.iterrows():
        history = SingleRunLoader(row["run"]).load_history()
        tail = history[history["epoch"] >= 0.8 * history["epoch"].max()]
        axes[0, 0].plot(history["epoch"], history["train_mutual_information"],
                        color=colour, alpha=.6)
        axes[0, 1].plot(tail["epoch"], tail["train_mutual_information"],
                        color=colour, alpha=.6)

    # scripts/evaluation_budget.py writes these; skip the bottom row without them
    ladders = [Path(row["run"]) / "budgets.json" for _, row in curve.iterrows()]
    budgets = [record for path in ladders if path.exists()
               for record in json.loads(path.read_text())["records"]]
    if not budgets:
        continue
    ladder = pd.DataFrame(budgets).groupby("samples").mean(numeric_only=True)
    axes[1, 0].plot(ladder.index, ladder["mutual_information_grouped_counting_mm"],
                    marker, color=colour, label=arm)
    axes[1, 1].plot(ladder.index, ladder["grouped_counting_missing_mass"],
                    marker, color=colour, label=arm)

axes[0, 0].set(xlabel="epoch", ylabel="train MI [bits]", title="training objective")
axes[0, 1].set(xlabel="epoch", ylabel="train MI [bits]", title="last 20% of training")
axes[1, 0].set(xlabel="input samples", ylabel="MI [bits]", xscale="log",
               title="measurement against evaluation budget")
axes[1, 1].set(xlabel="input samples", ylabel="Good-Turing missing mass",
               xscale="log", yscale="log", title="coverage of the response alphabet")
axes[1, 0].legend(fontsize=7)
axes[1, 1].legend(fontsize=7)
fig.suptitle(f"G = {G}, C = {C}")
plt.tight_layout()
plt.show()
