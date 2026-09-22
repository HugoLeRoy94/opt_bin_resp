# %%
"""replicates: independent-run means, uncertainty, and one-gene baseline retention."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tasks.cells.gene_expression.analysis._shared import load_study, summarize, report, plot_curves

DATA = ROOT / "data" / "gene_expression"
# None selects the latest sweep per condition/coverage. Set explicit compatible
# folders to pool disjoint replicate ranges or select an older campaign.
SWEEPS = None
RUNS = load_study(DATA, "replicates", SWEEPS)
SUMMARY = summarize(RUNS)
report(SUMMARY)
# SUMMARY.to_csv(Path(__file__).with_name("replicates_summary.csv"), index=False)

# %%
if not SUMMARY.empty:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_curves(axes[0, 0], SUMMARY, "genes_per_cell", "mi_mean", error="mi_sem")
    plot_curves(axes[0, 1], SUMMARY, "genes_per_cell", "retained")
    axes[0, 1].axhline(1, color="gray", linestyle=":")
    plot_curves(axes[1, 0], SUMMARY, "genes_per_cell", "noise_mean")
    plot_curves(axes[1, 1], SUMMARY, "genes_per_cell", "represented_mean")
    axes[0, 0].set_ylabel("total MI [bits], mean ± SEM")
    axes[0, 1].set_ylabel("MI / mean one-gene MI")
    axes[1, 0].set_ylabel("H(response | input) [bits]")
    axes[1, 1].set_ylabel("genes represented in the array")
    for ax in axes.flat:
        ax.set_xlabel("genes expressed per cell")
    fig.tight_layout()
    # fig.savefig(Path(__file__).with_name("replicates.png"), dpi=180)
    plt.show()

# %%
# Each dot is an independently optimized run; repeated test batches are averaged.
if not RUNS.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    for labels, part in RUNS.groupby(["condition", "coverage"]):
        ax.scatter(part.genes_per_cell, part.mi, alpha=.6, label=", ".join(labels))
    ax.set(xlabel="genes expressed per cell", ylabel="total MI per optimization [bits]")
    ax.legend()
    fig.tight_layout()
    plt.show()
