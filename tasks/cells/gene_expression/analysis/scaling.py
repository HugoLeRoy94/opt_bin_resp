# %%
"""scaling: independent-run means, uncertainty, and one-gene baseline retention."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tasks.cells.gene_expression.analysis._shared import load_study, summarize, report, plot_curves

DATA = ROOT / "data" / "gene_expression"
# None selects the latest sweep per condition/coverage. Set explicit compatible
# folders to pool disjoint replicate ranges or select an older campaign.
SWEEPS = None
RUNS = load_study(DATA, "scaling", SWEEPS)
SUMMARY = summarize(RUNS)
report(SUMMARY)
# SUMMARY.to_csv(Path(__file__).with_name("scaling_summary.csv"), index=False)

# %%
if not SUMMARY.empty:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    groups = ("condition", "coverage", "n_genes")
    plot_curves(axes[0, 0], SUMMARY, "genes_per_cell", "mi_mean", groups, error="mi_sem")
    plot_curves(axes[0, 1], SUMMARY, "genes_per_cell", "retained", groups)
    axes[0, 1].axhline(1, color="gray", linestyle=":")
    baseline = SUMMARY[SUMMARY.genes_per_cell == 1]
    plot_curves(axes[1, 0], baseline, "n_genes", "mi_mean", error="mi_sem")
    plot_curves(axes[1, 1], SUMMARY, "genes_per_cell", "states_max", groups)
    axes[1, 1].set_yscale("log", base=2)
    axes[0, 0].set(xlabel="genes expressed per cell", ylabel="total MI [bits], mean ± SEM")
    axes[0, 1].set(xlabel="genes expressed per cell", ylabel="MI / mean one-gene MI")
    axes[1, 0].set(xlabel="available genes", ylabel="one-gene baseline MI [bits]")
    axes[1, 1].set(xlabel="genes expressed per cell", ylabel="joint count states S (max across replicates)")
    for ax in axes.flat:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    print("Count states S = product(n_j + 1), for identical-cell group sizes n_j. "
          "This is the enumeration cost; log2(S) bounds count entropy, not attained MI.")
    fig.tight_layout()
    # fig.savefig(Path(__file__).with_name("scaling.png"), dpi=180)
    plt.show()

# %%
# Coverage and physics cost: these help distinguish missing genes from pooling loss.
if not SUMMARY.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_curves(axes[0], SUMMARY, "genes_per_cell", "represented_mean", groups)
    plot_curves(axes[1], SUMMARY, "genes_per_cell", "pool_mean", groups)
    axes[0].set_ylabel("genes represented in the array")
    axes[1].set_ylabel("distinct receptor types across the whole array")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("genes expressed per cell")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    print("The receptor pool counts the union across cells. With complete coverage, "
          "homomers already represent all G types at one gene/cell; expressing g "
          "genes gives g homomer types per cell, but still G across the array.")
    fig.tight_layout()
    plt.show()
