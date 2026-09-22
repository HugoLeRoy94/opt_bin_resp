# %%
"""environment: independent-run means, uncertainty, and one-gene baseline retention."""
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
RUNS = load_study(DATA, "environment", SWEEPS)
SUMMARY = summarize(RUNS)
report(SUMMARY)
# SUMMARY.to_csv(Path(__file__).with_name("environment_summary.csv"), index=False)

# %%
if not SUMMARY.empty:
    # Each profile changes a named environment parameter; all have one family.
    baseline = SUMMARY[SUMMARY.genes_per_cell == 1]
    profiles = list(dict.fromkeys(SUMMARY.profile))
    positions = {name: i for i, name in enumerate(profiles)}
    baseline = baseline.assign(profile_index=baseline.profile.map(positions))
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_curves(ax, baseline, "profile_index", "mi_mean",
                ("condition", "coverage", "n_genes"), error="mi_sem")
    ax.set_xticks(range(len(profiles)), profiles)
    ax.set(ylabel="one-gene baseline MI [bits], mean ± SEM", xlabel="environment profile")
    fig.tight_layout()
    plt.show()

# %%
if not SUMMARY.empty and (SUMMARY.genes_per_cell > 1).any():
    fig, axes = plt.subplots(1, len(profiles), figsize=(5 * len(profiles), 4), squeeze=False)
    for ax, profile in zip(axes.flat, profiles):
        part = SUMMARY[SUMMARY.profile == profile]
        plot_curves(ax, part, "fraction_expressed", "retained", ("condition", "coverage", "n_genes"))
        ax.axhline(1, color="gray", linestyle=":")
        ax.set(title=profile, xlabel="fraction of genes expressed per cell", ylabel="MI / mean one-gene MI")
    fig.tight_layout()
    # fig.savefig(Path(__file__).with_name("environment.png"), dpi=180)
    plt.show()
