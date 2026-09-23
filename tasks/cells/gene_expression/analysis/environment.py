# %%
"""environment: independent-run means, uncertainty, and one-gene baseline retention."""
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
PROFILE_LABELS = {
    "base": "base\n100 ligands, nearly singleton",
    "ligands": "ligands\n300 ligands",
    "dimension": "dimension\n12 latent dimensions",
    "mixtures": "mixtures\nmixture count rate μ = 3",
    "combined": "combined\n300 ligands, 12D, μ = 3",
}
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
    ax.set_xticks(range(len(profiles)), [PROFILE_LABELS.get(p, p) for p in profiles])
    ax.set(title="One gene per cell: sensitivity to the environment",
           ylabel="one-gene baseline MI [bits], mean ± SEM", xlabel="environment profile")
    print("This figure compares environments at one gene per cell. At this baseline, "
          "both strategies contain only homomers; differences reflect independent "
          "worlds and optimizations. Profile names are categories, not a numeric scale.")
    fig.tight_layout()
    plt.show()

# %%
if not SUMMARY.empty and (SUMMARY.genes_per_cell > 1).any():
    fig, axes = plt.subplots(1, len(profiles), figsize=(5 * len(profiles), 4), squeeze=False)
    for ax, profile in zip(axes.flat, profiles):
        part = SUMMARY[SUMMARY.profile == profile]
        plot_curves(ax, part, "genes_per_cell", "retained", ("condition", "coverage", "n_genes"))
        ax.axhline(1, color="gray", linestyle=":")
        ax.set(title=profile, xlabel="genes expressed per cell", ylabel="MI / mean one-gene MI")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    # fig.savefig(Path(__file__).with_name("environment.png"), dpi=180)
    plt.show()
elif not SUMMARY.empty:
    print("Only one-gene baselines are available, so there is no expression sweep "
          "to plot. Run scripts/environment.py for both conditions without "
          "--baseline_only, using the same --profiles, to measure MI retention "
          "as genes per cell increases.")
