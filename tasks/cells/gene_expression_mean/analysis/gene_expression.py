# %%
"""Compare the latest full-repertoire and homomer-only gene-expression sweeps.

Run the cells in order. Worlds are independent across sweep points and conditions,
so these figures are descriptive rather than paired estimates of a heteromer effect.
What IS matched is the gene-set sampling (same cell seed), so at each expression
level the two conditions hold the same cells and differ only in whether heteromers
may assemble.

PRIMARY_MI is the metric for the two-curve figure. It must be the metric both
sweeps optimized, so it follows ENTROPY in the scripts: "grouped MI" for
ENTROPY="grouped_mi" (the current setting), "KT MI lower" for ENTROPY="kt_mi".
Reading a measured-but-not-optimized metric as the headline would rank the two
conditions on something neither of them was trained to maximise. The
identity/concentration channel split is only estimable when presence patterns
repeat, i.e. near-single-ligand sniffs; at mu_ligands_per_source > 1 rows are
nearly unique, concentration MI collapses toward 0 and identity MI drifts up
toward H(A). Those two columns are printed but not used as the headline curve.
"""
import sys
from pathlib import Path

ROOT = Path("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")
HERE = ROOT / "tasks/cells/gene_expression_mean/analysis"
FIGURES = HERE.parent / "figures"
DATA = ROOT / "data" / "gene_expression"
assert HERE.is_dir(), HERE
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.plotlib import load_model
from src.IO import find_latest_sweep, SweepLoader, SingleRunLoader
from src.cells import CellReadout
from src.analysis_helper import (build_latent_umap, plot_latent_umap,
                                 cell_ligand_responses, plot_cell_response_umap)
FIGURES.mkdir(exist_ok=True)

# Sweep-name prefixes, matching SWEEP_NAME in the two scripts. Point DATA and these
# prefixes at gene_expression_threshold to read the threshold arm with the same code.
CONDITIONS = (
    ("Full repertoire", "mean_heteromers"),
    ("Homomers only", "mean_homomers"),
)
RUNS = {}
for condition, prefix in CONDITIONS:
    sweep = find_latest_sweep(str(DATA), prefix=prefix)[0]
    loader = SweepLoader(sweep)
    runs = []
    for cfg, run_dir in loader.iter_run_dirs():
        if not all((Path(run_dir) / name).is_file() for name in
                   ("test_results.json", "stats.csv", "best_model.pt")):
            print(f"Skipping incomplete run: {run_dir}")
            continue
        genes_per_cell = len(cfg.cell_gene_sets[0])
        assert all(len(genes) == genes_per_cell for genes in cfg.cell_gene_sets)
        runs.append((genes_per_cell, cfg, run_dir))
    runs.sort(key=lambda run: run[0])
    RUNS[condition] = runs
    print(f"{condition}: {sweep}")
    print(f"  completed genes/cell: {[run[0] for run in runs]}")
    if loader.config is not None:
        axes = loader.config._axes()
        expected = len(axes.get("cell_receptors") or axes.get("cell_gene_sets") or [])
        if len(runs) < expected:
            print(f"  INCOMPLETE SWEEP: {len(runs)}/{expected} runs available")

# %%
# ── final information: the same three panels for each condition ─────────────
SUMMARIES = {}
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for row, (condition, _) in enumerate(CONDITIONS):
    rows = []
    for genes_per_cell, cfg, run_dir in RUNS[condition]:
        with open(Path(run_dir) / "test_results.json") as stream:
            test = json.load(stream)
        rows.append({
            "genes_per_cell": genes_per_cell,
            "receptor_pool": len(cfg.receptor_indices),
            "mu_ligands": cfg.mu_ligands_per_source,
            "conc_mean_span": float(np.ptp(cfg.conc_mean)),
            "identity MI": np.mean(test.get("identity_channel", [np.nan])),
            "grouped MI": np.mean(test.get("mutual_information_grouped", [np.nan])),
            "KT MI lower": np.mean(test.get("mutual_information_kt", [np.nan])),
            "KT MI upper": np.mean(test.get("mutual_information_kt_upper", [np.nan])),
            "count entropy": np.mean(test.get("grouped_count_entropy", [np.nan])),
            "count entropy ceiling": np.mean(test.get("grouped_count_entropy_upper", [np.nan])),
            "counting MI": np.mean(test.get("mutual_information_counting_mm", [np.nan])),
            "response noise": np.mean(test.get("conditional_entropy_response", [np.nan])),
        })
    summary = pd.DataFrame(rows)
    SUMMARIES[condition] = summary
    print(f"\n{condition}\n{summary.to_string(index=False)}")
    if summary.empty:
        continue

    ax_mi, ax_noise, ax_pool = axes[row]
    for metric, style in (
        ("grouped MI", "s-"), ("KT MI lower", "s-"), ("KT MI upper", "s--"),
        ("counting MI", "^:"), ("identity MI", "o--"),
    ):
        if summary[metric].notna().any():
            ax_mi.plot(summary["genes_per_cell"], summary[metric], style, label=metric)
    ax_mi.set_ylabel(f"{condition}\nmutual information [bits]")
    ax_mi.legend(fontsize=8)
    ax_noise.plot(summary["genes_per_cell"], summary["response noise"], "o-")
    ax_noise.set_ylabel("H(response | sniff) [bits]")
    ax_pool.plot(summary["genes_per_cell"], summary["receptor_pool"], "o-")
    ax_pool.set_ylabel("distinct receptors in pool")

for ax in axes.flat:
    ax.set_xlabel("genes expressed per cell")
    ax.grid(axis="y", alpha=.2)
fig.tight_layout()
# fig.savefig(FIGURES / "gene_expression_both_final.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
# ── THE two-curve figure: MI vs genes/cell, homomers against full repertoire ──
# Both sweeps optimize PRIMARY_MI, so this is a like-for-like comparison. The
# count-alphabet ceiling log2(n_states), drawn per condition whenever the grouped
# estimator ran. Grouping is LOSSLESS for MI, so this is a real ceiling on the array,
# not an estimator artifact — and the KT estimators are bounded by the same number
# even though they see full labelled responses. It is non-monotonic in genes per cell
# because C(G, g) is: for G=5, C=30 complete coverage it runs 14.0, 18.8, 18.6, 13.7,
# 5.0 bits across g=1..5. A curve ON the line is saturating its combinatorics; a curve
# BELOW it is limited by the receptors.
PRIMARY_MI = "grouped MI"   # must match ENTROPY in the sweep scripts
STYLES = {"Full repertoire": ("o-", "tab:blue"), "Homomers only": ("s-", "tab:orange")}

fig, ax = plt.subplots(figsize=(7, 5))
for condition, _ in CONDITIONS:
    summary = SUMMARIES[condition]
    if summary.empty or not summary[PRIMARY_MI].notna().any():
        print(f"{condition}: no {PRIMARY_MI} to plot")
        continue
    style, colour = STYLES[condition]
    ax.plot(summary["genes_per_cell"], summary[PRIMARY_MI], style,
            color=colour, label=condition)
    if PRIMARY_MI == "grouped MI" and summary["count entropy ceiling"].notna().any():
        ax.plot(summary["genes_per_cell"], summary["count entropy ceiling"],
                ":", color=colour, alpha=.5,
                label=f"{condition} — count alphabet ceiling")
ax.set_xlabel("genes expressed per cell")
ax.set_ylabel(f"{PRIMARY_MI} [bits]")
regimes = {(row["mu_ligands"], round(row["conc_mean_span"], 2))
           for summary in SUMMARIES.values() for _, row in summary.iterrows()}
ax.set_title("Information per sniff: heteromers allowed or not\n"
             + "; ".join(f"mu={mu:g} ligands/sniff, conc means over "
                         f"{span:g} decades" for mu, span in sorted(regimes)))
ax.grid(axis="y", alpha=.2)
ax.legend(fontsize=8)
fig.tight_layout()
# fig.savefig(FIGURES / "gene_expression_mi_vs_genes.png", dpi=180, bbox_inches="tight")
plt.show()

# The heteromer contribution, where both conditions reached the same point.
both = (SUMMARIES["Full repertoire"][["genes_per_cell", PRIMARY_MI]]
        .merge(SUMMARIES["Homomers only"][["genes_per_cell", PRIMARY_MI]],
               on="genes_per_cell", suffixes=(" full", " homomer")))
both["heteromer gain"] = both[f"{PRIMARY_MI} full"] - both[f"{PRIMARY_MI} homomer"]
print(f"\nHeteromer gain in {PRIMARY_MI} [bits]\n{both.to_string(index=False)}")

# %%
# ── training trajectories, one column per condition ────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex="col")
for col, (condition, _) in enumerate(CONDITIONS):
    for genes_per_cell, cfg, run_dir in RUNS[condition]:
        history = SingleRunLoader(run_dir).load_history()
        steps = history["epoch"].to_numpy()
        if np.array_equal(steps, np.arange(len(history))):
            steps = steps * max(1, cfg.epochs // 100)  # legacy logging indices
        label = f"{genes_per_cell} genes/cell"
        mi_key = ("mutual_information_grouped" if "mutual_information_grouped" in history
                  else "mutual_information_kt")
        estimator = "grouped" if mi_key == "mutual_information_grouped" else "KT lower"
        axes[0, col].plot(steps, history[mi_key], label=f"{label} ({estimator})")
        axes[1, col].plot(steps, history["conditional_entropy_response"], label=label)
    axes[0, col].set_title(condition)
    axes[0, col].legend(fontsize=8)
    axes[1, col].legend(fontsize=8)
    axes[1, col].set_xlabel("optimization update")
axes[0, 0].set_ylabel("mutual information [bits]")
axes[1, 0].set_ylabel("H(response | sniff) [bits]")
for ax in axes.flat:
    ax.grid(axis="y", alpha=.2)
fig.tight_layout()
# fig.savefig(FIGURES / "gene_expression_both_training.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
# ── one latent UMAP per completed run in both conditions ────────────────────
MODELS = []
for condition, _ in CONDITIONS:
    for genes_per_cell, cfg, run_dir in RUNS[condition]:
        env, physics, receptor_indices = load_model(run_dir=run_dir)
        gene_homomers = torch.arange(env.n_genes)[:, None].expand(-1, cfg.k_sub)
        embedding = build_latent_umap(env, gene_homomers)
        fig, ax = plt.subplots(figsize=(9, 7))
        plot_latent_umap(env, gene_homomers, ax=ax, embedding=embedding)
        ax.set_title(f"{condition} — {genes_per_cell} genes/cell")
        fig.tight_layout()
        # fig.savefig(FIGURES / f"latent_umap_{condition}_{genes_per_cell}.png",
        #             dpi=180, bbox_inches="tight")
        plt.show()
        MODELS.append((condition, genes_per_cell, cfg, run_dir, env, physics,
                       receptor_indices, embedding))

# %%
# ── one per-cell response UMAP per completed run ────────────────────────────
CONCENTRATION = 1.0
for (condition, genes_per_cell, cfg, run_dir, env, physics,
     receptor_indices, embedding) in MODELS:
    checkpoint = SingleRunLoader(run_dir).load_checkpoint(map_location="cpu")
    readout = CellReadout(
        checkpoint["readout_state"]["W"], mode=checkpoint["readout_mode"],
        temperature=checkpoint["readout_temperature"], k_sub=cfg.k_sub,
        learnable_threshold=cfg.cell_threshold_learnable,
    )
    readout.load_state_dict(checkpoint["readout_state"])
    readout.eval()
    responses = cell_ligand_responses(
        env, physics, receptor_indices, readout, CONCENTRATION
    )
    fig, axes = plot_cell_response_umap(
        embedding, responses, cfg.cell_gene_sets, concentration=CONCENTRATION
    )
    fig.suptitle(f"{condition} — {genes_per_cell} genes/cell\n"
                 f"Single-ligand responses at concentration {CONCENTRATION:g}")
    # fig.savefig(FIGURES / f"response_umap_{condition}_{genes_per_cell}.png",
    #             dpi=180, bbox_inches="tight")
    plt.show()

# %%
