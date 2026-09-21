# %%
"""Compare the latest full-repertoire and homomer-only gene-expression sweeps.

Run the cells in order. Worlds are independent across sweep points and conditions,
so these figures are descriptive rather than paired estimates of a heteromer effect.
"""
import sys
from pathlib import Path
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.plotlib import DATA_ROOT, load_model
from src.IO import find_latest_sweep, SweepLoader, SingleRunLoader
from src.cells import CellReadout
from src.analysis_helper import (build_latent_umap, plot_latent_umap,
                                 cell_ligand_responses, plot_cell_response_umap)

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

CONDITIONS = (
    ("Full repertoire", "cell_gene_expression"),
    ("Homomers only", "homomer_gene_expression"),
)
RUNS = {}
for condition, prefix in CONDITIONS:
    sweep = find_latest_sweep(str(DATA_ROOT / "gene_expression"), prefix=prefix)[0]
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
        expected = len(loader.config._axes().get(
            "cell_receptors" if prefix == "homomer_gene_expression" else "cell_size_pmf", []
        ))
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
            "identity MI": np.mean(test["identity_channel"]),
            "grouped MI": np.mean(test.get("mutual_information_grouped", [np.nan])),
            "KT MI lower": np.mean(test.get("mutual_information_kt", [np.nan])),
            "KT MI upper": np.mean(test.get("mutual_information_kt_upper", [np.nan])),
            "count entropy": np.mean(test.get("grouped_count_entropy", [np.nan])),
            "count entropy ceiling": np.mean(test.get("grouped_count_entropy_upper", [np.nan])),
            "counting MI": np.mean(test["mutual_information_counting_mm"]),
            "response noise": np.mean(test["conditional_entropy_response"]),
        })
    summary = pd.DataFrame(rows)
    SUMMARIES[condition] = summary
    print(f"\n{condition}\n{summary.to_string(index=False)}")
    if summary.empty:
        continue

    ax_mi, ax_noise, ax_pool = axes[row]
    for metric, style in (
        ("identity MI", "o-"), ("grouped MI", "s-"), ("KT MI lower", "s-"),
        ("KT MI upper", "s--"), ("counting MI", "^:"),
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
