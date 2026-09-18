# %%
"""Analyse the sweep over the number of genes expressed per cell."""
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

SWEEP = find_latest_sweep(
    str(DATA_ROOT / "gene_expression"), prefix="cell_gene_expression"
)[0]

# This task gives every cell in a run the same number of genes. Use the resolved
# gene sets saved in config.json rather than reconstructing the sweep settings.
RUNS = []
for cfg, run_dir in SweepLoader(SWEEP).iter_run_dirs():
    genes_per_cell = len(cfg.cell_gene_sets[0])
    assert all(len(genes) == genes_per_cell for genes in cfg.cell_gene_sets)
    RUNS.append((genes_per_cell, cfg, run_dir))
RUNS.sort()

print(f"sweep: {SWEEP}")
for genes_per_cell, cfg, run_dir in RUNS:
    print(f"{genes_per_cell} genes/cell, "
          f"{len(cfg.receptor_indices)} pooled receptors: {run_dir}")

# %%
# ── final information across expression levels ──────────────────────────────
rows = []
for genes_per_cell, cfg, run_dir in RUNS:
    with open(Path(run_dir) / "test_results.json") as stream:
        test = json.load(stream)
    rows.append({
        "genes_per_cell": genes_per_cell,
        "receptor_pool": len(cfg.receptor_indices),
        "identity MI": np.mean(test["identity_channel"]),
        "KT MI lower": np.mean(test["mutual_information_kt"]),
        "KT MI upper": np.mean(test["mutual_information_kt_upper"]),
        "counting MI": np.mean(test["mutual_information_counting_mm"]),
        "response noise": np.mean(test["conditional_entropy_response"]),
    })
summary = pd.DataFrame(rows)
display(summary) if "display" in globals() else print(summary.to_string(index=False))

fig, (ax_mi, ax_noise, ax_pool) = plt.subplots(1, 3, figsize=(15, 4.5))
for metric, style in (
    ("identity MI", "o-"),
    ("KT MI lower", "s-"),
    ("KT MI upper", "s--"),
    ("counting MI", "^:"),
):
    ax_mi.plot(summary["genes_per_cell"], summary[metric], style, label=metric)
ax_mi.set_ylabel("mutual information [bits]")
ax_mi.legend(fontsize=8)

ax_noise.plot(summary["genes_per_cell"], summary["response noise"], "o-")
ax_noise.set_ylabel("H(response | sniff) [bits]")

ax_pool.plot(summary["genes_per_cell"], summary["receptor_pool"], "o-")
ax_pool.set_ylabel("distinct receptors in pool")

for ax in (ax_mi, ax_noise, ax_pool):
    ax.set_xlabel("genes expressed per cell")
    ax.grid(axis="y", alpha=.2)
fig.tight_layout()
# fig.savefig(FIGURES / "gene_expression_final.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
# ── training trajectories ───────────────────────────────────────────────────
fig, (ax_mi, ax_noise) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
for genes_per_cell, cfg, run_dir in RUNS:
    history = SingleRunLoader(run_dir).load_history()
    steps = history["epoch"].to_numpy()
    if np.array_equal(steps, np.arange(len(history))):
        steps = steps * max(1, cfg.epochs // 100)  # legacy logging indices
    label = f"{genes_per_cell} genes/cell"
    ax_mi.plot(steps, history["mutual_information_kt"], label=label)
    ax_noise.plot(steps, history["conditional_entropy_response"], label=label)

ax_mi.set_ylabel("KT MI lower [bits]")
ax_noise.set_ylabel("H(response | sniff) [bits]")
ax_noise.set_xlabel("optimization update")
for ax in (ax_mi, ax_noise):
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.2)
fig.tight_layout()
# fig.savefig(FIGURES / "gene_expression_training.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
# ── one latent-space UMAP for every expression level ────────────────────────
MODELS = []
for genes_per_cell, cfg, run_dir in RUNS:
    env, physics, receptor_indices = load_model(run_dir=run_dir)
    gene_homomers = torch.arange(env.n_genes)[:, None].expand(-1, cfg.k_sub)
    embedding = build_latent_umap(env, gene_homomers)

    fig, ax = plt.subplots(figsize=(9, 7))
    plot_latent_umap(env, gene_homomers, ax=ax, embedding=embedding)
    ax.set_title(f"Latent-space UMAP — {genes_per_cell} genes/cell")
    fig.tight_layout()
    # fig.savefig(FIGURES / f"latent_umap_g{genes_per_cell}.png",
    #             dpi=180, bbox_inches="tight")
    plt.show()

    MODELS.append((genes_per_cell, cfg, run_dir, env, physics,
                   receptor_indices, embedding))

# %%
# ── one five-cell response UMAP for every expression level ──────────────────
CONCENTRATION = 1.0
for (genes_per_cell, cfg, run_dir, env, physics,
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
    fig.suptitle(
        f"{genes_per_cell} genes expressed per cell\n"
        f"Single-ligand responses at concentration {CONCENTRATION:g}"
    )
    # fig.savefig(FIGURES / f"response_umap_g{genes_per_cell}.png",
    #             dpi=180, bbox_inches="tight")
    plt.show()

# %%
