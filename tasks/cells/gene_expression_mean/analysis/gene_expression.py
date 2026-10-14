# %%
"""Plot any set of gene-expression sweeps, chosen by hand.

Cell 1 prints an inventory of every sweep on disk, with what it actually ran.
Cell 2 holds SWEEPS: paste the folder names you want from that table. Nothing is
picked automatically — "the latest sweep" is how a KT-trained run once ended up on
the same axis as three exactly-trained ones without anyone noticing.

SWEEPS may mix goals. GOALS below lists the folders that get scanned, so a
gene_expression_mean sweep and a gene_expression_threshold sweep can sit on one
figure. Everything downstream is keyed on the sweep folder name.

PRIMARY_MI is the metric for the overlay figure. It must be the metric the sweeps
optimized, so it follows ENTROPY in the scripts: "grouped MI" for
ENTROPY="grouped_mi", "KT MI lower" for ENTROPY="kt_mi". Reading a
measured-but-not-optimized metric as the headline would rank runs on something none
of them was trained to maximise.

Two columns to distrust: identity MI and concentration MI split H(A) by grouping
sniffs with the same presence pattern, which needs those patterns to REPEAT. That
holds at one ligand per sniff and fails in a mixture, where nearly every row is
unique, concentration MI collapses toward 0 and identity MI drifts up toward H(A).
The inventory prints mu so you can see which regime a sweep was in.
"""
import sys
from pathlib import Path

ROOT = Path("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")
HERE = ROOT / "tasks/cells/gene_expression_mean/analysis"
FIGURES = HERE.parent / "figures"
assert HERE.is_dir(), HERE
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.plotlib import load_model
from src.IO import SweepLoader, SingleRunLoader
from src.cells import CellReadout
from src.analysis_helper import (build_latent_umap, plot_latent_umap,
                                 cell_ligand_responses, plot_cell_response_umap)
FIGURES.mkdir(exist_ok=True)

# Data folders scanned for sweeps. Add one and its sweeps join the inventory.
GOALS = [ROOT / "data" / "gene_expression",
         ROOT / "data" / "gene_expression_threshold"]

REQUIRED = ("test_results.json", "stats.csv", "best_model.pt")


def load_runs(sweep_dir):
    """Completed runs in a sweep, as (genes_per_cell, cfg, run_dir), sorted."""
    runs = []
    for cfg, run_dir in SweepLoader(str(sweep_dir)).iter_run_dirs():
        if not all((Path(run_dir) / name).is_file() for name in REQUIRED):
            continue
        genes_per_cell = len(cfg.cell_gene_sets[0])
        runs.append((genes_per_cell, cfg, run_dir))
    return sorted(runs, key=lambda run: run[0])


def condition_of(cfg):
    """Homomer sweeps state their repertoires outright; heteromer ones pass gene sets."""
    return "homomers" if cfg.cell_receptors is not None else "heteromers"


def levels_of(runs):
    """Compact "1-5" / "1,3,5" for the expression levels, plus runs per level.

    The raw list is unreadable once a sweep has replicates: a 90-run scaling sweep
    prints ninety digits.
    """
    from collections import Counter
    counts = Counter(g for g, _, _ in runs)
    levels = sorted(counts)
    span = (f"{levels[0]}-{levels[-1]}"
            if len(levels) > 2 and levels == list(range(levels[0], levels[-1] + 1))
            else ",".join(map(str, levels)))
    reps = sorted(set(counts.values()))
    return span, (reps[0] if len(reps) == 1 else f"{reps[0]}-{reps[-1]}")


def inventory():
    """Every sweep under GOALS, described by what its runs actually recorded.

    Read from each run's own config.json, so this works for sweeps written by any
    version of the scripts, with or without an experiment.json manifest.
    """
    rows = []
    for goal in GOALS:
        if not goal.is_dir():
            continue
        for sweep_dir in sorted(p for p in goal.iterdir() if p.is_dir()):
            runs = load_runs(sweep_dir)
            loader = SweepLoader(str(sweep_dir))
            planned = ""
            if loader.config is not None:
                axes = loader.config._axes()
                n = len(axes.get("cell_receptors") or axes.get("cell_gene_sets")
                        or axes.get("cell_size_pmf") or [])
                planned = n or ""
            if not runs:
                rows.append({"sweep": sweep_dir.name, "goal": goal.name,
                             "complete": 0, "planned": planned})
                continue
            _, cfg, _ = runs[0]
            span, reps = levels_of(runs)
            rows.append({
                "sweep": sweep_dir.name,
                "goal": goal.name,
                "condition": condition_of(cfg),
                "readout": cfg.cell_readout,
                "entropy": cfg.entropy,
                "G": cfg.n_genes,
                "C": len(cfg.cell_gene_sets),
                "genes/cell": span,
                "reps": reps,
                "mu": f"{cfg.mu_ligands_per_source:g}",
                "conc span": round(float(np.ptp(cfg.conc_mean)), 2),
                "epochs": cfg.epochs,
                "complete": len(runs),
                "planned": planned,
            })
    return pd.DataFrame(rows)


INVENTORY = inventory()
print(f"{len(INVENTORY)} sweeps under {', '.join(g.name for g in GOALS)}\n")
print(INVENTORY.to_string(index=False) if not INVENTORY.empty else "  (none)")
print("\nCopy the folder names you want into SWEEPS in the next cell.")

# %%
# ════════════════════════════════════════════════════════════════════════════
# WHICH SWEEPS GO ON THE FIGURES — paste folder names from the table above
# ════════════════════════════════════════════════════════════════════════════
SWEEPS = [
    "mean_heteromers_20261013_111933",
    "mean_heteromers_20261013_130403",
    "mean_heteromers_20261013_130949",
    "mean_homomers_20261013_111944"
]

if not SWEEPS:
    raise SystemExit("SWEEPS is empty: paste folder names from the inventory above.")

LOCATIONS = {p.name: p for goal in GOALS if goal.is_dir()
             for p in goal.iterdir() if p.is_dir()}
missing = [s for s in SWEEPS if s not in LOCATIONS]
if missing:
    raise SystemExit(f"Not found under GOALS: {missing}")

RUNS, LABELS = {}, {}
for sweep in SWEEPS:
    runs = load_runs(LOCATIONS[sweep])
    if not runs:
        raise SystemExit(f"{sweep} has no completed run.")
    RUNS[sweep] = runs
    cfg = runs[0][1]
    # short enough for a legend, specific enough to tell two sweeps apart
    LABELS[sweep] = f"{condition_of(cfg)}, {cfg.cell_readout}  [{sweep.split('_')[-1]}]"
    print(f"{sweep}\n  {LABELS[sweep]}: genes/cell {[g for g, _, _ in runs]}, "
          f"entropy={cfg.entropy}, mu={cfg.mu_ligands_per_source:g}")

# %%
# ── one row of panels per selected sweep ────────────────────────────────────
SUMMARIES = {}
fig, axes = plt.subplots(len(SWEEPS), 3, figsize=(15, 4.5 * len(SWEEPS)), squeeze=False)
for row, sweep in enumerate(SWEEPS):
    rows = []
    for genes_per_cell, cfg, run_dir in RUNS[sweep]:
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
    SUMMARIES[sweep] = summary
    print(f"\n{LABELS[sweep]}\n{summary.to_string(index=False)}")

    ax_mi, ax_noise, ax_pool = axes[row]
    for metric, style in (
        ("grouped MI", "s-"), ("KT MI lower", "s-"), ("KT MI upper", "s--"),
        ("counting MI", "^:"), ("identity MI", "o--"),
    ):
        if summary[metric].notna().any():
            ax_mi.plot(summary["genes_per_cell"], summary[metric], style, label=metric)
    ax_mi.set_ylabel(f"{LABELS[sweep]}\nmutual information [bits]", fontsize=8)
    ax_mi.legend(fontsize=8)
    ax_noise.plot(summary["genes_per_cell"], summary["response noise"], "o-")
    ax_noise.set_ylabel("H(response | sniff) [bits]")
    ax_pool.plot(summary["genes_per_cell"], summary["receptor_pool"], "o-")
    ax_pool.set_ylabel("distinct receptors in pool")

for ax in axes.flat:
    ax.set_xlabel("genes expressed per cell")
    ax.grid(axis="y", alpha=.2)
fig.tight_layout()
# fig.savefig(FIGURES / "gene_expression_panels.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
# ── the overlay figure: one curve per selected sweep ────────────────────────
# The count-alphabet ceiling log2(n_states) is drawn alongside whenever the grouped
# estimator ran. Grouping is LOSSLESS for MI, so this is a real ceiling on the array,
# not an estimator artifact — and the KT estimators are bounded by the same number
# even though they see full labelled responses. It is non-monotonic in genes per cell
# because C(G, g) is: for G=5, C=30 complete coverage it runs 14.0, 18.8, 18.6, 13.7,
# 5.0 bits across g=1..5. A curve ON the line is saturating its combinatorics; a curve
# BELOW it is limited by the receptors.
PRIMARY_MI = "grouped MI"   # must match ENTROPY in the sweep scripts
MARKERS = ("o-", "s-", "^-", "v-", "D-", "P-")
COLOURS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, ax = plt.subplots(figsize=(7.5, 5))
for i, sweep in enumerate(SWEEPS):
    summary = SUMMARIES[sweep]
    if not summary[PRIMARY_MI].notna().any():
        print(f"{LABELS[sweep]}: no {PRIMARY_MI} recorded")
        continue
    colour = COLOURS[i % len(COLOURS)]
    ax.plot(summary["genes_per_cell"], summary[PRIMARY_MI],
            MARKERS[i % len(MARKERS)], color=colour, label=LABELS[sweep])
    #if PRIMARY_MI == "grouped MI" and summary["count entropy ceiling"].notna().any():
        #ax.plot(summary["genes_per_cell"], summary["count entropy ceiling"],
        #        ":", color=colour, alpha=.5, label=f"{LABELS[sweep]} — alphabet ceiling")
ax.set_xlabel("genes expressed per cell")
ax.set_ylabel(f"{PRIMARY_MI} [bits]")
# The environment a sweep RAN in, read back from its configs. This is the check that
# matters: the scripts' constants say what the files hold today, these say what the
# data on disk was produced under. Overlaying two regimes is sometimes exactly what
# you want (that is the ablation), so this shouts rather than raises.
REGIMES = {sweep: (summary["mu_ligands"].iloc[0],
                   round(summary["conc_mean_span"].iloc[0], 2))
           for sweep, summary in SUMMARIES.items() if not summary.empty}
regimes = set(REGIMES.values())
if len(regimes) > 1:
    print("\n" + "!" * 78)
    print("SELECTED SWEEPS SPAN MORE THAN ONE ENVIRONMENT. A difference between these")
    print("curves is NOT the heteromer effect or the readout effect on its own.")
    for sweep, (mu, span) in REGIMES.items():
        print(f"  mu={mu:<10g} conc span={span:<6g} {LABELS[sweep]}")
    print("!" * 78 + "\n")
ax.set_title("Information per sniff\n"
             + "; ".join(f"mu={mu:g} ligands/sniff, conc means over {span:g} decades"
                         for mu, span in sorted(regimes)), fontsize=9)
ax.grid(axis="y", alpha=.2)
ax.legend(fontsize=7)
fig.tight_layout()
# fig.savefig(FIGURES / "gene_expression_mi_vs_genes.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
# ── pairwise difference between any two selected sweeps ─────────────────────
# Set to two entries of SWEEPS. A heteromer sweep minus its homomer counterpart is
# the heteromer contribution; a mean sweep minus its threshold counterpart is the
# readout effect. Only points present in BOTH are differenced.
PAIR = SWEEPS[:2]
if len(PAIR) == 2:
    left, right = PAIR
    both = (SUMMARIES[left][["genes_per_cell", PRIMARY_MI]]
            .merge(SUMMARIES[right][["genes_per_cell", PRIMARY_MI]],
                   on="genes_per_cell", suffixes=(" left", " right")))
    both["difference"] = both[f"{PRIMARY_MI} left"] - both[f"{PRIMARY_MI} right"]
    print(f"{PRIMARY_MI} [bits]\n  left  = {LABELS[left]}\n  right = {LABELS[right]}\n")
    print(both.to_string(index=False))

# %%
# ── training trajectories, one column per selected sweep ────────────────────
fig, axes = plt.subplots(2, len(SWEEPS), figsize=(7 * len(SWEEPS), 7),
                         sharex="col", squeeze=False)
for col, sweep in enumerate(SWEEPS):
    for genes_per_cell, cfg, run_dir in RUNS[sweep]:
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
    axes[0, col].set_title(LABELS[sweep], fontsize=9)
    axes[0, col].legend(fontsize=8)
    axes[1, col].legend(fontsize=8)
    axes[1, col].set_xlabel("optimization update")
axes[0, 0].set_ylabel("mutual information [bits]")
axes[1, 0].set_ylabel("H(response | sniff) [bits]")
for ax in axes.flat:
    ax.grid(axis="y", alpha=.2)
fig.tight_layout()
# fig.savefig(FIGURES / "gene_expression_training.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
# ── one latent UMAP per completed run, over every selected sweep ────────────
MODELS = []
for sweep in SWEEPS:
    for genes_per_cell, cfg, run_dir in RUNS[sweep]:
        env, physics, receptor_indices = load_model(run_dir=run_dir)
        gene_homomers = torch.arange(env.n_genes)[:, None].expand(-1, cfg.k_sub)
        embedding = build_latent_umap(env, gene_homomers)
        fig, ax = plt.subplots(figsize=(9, 7))
        plot_latent_umap(env, gene_homomers, ax=ax, embedding=embedding)
        ax.set_title(f"{LABELS[sweep]} — {genes_per_cell} genes/cell")
        fig.tight_layout()
        # fig.savefig(FIGURES / f"latent_umap_{sweep}_{genes_per_cell}.png",
        #             dpi=180, bbox_inches="tight")
        plt.show()
        MODELS.append((sweep, genes_per_cell, cfg, run_dir, env, physics,
                       receptor_indices, embedding))

# %%
# ── one per-cell response UMAP per completed run ────────────────────────────
CONCENTRATION = 1.0
for (sweep, genes_per_cell, cfg, run_dir, env, physics,
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
    fig.suptitle(f"{LABELS[sweep]} — {genes_per_cell} genes/cell\n"
                 f"Single-ligand responses at concentration {CONCENTRATION:g}")
    # fig.savefig(FIGURES / f"response_umap_{sweep}_{genes_per_cell}.png",
    #             dpi=180, bbox_inches="tight")
    plt.show()

# %%
