# %%
"""gene_expression.py — thresholded cells against mean cells, genes/cell 1..5.

SELF-CONTAINED ON PURPOSE. Which folders are read, what each average collapses,
and what goes on every axis is in this file. Nothing is imported from a shared
analysis helper except the loaders and the UMAP drawing code in `src/`.

The data layout it relies on is documented in `doc/data_pipeline.md`.

The only intended difference between the two readouts is the cell:

  mean       A = S            cell fires at the weighted fraction of open receptors
  threshold  A = sigmoid((S - theta) / T)      cell fires above a current

S is the drive, theta one scalar shared by every cell, T the annealed cell
sharpness. The repertoires are byte-identical between the two (same
`cell_sampling_seed`, same coverage design), so the gene sets are paired. The
ENVIRONMENTS are not: the world seed is hashed from the experiment name, so each
sweep draws its own ligand cloud. Treat a small readout gap as unresolved.

What it does, in order:
  1. you choose the sweeps explicitly, and it prints every candidate on disk
  2. one row per RUN     <- first average: the test repeats collapse to one number
  3. one row per POINT   <- second average: replicates collapse to one number
  4. it prints every curve it is about to draw, then draws them
  5. where the threshold actually sat, and how often cells fired
  6. latent UMAP and per-cell response UMAP, one per run
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")
HERE = ROOT / "tasks/cells/gene_expression_threshold/analysis"
FIGURES = HERE.parent / "figures"
assert HERE.is_dir(), HERE
sys.path.insert(0, str(ROOT))

from src.IO import SweepLoader, SingleRunLoader
from src.plotlib import load_model
from src.cells import CellReadout
from src.analysis_helper import (build_latent_umap, plot_latent_umap,
                                 cell_ligand_responses, plot_cell_response_umap)

FIGURES.mkdir(exist_ok=True)
SAVE_FIGURES = False

# The two goals live in different data roots because they are different tasks.
GOALS = {
    "threshold": ROOT / "data" / "gene_expression_threshold",
    "mean":      ROOT / "data" / "gene_expression",
}


# %%
# ════════════════════════════════════════════════════════════════════════════
# 1. WHICH SWEEPS GO INTO THIS FIGURE
# ════════════════════════════════════════════════════════════════════════════
# Named explicitly, (readout, goal, sweep folder). There is no "pick the latest"
# rule: that is what once put a KT-trained sweep on the same axis as three
# exactly-trained ones without anyone noticing.
#
# The mean entries below are replicate 0 of the G=5, C=30 complete-coverage
# sweeps, which used exactly the gene sets this task reproduces.

TODO = "<paste the folder name printed above>"

SWEEPS = [
    ("threshold", "threshold", TODO),   # threshold_heteromers_complete_<timestamp>
    ("threshold", "threshold", TODO),   # threshold_homomers_complete_<timestamp>
    ("mean",      "mean",      "replicates_heteromers_complete_20260923_112537"),
    ("mean",      "mean",      "replicates_homomers_complete_20260923_112624"),
]


def available():
    """Every sweep on disk in both goals, with the design it recorded.

    `experiment.json` is written by the sweep script before training starts and
    records the plan: the flags used, the design points, and every run that was
    supposed to happen.
    """
    rows = []
    for goal, data in GOALS.items():
        for manifest_path in sorted(data.glob("*/experiment.json")):
            manifest = json.loads(manifest_path.read_text())
            args = manifest["arguments"]
            first = manifest["rows"][0]
            rows.append({
                "goal": goal,
                "sweep": manifest_path.parent.name,
                "readout": args.get("cell_readout", "mean"),
                "theta": args.get("cell_threshold", "-"),
                "condition": manifest["condition"],
                "coverage": manifest["coverage"],
                "array": f"G={first['n_genes']}, C={first['n_cells']}",
                "training": args["entropy"],
                "evaluation": args.get("evaluation", "exact"),
                "planned runs": len(manifest["rows"]),
                "used here": "YES" if manifest_path.parent.name in
                             {s for _, _, s in SWEEPS} else "",
            })
    return pd.DataFrame(rows)


catalogue = available()
print("Sweeps available on disk:\n")
print(catalogue.to_string(index=False) if not catalogue.empty else "  (none)")
if any(s == TODO for _, _, s in SWEEPS):
    raise SystemExit(
        "\nSWEEPS still contains a placeholder. Run the sweep, then paste the two "
        "threshold_* folder names from the table above into SWEEPS."
    )


# %%
# ════════════════════════════════════════════════════════════════════════════
# 2. ONE ROW PER RUN  (first average: the test repeats -> one number)
# ════════════════════════════════════════════════════════════════════════════
# Two files per run:
#   config.json        what was simulated, via SweepLoader -> SingleRunConfig
#   test_results.json  what was measured; every value there is a list
#
# Those repeats are measurements of the SAME trained model on fresh input
# batches. They describe evaluation sampling noise. They are NOT replicates, and
# their spread must never be quoted as an uncertainty on a claim.

MI_KEY = "mutual_information_grouped"              # exact enumeration
NOISE_KEY = "conditional_entropy_response_grouped"
TOTAL_KEY = "response_entropy_grouped"

records = []
for readout, goal, sweep_name in SWEEPS:
    sweep_dir = GOALS[goal] / sweep_name
    manifest = json.loads((sweep_dir / "experiment.json").read_text())
    if manifest["arguments"].get("evaluation", "exact") != "exact":
        raise SystemExit(f"{sweep_name} was not measured exactly; this figure assumes it was.")

    # What the sweep was SUPPOSED to produce, keyed by (replicate seed, genes/cell).
    # Comparing against it catches a folder re-run with a different design but kept
    # under its old name, which no plot would ever reveal.
    planned = {(row["cell_sampling_seed"], row["genes_per_cell"]): row
               for row in manifest["rows"]}

    found = 0
    for cfg, run_dir in SweepLoader(str(sweep_dir)).iter_run_dirs():
        results_path = Path(run_dir) / "test_results.json"
        if not results_path.exists():
            continue                      # run crashed, or is still training
        results = json.loads(results_path.read_text())

        sizes = {len(genes) for genes in cfg.cell_gene_sets}
        if len(sizes) != 1:
            raise ValueError(f"Cells express different numbers of genes in {run_dir}")
        genes_per_cell = sizes.pop()

        design = planned.get((cfg.cell_sampling_seed, genes_per_cell))
        if design is None:
            raise ValueError(f"Run is absent from this sweep's experiment.json: {run_dir}")
        if [list(g) for g in design["cell_gene_sets"]] != [list(g) for g in cfg.cell_gene_sets]:
            raise ValueError(f"Gene sets on disk differ from the recorded plan: {run_dir}")

        records.append({
            "readout": readout,
            "condition": manifest["condition"],
            "array": f"G={cfg.n_genes}, C={cfg.n_cells}",
            "genes_per_cell": genes_per_cell,
            "replicate_seed": cfg.cell_sampling_seed,
            #  ↓ THE FIRST AVERAGE
            "mi": float(np.mean(results[MI_KEY])),
            "noise": float(np.mean(results[NOISE_KEY])),
            "total": float(np.mean(results[TOTAL_KEY])),
            "test_repeats": len(results[MI_KEY]),
            "eval_inputs": int(np.mean(results["response_evaluation_samples"])),
            "receptor_pool": len(cfg.receptor_indices),
            "sweep": sweep_name,
            "run_dir": str(run_dir),
        })
        found += 1

    n_planned = len(manifest["rows"])
    note = "" if found == n_planned else "   <-- INCOMPLETE, points below rest on fewer runs"
    print(f"{sweep_name}: {found}/{n_planned} runs complete{note}")

runs = pd.DataFrame(records)
if runs.empty:
    raise SystemExit("No completed runs in the selected sweeps.")
# The mean baselines carry 5 replicates; the threshold sweep carries 1 by default.
# Keep only the seeds the two readouts share, so the comparison stays paired.
shared = set.intersection(*(set(part.replicate_seed)
                            for _, part in runs.groupby("readout")))
dropped = len(runs) - int(runs.replicate_seed.isin(shared).sum())
if dropped:
    print(f"\nDropping {dropped} runs whose replicate seed has no counterpart in the "
          f"other readout. Shared seeds: {sorted(shared)}")
    runs = runs[runs.replicate_seed.isin(shared)]

print(f"\n{len(runs)} runs loaded. Runs per curve:\n")
print(runs.groupby(["readout", "condition", "array"]).size().to_string())


# %%
# ════════════════════════════════════════════════════════════════════════════
# 3. ONE ROW PER POINT  (second average: independent optimizations -> one number)
# ════════════════════════════════════════════════════════════════════════════
# Each replicate is a separate optimization with its own environment and its own
# random cell array (a different `cell_sampling_seed`). Their spread is the real
# uncertainty, and it is what the error bars show. With a single replicate there
# is no error bar, and the difference between two curves is not yet an estimate.

CURVE = ["readout", "condition"]
POINT = CURVE + ["genes_per_cell"]

points = runs.groupby(POINT).agg(
    n=("mi", "size"),
    mi_mean=("mi", "mean"),
    mi_sd=("mi", "std"),
    noise_mean=("noise", "mean"),
    total_mean=("total", "mean"),
    pool=("receptor_pool", "mean"),
    eval_inputs=("eval_inputs", "min"),
).reset_index()
points["mi_sem"] = points.mi_sd / np.sqrt(points.n)

# Retention: MI relative to this same curve's own one-gene-per-cell baseline, so
# a readout offset present at every g cannot be mistaken for an expression effect.
baseline = points[points.genes_per_cell == 1].set_index(CURVE).mi_mean
points["baseline_mi"] = pd.Index(list(map(tuple, points[CURVE].values))).map(baseline)
points["retained"] = points.mi_mean / points.baseline_mi.where(points.baseline_mi > 0)

# An MI within one bit of log2(evaluation inputs) may be capped by the evaluation
# budget rather than by the array. Checked here so it cannot pass unnoticed.
points["sample_ceiling"] = np.log2(points.eval_inputs)
capped = points[points.mi_mean >= points.sample_ceiling - 1]
if not capped.empty:
    print("\nWARNING: within 1 bit of log2(evaluation inputs):")
    print(capped[POINT + ["mi_mean", "sample_ceiling"]].to_string(index=False))


# %%
# ════════════════════════════════════════════════════════════════════════════
# 4. EXACTLY WHAT IS ABOUT TO BE DRAWN
# ════════════════════════════════════════════════════════════════════════════
curves = list(points.groupby(CURVE, sort=False))
print(f"\n{len(curves)} curves:\n")
for (readout, condition), curve in curves:
    curve = curve.sort_values("genes_per_cell")
    print(f"  {readout} | {condition}")
    print(f"      x  genes/cell : {list(curve.genes_per_cell)}")
    print(f"      y  mi_mean    : {[round(v, 3) for v in curve.mi_mean]}")
    print(f"         +- sem     : {[round(v, 3) if np.isfinite(v) else None for v in curve.mi_sem]}")
    print(f"         n runs     : {list(curve.n)}")
    print()

# The number the task exists to produce: threshold minus mean, at matched g.
wide = points.pivot_table(index=["condition", "genes_per_cell"],
                          columns="readout", values="mi_mean")
if {"mean", "threshold"}.issubset(wide.columns):
    wide["threshold - mean"] = wide.threshold - wide["mean"]
    print("MI difference, threshold minus mean [bits]:\n")
    print(wide.round(3).to_string())


# %%
# ════════════════════════════════════════════════════════════════════════════
# 5. THE FIGURE
# ════════════════════════════════════════════════════════════════════════════
# Colour  = biological strategy (what the project is actually asking about)
# Dashing = readout            (the assumption under test)
COLOR = {"heteromers": "tab:blue", "homomers": "tab:orange"}
DASH = {"mean": "--", "threshold": "-"}
MARK = {"mean": "s", "threshold": "o"}


def style(readout, condition):
    return {"color": COLOR.get(condition, "gray"),
            "linestyle": DASH.get(readout, ":"), "marker": MARK.get(readout, "^")}


def draw(ax, column, ylabel, error=None):
    for (readout, condition), curve in points.groupby(CURVE, sort=False):
        curve = curve.sort_values("genes_per_cell")
        kw = style(readout, condition)
        ax.plot(curve.genes_per_cell, curve[column], label=f"{condition}, {readout}", **kw)
        if error is not None and curve[error].notna().any():
            ok = curve[error].notna()
            ax.errorbar(curve.loc[ok, "genes_per_cell"], curve.loc[ok, column],
                        yerr=curve.loc[ok, error], fmt="none",
                        color=kw["color"], capsize=3)
    ax.set_xlabel("genes expressed per cell")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(points.genes_per_cell.unique()))
    ax.grid(axis="y", alpha=.2)


fig, axes = plt.subplots(2, 2, figsize=(13, 9))
draw(axes[0, 0], "mi_mean", "MI [bits]", error="mi_sem")
draw(axes[0, 1], "retained", "MI / this curve's own 1-gene MI")
axes[0, 1].axhline(1, color="gray", linestyle=":", linewidth=1)
draw(axes[1, 0], "noise_mean", "H(response | sniff) [bits]")
draw(axes[1, 1], "total_mean", "H(response) [bits]")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Thresholded cell against mean cell, G=5, C=30, complete coverage.\n"
             "Error bars are SEM across independent optimizations.", fontsize=10)
fig.tight_layout()
if SAVE_FIGURES:
    fig.savefig(FIGURES / "threshold_vs_mean.png", dpi=180, bbox_inches="tight")
plt.show()


# %%
# Every individual optimization, unaveraged. The scatter within one x position is
# the replicate-to-replicate spread the error bars above summarise.
fig, ax = plt.subplots(figsize=(9, 5))
jitter = np.random.default_rng(0)
for (readout, condition), part in runs.groupby(CURVE, sort=False):
    kw = style(readout, condition)
    ax.scatter(part.genes_per_cell + jitter.uniform(-.08, .08, len(part)), part.mi,
               alpha=.65, color=kw["color"], marker=kw["marker"],
               label=f"{condition}, {readout}")
ax.set(xlabel="genes expressed per cell", ylabel="MI per optimization [bits]")
ax.set_xticks(sorted(runs.genes_per_cell.unique()))
ax.grid(axis="y", alpha=.2)
ax.legend(fontsize=8)
fig.tight_layout()
if SAVE_FIGURES:
    fig.savefig(FIGURES / "threshold_vs_mean_runs.png", dpi=180, bbox_inches="tight")
plt.show()


# %%
# ════════════════════════════════════════════════════════════════════════════
# 6. WHERE THE THRESHOLD ACTUALLY SAT
# ════════════════════════════════════════════════════════════════════════════
# theta is pinned to the median of the drive, so it is a property of the trained
# chemistry rather than a fitted parameter. Reading it back is the only way to
# see whether the cell ended up operating where it was supposed to.
#
# P(fire) is read from the entropy, not re-simulated: the array's total response
# entropy divided by the number of cells is an upper bound on the per-cell output
# entropy, and a cell that never leaves one state contributes zero.
thetas = []
for _, run in runs[runs.readout == "threshold"].iterrows():
    checkpoint = SingleRunLoader(run.run_dir).load_checkpoint(map_location="cpu")
    state = checkpoint["readout_state"]
    thetas.append({
        "condition": run.condition,
        "genes_per_cell": run.genes_per_cell,
        "theta": float(state["theta"]) if "theta" in state else np.nan,
        "T_cell": float(checkpoint["readout_temperature"]),
        "H(Y)/cell": run.total / 30,
        "H(Y|X)/cell": run.noise / 30,
    })
if thetas:
    print("Threshold placement, one row per run:\n")
    print(pd.DataFrame(thetas).sort_values(["condition", "genes_per_cell"])
          .to_string(index=False))
    # 1/g atoms: with uniform homomer abundances the drive can only take the
    # values k/g, so theta must land in a gap. Printed next to the atoms it sits
    # between, because a theta exactly on an atom is the degenerate case.
    for row in thetas:
        g = row["genes_per_cell"]
        atoms = np.arange(g + 1) / g
        gap = np.searchsorted(atoms, row["theta"])
        print(f"  g={g} {row['condition']}: theta={row['theta']:.4f} sits between "
              f"{atoms[max(gap - 1, 0)]:.3f} and {atoms[min(gap, g)]:.3f}")


# %%
# ════════════════════════════════════════════════════════════════════════════
# 7. LATENT UMAP, ONE PER RUN
# ════════════════════════════════════════════════════════════════════════════
# The chemical landscape the optimizer settled on: ligands in latent space,
# coloured by how strongly each gene's homomer responds. The embedding is built
# once per run and reused for the per-cell panels below, so the two figures of a
# run share coordinates.
MODELS = []
for _, run in runs.sort_values(["readout", "condition", "genes_per_cell"]).iterrows():
    cfg = SingleRunLoader(run.run_dir).load_config()
    env, physics, receptor_indices = load_model(run_dir=run.run_dir)
    gene_homomers = torch.arange(env.n_genes)[:, None].expand(-1, cfg.k_sub)
    embedding = build_latent_umap(env, gene_homomers)
    fig, ax = plt.subplots(figsize=(9, 7))
    plot_latent_umap(env, gene_homomers, ax=ax, embedding=embedding)
    ax.set_title(f"{run.readout} — {run.condition} — {run.genes_per_cell} genes/cell")
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(FIGURES / f"latent_umap_{run.readout}_{run.condition}_"
                              f"{run.genes_per_cell}.png", dpi=180, bbox_inches="tight")
    plt.show()
    MODELS.append((run, cfg, env, physics, receptor_indices, embedding))


# %%
# ════════════════════════════════════════════════════════════════════════════
# 8. PER-CELL RESPONSE UMAP, ONE PANEL PER CELL
# ════════════════════════════════════════════════════════════════════════════
# Each panel is one cell's firing probability over the same chemical landscape,
# at a single ligand presented at fixed concentration. This is where the two
# readouts should look visibly different: a mean cell shades, a thresholded cell
# at the annealed sharpness is close to a hard partition of the landscape.
CONCENTRATION = 1.0
for run, cfg, env, physics, receptor_indices, embedding in MODELS:
    checkpoint = SingleRunLoader(run.run_dir).load_checkpoint(map_location="cpu")
    readout = CellReadout(
        checkpoint["readout_state"]["W"], mode=checkpoint["readout_mode"],
        temperature=checkpoint["readout_temperature"], k_sub=cfg.k_sub,
        learnable_threshold=cfg.cell_threshold_learnable,
    )
    readout.load_state_dict(checkpoint["readout_state"])
    readout.eval()
    responses = cell_ligand_responses(env, physics, receptor_indices, readout,
                                      CONCENTRATION)
    fig, axes = plot_cell_response_umap(embedding, responses, cfg.cell_gene_sets,
                                        concentration=CONCENTRATION)
    fig.suptitle(f"{run.readout} — {run.condition} — {run.genes_per_cell} genes/cell\n"
                 f"Single-ligand responses at concentration {CONCENTRATION:g}")
    if SAVE_FIGURES:
        fig.savefig(FIGURES / f"response_umap_{run.readout}_{run.condition}_"
                              f"{run.genes_per_cell}.png", dpi=180, bbox_inches="tight")
    plt.show()

# %%
