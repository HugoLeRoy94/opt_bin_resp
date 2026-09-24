# %%
"""expression_law.py — MI versus the MEAN number of genes per cell.

SELF-CONTAINED ON PURPOSE, like tasks/cells/gene_expression/analysis/replicates.py.
Everything from "which folders am I reading" to "what is on each axis" is here.

The question: the gene_expression task found MI peaking at exactly 2 genes per
cell, and raising the cell count did not move it. The combinatorial reading is
that the number of distinct gene sets available to a cell is C(G, g), maximal at
g = G/2, so the peak should be set by G and not by C. This sweeps G and the mean
genes per cell together to test that.

The x axis is the REALISED mean genes per cell, measured from the arrays that were
actually drawn, not the target that was requested. Independent sampling makes the
two differ.

Read doc/data_pipeline.md for the data layout this relies on.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.IO import SweepLoader

DATA = ROOT / "data" / "expression_law"
HERE = Path(__file__).resolve().parent
SAVE_FIGURES = False


# %%
# ════════════════════════════════════════════════════════════════════════════
# 1. WHICH SWEEPS
# ════════════════════════════════════════════════════════════════════════════
# Named explicitly. One curve is drawn per (condition, G, size law, gene law).
SWEEPS = [
    "expression_law_heteromers_uniform_uniform_PLACEHOLDER",
    "expression_law_homomers_uniform_uniform_PLACEHOLDER",
]


def available():
    """Every expression_law sweep on disk, with the laws it used."""
    rows = []
    for path in sorted(DATA.glob("expression_law_*/experiment.json")):
        m = json.loads(path.read_text())
        rows.append({
            "sweep": path.parent.name,
            "condition": m["condition"],
            "size law": m["size_family"],
            "gene law": m["gene_family"],
            "gene ratio": m["gene_ratio"],
            "G": sorted({r["n_genes"] for r in m["rows"]}),
            "planned runs": len(m["rows"]),
            "used here": "YES" if path.parent.name in SWEEPS else "",
        })
    return pd.DataFrame(rows)


if not DATA.exists():
    raise SystemExit(f"No data yet: {DATA} does not exist. Run scripts/expression_law.py first.")
print("Sweeps available on disk:\n")
print(available().to_string(index=False) if len(available()) else "  (none)")


# %%
# ════════════════════════════════════════════════════════════════════════════
# 2. ONE ROW PER RUN  (first average: 10 test repeats -> 1 number)
# ════════════════════════════════════════════════════════════════════════════
# These sweeps use sampled counting, so the MI key carries the estimator name.
# The plug-in estimator is biased DOWNWARD when the symbol space is undersampled,
# which is why grouped_counting_unique_fraction is carried through to step 4.
MI_KEY = "mutual_information_grouped_counting_plugin"
NOISE_KEY = "conditional_entropy_response_grouped_counting"

records = []
for sweep_name in SWEEPS:
    sweep_dir = DATA / sweep_name
    if not sweep_dir.is_dir():
        print(f"skip {sweep_name}: not on disk")
        continue
    manifest = json.loads((sweep_dir / "experiment.json").read_text())
    planned = {(r["cell_sampling_seed"]): r for r in manifest["rows"]}

    found = 0
    for cfg, run_dir in SweepLoader(str(sweep_dir)).iter_run_dirs():
        results_path = Path(run_dir) / "test_results.json"
        if not results_path.exists():
            continue
        results = json.loads(results_path.read_text())
        design = planned.get(cfg.cell_sampling_seed)
        if design is None:
            raise ValueError(f"Run absent from this sweep's experiment.json: {run_dir}")

        sizes = [len(genes) for genes in cfg.cell_gene_sets]
        records.append({
            "condition": manifest["condition"],
            "G": cfg.n_genes,
            "C": cfg.n_cells,
            "size_law": manifest["size_family"],
            "gene_law": manifest["gene_family"],
            "target_mean": design["target_mean_genes"],
            # measured from the array actually drawn, not the target requested
            "mean_genes": float(np.mean(sizes)),
            "replicate": design["replicate"],
            #  ↓ THE FIRST AVERAGE: 10 measurements of ONE trained model
            "mi": float(np.mean(results[MI_KEY])),
            "noise": float(np.mean(results[NOISE_KEY])),
            "unique_fraction": float(np.mean(results["grouped_counting_unique_fraction"])),
            "eval_inputs": int(np.mean(results["response_evaluation_samples"])),
            "cell_types": len({tuple(g) for g in cfg.cell_gene_sets}),
            "genes_seen": len({g for cell in cfg.cell_gene_sets for g in cell}),
            "pool": len(cfg.receptor_indices),
            "run_dir": str(run_dir),
        })
        found += 1
    print(f"{sweep_name}: {found}/{len(manifest['rows'])} runs complete")

runs = pd.DataFrame(records)
if runs.empty:
    raise SystemExit("No completed runs in the selected sweeps.")


# %%
# ════════════════════════════════════════════════════════════════════════════
# 3. ONE ROW PER POINT  (second average: independent optimizations -> 1 number)
# ════════════════════════════════════════════════════════════════════════════
CURVE = ["condition", "G", "size_law", "gene_law"]
POINT = CURVE + ["target_mean"]

points = runs.groupby(POINT).agg(
    n=("mi", "size"), mi_mean=("mi", "mean"), mi_sd=("mi", "std"),
    mean_genes=("mean_genes", "mean"), noise_mean=("noise", "mean"),
    cell_types=("cell_types", "mean"), genes_seen=("genes_seen", "mean"),
    pool=("pool", "mean"), unique_fraction=("unique_fraction", "mean"),
    eval_inputs=("eval_inputs", "min"),
).reset_index()
points["mi_sem"] = points.mi_sd / np.sqrt(points.n)

# The combinatorial ceiling this experiment is testing: with G genes and g of them
# expressed, a cell can have one of C(G, g) gene sets, so no more than
# log2(C(G, g)) bits can come from WHICH cell is which.
points["types_available"] = [math.comb(int(G), int(round(m)))
                             for G, m in zip(points.G, points.mean_genes)]
points["log2_types"] = np.log2(points.types_available)

# Sampled counting saturates when almost every evaluation input gives a symbol seen
# once. Then the plug-in entropy is an artefact of the budget, not of the array.
suspect = points[(points.unique_fraction > 0.5) |
                 (points.mi_mean >= np.log2(points.eval_inputs) - 1)]
if not suspect.empty:
    print("\nWARNING: undersampled counting or MI near log2(evaluation inputs).")
    print("These points measure the evaluation budget as much as the array:")
    print(suspect[POINT + ["mi_mean", "unique_fraction", "eval_inputs"]].to_string(index=False))


# %%
# ════════════════════════════════════════════════════════════════════════════
# 4. EXACTLY WHAT IS ABOUT TO BE DRAWN
# ════════════════════════════════════════════════════════════════════════════
curves = list(points.groupby(CURVE, sort=False))
print(f"\n{len(curves)} curves:\n")
for (condition, G, size_law, gene_law), curve in curves:
    curve = curve.sort_values("mean_genes")
    peak = curve.loc[curve.mi_mean.idxmax()]
    print(f"  {condition} | G={G} | size {size_law} | gene {gene_law}")
    print(f"      x  mean genes/cell : {[round(v, 2) for v in curve.mean_genes]}")
    print(f"      y  mi_mean         : {[round(v, 3) for v in curve.mi_mean]}")
    print(f"         +- sem          : {[round(v, 3) for v in curve.mi_sem]}")
    print(f"         n runs          : {list(curve.n)}")
    print(f"      PEAK at {peak.mean_genes:.2f} genes/cell "
          f"(combinatorial maximum of C({G}, g) sits at g={G / 2:.1f})")
    print()


# %%
# ════════════════════════════════════════════════════════════════════════════
# 5. THE FIGURE
# ════════════════════════════════════════════════════════════════════════════
COLOR = {"heteromers": "tab:blue", "homomers": "tab:orange"}
MARK = {3: "o", 5: "s", 8: "^"}
DASH = {"uniform": "-", "exponential": "--"}


def style(condition, G, size_law, gene_law):
    return {"color": COLOR.get(condition, "gray"),
            "marker": MARK.get(int(G), "D"),
            "linestyle": DASH.get(size_law, ":")}


def draw(ax, column, ylabel, error=None):
    for key, curve in points.groupby(CURVE, sort=False):
        condition, G, size_law, gene_law = key
        curve = curve.sort_values("mean_genes")
        kw = style(*key)
        ax.plot(curve.mean_genes, curve[column],
                label=f"{condition}, G={G}, {size_law}/{gene_law}", **kw)
        if error is not None:
            ok = curve[error].notna()
            ax.errorbar(curve.loc[ok, "mean_genes"], curve.loc[ok, column],
                        yerr=curve.loc[ok, error], fmt="none", color=kw["color"], capsize=3)
    ax.set_xlabel("mean genes expressed per cell (realised)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.2)


fig, axes = plt.subplots(2, 2, figsize=(13, 9))
draw(axes[0, 0], "mi_mean", "MI [bits], mean over replicates", error="mi_sem")
draw(axes[0, 1], "log2_types", "log2 C(G, g): bits available from cell identity")
draw(axes[1, 0], "noise_mean", "H(response | input) [bits]")
draw(axes[1, 1], "cell_types", "distinct gene sets realised in the array")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Does the MI peak move with G? The combinatorial ceiling (top right) "
             "says it should sit near g = G/2.", fontsize=10)
fig.tight_layout()
if SAVE_FIGURES:
    fig.savefig(HERE / "expression_law.png", dpi=180)
plt.show()


# %%
# Peak position against G: the actual test. If the peak is combinatorial it
# follows G/2; if it is a property of the physics it stays put.
fig, ax = plt.subplots(figsize=(7, 5))
for (condition, size_law, gene_law), part in points.groupby(
        ["condition", "size_law", "gene_law"], sort=False):
    peaks = part.loc[part.groupby("G").mi_mean.idxmax()].sort_values("G")
    ax.plot(peaks.G, peaks.mean_genes, "o-", color=COLOR.get(condition, "gray"),
            linestyle=DASH.get(size_law, ":"),
            label=f"{condition}, {size_law}/{gene_law}")
grid = np.array(sorted(points.G.unique()), dtype=float)
ax.plot(grid, grid / 2, "k:", label="combinatorial prediction, g = G/2")
ax.set(xlabel="G, genes in the pool", ylabel="mean genes/cell at peak MI")
ax.legend(fontsize=8)
ax.grid(alpha=.2)
fig.tight_layout()
if SAVE_FIGURES:
    fig.savefig(HERE / "expression_law_peak.png", dpi=180)
plt.show()
