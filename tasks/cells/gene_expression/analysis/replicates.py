# %%
"""replicates.py — mutual information versus genes expressed per cell.

SELF-CONTAINED ON PURPOSE. Everything from "which folders am I reading" to "what
is on each axis" is in this file. Nothing is imported from a shared analysis
helper, so changing what this figure shows cannot change any other figure.

The data layout it relies on is documented in `doc/data_pipeline.md`.

What it does, in order:
  1. you choose the sweeps explicitly, and it prints every candidate on disk
  2. one row per RUN        <- first average: 10 test repeats collapse to 1 number
  3. one row per POINT      <- second average: N replicates collapse to 1 number
  4. it prints every curve it is about to draw, then draws them

It refuses to plot runs that do not match the plan recorded in experiment.json,
and it reports how many of the planned runs actually finished. It does NOT check
that the selected sweeps used the same epochs, batch sizes or environment: pooling
sweeps that differ in those is your call, and the printout in step 1 is where you
make it.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.IO import SweepLoader   # crawls a sweep, yields (SingleRunConfig, run_dir)

DATA = ROOT / "data" / "gene_expression"
HERE = Path(__file__).resolve().parent
SAVE_FIGURES = False


# %%
# ════════════════════════════════════════════════════════════════════════════
# 1. WHICH SWEEPS GO INTO THIS FIGURE
# ════════════════════════════════════════════════════════════════════════════
# Named explicitly. There is no "pick the latest" rule, because that is exactly
# what once put a KT-trained sweep on the same axis as three exactly-trained ones
# without anyone noticing. If you want a different comparison, edit this list.
#
# One curve is drawn per (condition, array, method) combination present.

SWEEPS = [
    # 30-cell arrays, exact count enumeration, both biological strategies
    "replicates_heteromers_complete_20260923_112537",
    "replicates_homomers_complete_20260923_112624",
    # the same 30-cell array trained with KT and measured by sampled counting.
    # Heteromers only: there is no homomer counterpart for this method.
    "replicates_heteromers_complete_20260923_135950",
    # Uncomment for the 10-cell random-coverage arrays as well (5 curves total).
    # These are a DIFFERENT array (G=3, C=10), not more replicates of the above.
    "replicates_heteromers_random_20260922_084038",
    "replicates_homomers_random_20260922_084103",
]


def available(experiment="replicates"):
    """Every sweep on disk for this experiment, with the method it used.

    `experiment.json` is written by scripts/<experiment>.py before the sweep runs
    and records the plan: the flags used, the design points, and every run that
    was supposed to happen.
    """
    rows = []
    for manifest_path in sorted(DATA.glob(f"{experiment}_*/experiment.json")):
        manifest = json.loads(manifest_path.read_text())
        first = manifest["rows"][0]
        rows.append({
            "sweep": manifest_path.parent.name,
            "condition": manifest["condition"],
            "coverage": manifest["coverage"],
            "training": manifest["arguments"]["entropy"],
            "evaluation": manifest["arguments"].get("evaluation", "exact"),
            "array": f"G={first['n_genes']}, C={first['n_cells']}",
            "planned runs": len(manifest["rows"]),
            "used here": "YES" if manifest_path.parent.name in SWEEPS else "",
        })
    return pd.DataFrame(rows)


print("Sweeps available on disk:\n")
print(available().to_string(index=False))


# %%
# ════════════════════════════════════════════════════════════════════════════
# 2. ONE ROW PER RUN  (first average: 10 test repeats -> 1 number)
# ════════════════════════════════════════════════════════════════════════════
# Two files are read per run:
#   config.json        what was simulated, via SweepLoader -> SingleRunConfig
#   test_results.json  what was measured. EVERY value there is a list of 10.
#
# Those 10 are ten measurements of the SAME trained model, each on a fresh input
# batch. They describe evaluation sampling noise. They are NOT ten replicates,
# and their spread must never be quoted as an uncertainty on a claim.

records = []
for sweep_name in SWEEPS:
    sweep_dir = DATA / sweep_name
    manifest = json.loads((sweep_dir / "experiment.json").read_text())
    training = manifest["arguments"]["entropy"]
    evaluation = manifest["arguments"].get("evaluation", "exact")

    # Which key holds the MI depends on how the run was finally measured.
    # exact    : the grouped estimator enumerated every joint count state
    # counting : group counts were sampled and the entropy estimated by frequency
    if evaluation == "counting":
        mi_key = "mutual_information_grouped_counting_plugin"
        noise_key = "conditional_entropy_response_grouped_counting"
    else:
        mi_key = "mutual_information_grouped"
        noise_key = "conditional_entropy_response_grouped"

    # What the sweep was SUPPOSED to produce, keyed by (replicate seed, genes/cell).
    # Comparing against it catches a folder that was re-run with a different design
    # but kept its old name, which no plot would ever reveal.
    planned = {(row["cell_sampling_seed"], row["genes_per_cell"]): row
               for row in manifest["rows"]}

    found = 0
    for cfg, run_dir in SweepLoader(str(sweep_dir)).iter_run_dirs():
        results_path = Path(run_dir) / "test_results.json"
        if not results_path.exists():
            continue                      # run crashed, or is still training
        results = json.loads(results_path.read_text())

        # Genes expressed per cell, taken from what was actually simulated rather
        # than from the directory name.
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
            "condition": manifest["condition"],
            "array": f"G={cfg.n_genes}, C={cfg.n_cells}",
            "method": f"{training} / {evaluation}",
            "genes_per_cell": genes_per_cell,
            "replicate_seed": cfg.cell_sampling_seed,
            #  ↓ THE FIRST AVERAGE
            "mi": float(np.mean(results[mi_key])),
            "noise": float(np.mean(results[noise_key])),
            "test_repeats": len(results[mi_key]),
            # Good-Turing f1/B: probability mass never sampled. Only the counting
            # estimator can be undersampled, and only newer runs record it.
            "missing_mass": float(np.mean(results["grouped_counting_missing_mass"]))
            if "grouped_counting_missing_mass" in results else np.nan,
            "eval_inputs": int(np.mean(results["response_evaluation_samples"])),
            "genes_represented": len({g for cell in cfg.cell_gene_sets for g in cell}),
            "sweep": sweep_name,
            "run_dir": str(run_dir),
        })
        found += 1

    planned = len(manifest["rows"])
    note = "" if found == planned else "   <-- INCOMPLETE, points below rest on fewer runs"
    print(f"{sweep_name}: {found}/{planned} runs complete{note}")

runs = pd.DataFrame(records)
if runs.empty:
    raise SystemExit("No completed runs in the selected sweeps.")

print(f"\n{len(runs)} runs loaded. Runs per curve:\n")
print(runs.groupby(["condition", "array", "method"]).size().to_string())


# %%
# ════════════════════════════════════════════════════════════════════════════
# 3. ONE ROW PER POINT  (second average: independent optimizations -> 1 number)
# ════════════════════════════════════════════════════════════════════════════
# Each replicate is a separate optimization, with its own environment and its own
# random cell array (a different `cell_sampling_seed`). Their spread is the real
# uncertainty, and it is what the error bars show.

CURVE = ["condition", "array", "method"]   # one line per combination of these
POINT = CURVE + ["genes_per_cell"]         # one marker per combination of these

points = runs.groupby(POINT).agg(
    n=("mi", "size"),
    mi_mean=("mi", "mean"),
    mi_sd=("mi", "std"),
    noise_mean=("noise", "mean"),
    genes_repr=("genes_represented", "mean"),
    missing_mass=("missing_mass", "mean"),
    eval_inputs=("eval_inputs", "min"),
).reset_index()
points["mi_sem"] = points.mi_sd / np.sqrt(points.n)

# Retention: MI relative to this same curve's own one-gene-per-cell baseline.
# Each curve gets its own baseline, so a method or array offset cannot leak in.
baseline = points[points.genes_per_cell == 1].set_index(CURVE).mi_mean
points["baseline_mi"] = pd.Index(list(map(tuple, points[CURVE].values))).map(baseline)
points["retained"] = points.mi_mean / points.baseline_mi.where(points.baseline_mi > 0)

# An MI within one bit of log2(evaluation inputs) may be capped by the evaluation
# budget rather than by the array. Checked here so it cannot pass unnoticed.
points["sample_ceiling"] = np.log2(points.eval_inputs)
capped = points[points.mi_mean >= points.sample_ceiling - 1]
if not capped.empty:
    print("\nWARNING: within 1 bit of log2(evaluation inputs); run evaluation_budget.py:")
    print(capped[POINT + ["mi_mean", "sample_ceiling"]].to_string(index=False))


# %%
# ════════════════════════════════════════════════════════════════════════════
# 4. EXACTLY WHAT IS ABOUT TO BE DRAWN
# ════════════════════════════════════════════════════════════════════════════
curves = list(points.groupby(CURVE, sort=False))
print(f"\n{len(curves)} curves:\n")
for (condition, array, method), curve in curves:
    curve = curve.sort_values("genes_per_cell")
    print(f"  {condition} | {array} | {method}")
    print(f"      x  genes/cell : {list(curve.genes_per_cell)}")
    print(f"      y  mi_mean    : {[round(v, 3) for v in curve.mi_mean]}")
    print(f"         +- sem     : {[round(v, 3) for v in curve.mi_sem]}")
    print(f"         n runs     : {list(curve.n)}")
    if curve.baseline_mi.isna().all():
        print("         retention  : no 1-gene baseline in this curve, left blank")
    print()


# %%
# ════════════════════════════════════════════════════════════════════════════
# 5. THE FIGURE
# ════════════════════════════════════════════════════════════════════════════
# Colour  = biological strategy (what the project is actually asking about)
# Dashing = estimator method    (an artefact of how it was measured, not biology)
# Marker  = array size
COLOR = {"heteromers": "tab:blue", "homomers": "tab:orange"}
DASH = ["-", "--", ":", "-."]
MARK = ["o", "s", "^", "D"]
methods = sorted(points.method.unique())
arrays = sorted(points.array.unique())


def style(condition, array, method):
    return {
        "color": COLOR.get(condition, "gray"),
        "linestyle": DASH[methods.index(method) % len(DASH)],
        "marker": MARK[arrays.index(array) % len(MARK)],
    }


def draw(ax, column, ylabel, error=None):
    for (condition, array, method), curve in points.groupby(CURVE, sort=False):
        curve = curve.sort_values("genes_per_cell")
        label = f"{condition}, {array}, {method}"
        kw = style(condition, array, method)
        ax.plot(curve.genes_per_cell, curve[column], label=label, **kw)
        if error is not None:
            ok = curve[error].notna()
            ax.errorbar(curve.loc[ok, "genes_per_cell"], curve.loc[ok, column],
                        yerr=curve.loc[ok, error], fmt="none",
                        color=kw["color"], capsize=3)
    ax.set_xlabel("genes expressed per cell")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(points.genes_per_cell.unique()))
    ax.grid(axis="y", alpha=.2)


fig, axes = plt.subplots(2, 2, figsize=(13, 9))
draw(axes[0, 0], "mi_mean", "MI [bits], mean over replicates", error="mi_sem")
draw(axes[0, 1], "retained", "MI / this curve's own 1-gene MI")
axes[0, 1].axhline(1, color="gray", linestyle=":", linewidth=1)
draw(axes[1, 0], "noise_mean", "H(response | input) [bits]")
draw(axes[1, 1], "genes_repr", "distinct genes present in the array")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Error bars are SEM across independent optimizations, "
             "not across the 10 test repeats of one model", fontsize=10)
fig.tight_layout()
if SAVE_FIGURES:
    fig.savefig(HERE / "replicates.png", dpi=180)
plt.show()


# %%
# Every individual optimization, unaveraged. The scatter within one x position is
# the replicate-to-replicate spread that the error bars above summarise.
fig, ax = plt.subplots(figsize=(9, 5))
for (condition, array, method), part in runs.groupby(CURVE, sort=False):
    kw = style(condition, array, method)
    ax.scatter(part.genes_per_cell + np.random.default_rng(0).uniform(-.08, .08, len(part)),
               part.mi, alpha=.65, color=kw["color"], marker=kw["marker"],
               label=f"{condition}, {array}, {method}")
ax.set(xlabel="genes expressed per cell", ylabel="MI per optimization [bits]")
ax.set_xticks(sorted(runs.genes_per_cell.unique()))
ax.grid(axis="y", alpha=.2)
ax.legend(fontsize=8)
fig.tight_layout()
if SAVE_FIGURES:
    fig.savefig(HERE / "replicates_runs.png", dpi=180)
plt.show()


# %%
# How much of the response distribution the evaluation never sampled, for the
# curves measured by sampled counting. The plug-in estimator under-reads entropy
# in proportion to this, so an MI point at a high missing mass is partly a
# measurement of the evaluation budget rather than of the array. The exact
# estimator cannot be undersampled and is absent here by construction.
if points.missing_mass.notna().any():
    fig, ax = plt.subplots(figsize=(8, 5))
    for (condition, array, method), curve in points.groupby(CURVE, sort=False):
        curve = curve.sort_values("genes_per_cell")
        if curve.missing_mass.isna().all():
            continue
        kw = style(condition, array, method)
        ax.plot(curve.genes_per_cell, curve.missing_mass,
                label=f"{condition}, {array}, {method}", **kw)
    ax.set(xlabel="genes expressed per cell",
           ylabel="Good-Turing missing mass  f1/B")
    ax.set_xticks(sorted(points.genes_per_cell.unique()))
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=8)
    fig.suptitle(f"Unsampled mass at {int(runs.eval_inputs.min()):,} evaluation inputs",
                 fontsize=10)
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(HERE / "replicates_missing_mass.png", dpi=180)
    plt.show()
else:
    print("No missing-mass metric in these runs: they predate "
          "grouped_counting_missing_mass, or were measured by exact enumeration.")

# %%
