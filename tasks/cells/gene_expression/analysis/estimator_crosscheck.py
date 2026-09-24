# %%
"""estimator_crosscheck.py — is the exact/KT gap the optimizer or the estimator?

SELF-CONTAINED ON PURPOSE. Reads the JSON written by
scripts/estimator_crosscheck.py, which re-measured both sweeps' saved checkpoints
with both estimators.

The 2x2 per design point:

                        eval exact        eval counting
    train exact           A                     B
    train KT              C                     D

    optimization loss = A - C   what training on the KT bound gives up.
                                The exact estimator is unbiased, so this is a
                                clean comparison of two trained models.
    estimator bias    = A - B   what sampled counting loses at this budget,
                                measured on a model the exact estimator can read.
    reported gap      = A - D   the number the two sweeps print, which is the
                                sum of the two above plus their interaction.

If optimization loss is ~0 and estimator bias carries the whole gap, then raising
the TRAINING batch size will not help and raising the EVALUATION budget will.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "gene_expression"
HERE = Path(__file__).resolve().parent
SAVE_FIGURES = False

# %%
# ════════════════════════════════════════════════════════════════════════════
# 1. WHICH REPORT
# ════════════════════════════════════════════════════════════════════════════
# None takes the most recent. Set a path to read a specific one.
REPORT = None

candidates = sorted(DATA.glob("*/estimator_crosscheck_*.json"))
if not candidates:
    raise SystemExit("No crosscheck report found. Run scripts/estimator_crosscheck.py first.")
print("Reports on disk:")
for path in candidates:
    print(f"  {path.relative_to(DATA)}")
path = Path(REPORT) if REPORT else candidates[-1]
report = json.loads(path.read_text())
runs = pd.DataFrame(report["records"])
print(f"\nUsing {path.name}")
print(f"  exact-trained sweep: {Path(report['exact_sweep']).name}")
print(f"  KT-trained sweep   : {Path(report['kt_sweep']).name}")
print(f"  {len(runs)} measurements, budgets {sorted(runs.samples.unique())}, "
      f"expression levels {sorted(runs.genes_per_cell.unique())}")


# %%
# ════════════════════════════════════════════════════════════════════════════
# 2. THE 2x2, AVERAGED OVER REPLICATES AND REPEATS
# ════════════════════════════════════════════════════════════════════════════
cell = runs.groupby(["genes_per_cell", "samples", "trained_on", "evaluated_with"]).agg(
    mi=("mi", "mean"), sd=("mi", "std"), n=("mi", "size")).reset_index()
grid = cell.pivot_table(index=["genes_per_cell", "samples"],
                        columns=["trained_on", "evaluated_with"], values="mi")

table = pd.DataFrame(index=grid.index)
table["A train_exact/eval_exact"] = grid.get(("exact", "exact"))
table["B train_exact/eval_count"] = grid.get(("exact", "counting"))
table["C train_KT/eval_exact"] = grid.get(("KT", "exact"))
table["D train_KT/eval_count"] = grid.get(("KT", "counting"))
table["optimization loss A-C"] = table["A train_exact/eval_exact"] - table["C train_KT/eval_exact"]
table["estimator bias A-B"] = table["A train_exact/eval_exact"] - table["B train_exact/eval_count"]
table["reported gap A-D"] = table["A train_exact/eval_exact"] - table["D train_KT/eval_count"]

print("\nAll values in bits of mutual information:\n")
print(table.to_string(float_format=lambda v: f"{v:8.3f}"))


# %%
# ════════════════════════════════════════════════════════════════════════════
# 3. THE VERDICT
# ════════════════════════════════════════════════════════════════════════════
opt = table["optimization loss A-C"].abs()
bias = table["estimator bias A-B"].abs()
print(f"\noptimization loss |A-C| : median {opt.median():.3f}, max {opt.max():.3f} bits")
print(f"estimator bias    |A-B| : median {bias.median():.3f}, max {bias.max():.3f} bits")
if bias.median() > 5 * max(opt.median(), 1e-9):
    print("\n-> The estimator carries the gap. Training on the KT bound costs almost")
    print("   nothing; the reported difference is sampled counting under-reading the")
    print("   entropy at this budget. Raise the EVALUATION budget, not the training one.")
elif opt.median() > 5 * max(bias.median(), 1e-9):
    print("\n-> The optimizer carries the gap: the KT-trained models really are worse,")
    print("   and the estimator reads them fairly. Raise the TRAINING batch size.")
else:
    print("\n-> Both contribute at comparable size; neither knob alone will close the gap.")

# Does more evaluation budget actually help? The exact columns are unbiased, so any
# movement there is sampling noise and sets the scale for reading the counting ones.
print("\nMovement with evaluation budget (bits), per expression level:")
for g, part in table.groupby(level="genes_per_cell"):
    if len(part) < 2:
        continue
    lo, hi = part.index.get_level_values("samples").min(), part.index.get_level_values("samples").max()
    d_exact = part.loc[(g, hi), "A train_exact/eval_exact"] - part.loc[(g, lo), "A train_exact/eval_exact"]
    d_count = part.loc[(g, hi), "B train_exact/eval_count"] - part.loc[(g, lo), "B train_exact/eval_count"]
    print(f"  g={g}: B {lo:,} -> {hi:,}   exact {d_exact:+.3f} (noise floor)   "
          f"counting {d_count:+.3f}")


# %%
# ════════════════════════════════════════════════════════════════════════════
# 4. FIGURE
# ════════════════════════════════════════════════════════════════════════════
levels = sorted(runs.genes_per_cell.unique())
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

STYLE = {("exact", "exact"): ("tab:blue", "-", "o"),
         ("exact", "counting"): ("tab:blue", "--", "s"),
         ("KT", "exact"): ("tab:red", "-", "^"),
         ("KT", "counting"): ("tab:red", "--", "D")}
biggest = max(levels, key=lambda g: table.loc[g, "estimator bias A-B"].abs().max())
part = cell[cell.genes_per_cell == biggest]
for (trained, evaluated), curve in part.groupby(["trained_on", "evaluated_with"]):
    color, dash, mark = STYLE[(trained, evaluated)]
    curve = curve.sort_values("samples")
    axes[0].plot(curve.samples, curve.mi, color=color, linestyle=dash, marker=mark,
                 label=f"train {trained} / eval {evaluated}")
axes[0].set(xscale="log", xlabel="evaluation inputs",
            ylabel="MI [bits]", title=f"g={biggest} genes/cell (largest bias)")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=.2)

width = 0.35
x = np.arange(len(levels))
lo = table.index.get_level_values("samples").min()
axes[1].bar(x - width / 2, [table.loc[(g, lo), "optimization loss A-C"] for g in levels],
            width, label="optimization loss (A-C)", color="tab:green")
axes[1].bar(x + width / 2, [table.loc[(g, lo), "estimator bias A-B"] for g in levels],
            width, label="estimator bias (A-B)", color="tab:purple")
axes[1].axhline(0, color="k", linewidth=.8)
axes[1].set(xticks=x, xlabel="genes expressed per cell", ylabel="bits",
            title=f"Where the gap comes from, at {lo:,} evaluation inputs")
axes[1].set_xticklabels(levels)
axes[1].legend(fontsize=8)
axes[1].grid(axis="y", alpha=.2)
fig.tight_layout()
if SAVE_FIGURES:
    fig.savefig(HERE / "estimator_crosscheck.png", dpi=180)
plt.show()


# %%
# ════════════════════════════════════════════════════════════════════════════
# 5. CALIBRATING THE MISSING MASS
# ════════════════════════════════════════════════════════════════════════════
# This report is the only place where the counting bias is MEASURED rather than
# guessed, because the exact estimator supplies the truth for the same model. So
# it is the right place to see what the Good-Turing missing mass f1/B is worth as
# a warning sign. Older reports carry only unique_fraction; both are plotted if
# present, but f1/B is the one with theory behind it.
counting = runs[runs.evaluated_with == "counting"].copy()
truth = (runs[runs.evaluated_with == "exact"]
         .groupby(["genes_per_cell", "samples", "trained_on", "cell_sampling_seed"]).mi.mean())
counting["bias"] = counting.apply(
    lambda r: truth.get((r.genes_per_cell, r.samples, r.trained_on, r.cell_sampling_seed),
                        np.nan) - r.mi, axis=1)

available = [c for c in ("missing_mass", "unique_fraction")
             if c in counting and counting[c].notna().any()]
if not available:
    print("No coverage diagnostic in this report.")
else:
    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 4.5), squeeze=False)
    for ax, column in zip(axes.flat, available):
        for g, part in counting.groupby("genes_per_cell"):
            ax.scatter(part[column], part.bias, label=f"g={g}", alpha=.8)
        ax.set(xlabel=column, ylabel="measured counting bias [bits]",
               title=f"bias against {column}")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Bias is measured as (exact - counting) on the SAME trained model", fontsize=10)
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(HERE / "estimator_crosscheck_calibration.png", dpi=180)
    plt.show()

    for column in available:
        part = counting.dropna(subset=[column, "bias"])
        if len(part) > 2:
            print(f"\n{column} against measured bias:")
            print(part[[column, "bias"]].sort_values(column)
                  .to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
