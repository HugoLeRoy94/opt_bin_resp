# %%
"""Plateau check — test KT entropy vs test batch size for the het_casc runs.

Reads <data>/fig1/ng*/test_scaling.csv produced by scripts/test_scaling.py
(run it on the cluster with --data /app/data/fig1 --sweep_glob 'ng*' --test_sizes ...).

One panel per measured (n_genes, R): KT lower (solid) + upper (dashed) vs test size
(log2 x), averaged over whatever runs were measured. The vertical dotted line marks the
run's ACTUAL final-test size (4×train) — if the curve has flattened by there, the
figure's bracket is on the plateau; if it's still climbing, that condition needs a
larger test (re-run test_scaling.py with bigger --test_sizes).
"""
import sys, glob, os
from pathlib import Path
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")  # exec dir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.plotlib import DATA_ROOT

csvs = sorted(glob.glob(str(DATA_ROOT / "fig1" / "ng*" / "test_scaling.csv")))
if not csvs:
    raise SystemExit("no test_scaling.csv under data/fig1/ng*/ — run scripts/test_scaling.py "
                     "(--data /app/data/fig1 --sweep_glob 'ng*' ...) and sync first")
df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)

conds = sorted(set(zip(df["n_genes"], df["n_receptors"])))
genes = sorted({g for g, _ in conds})
# columns of the grid = distinct R/n_genes ratios present
ratios = sorted({int(round(r / g)) for g, r in conds})

# %%
fig, axs = plt.subplots(len(genes), len(ratios),
                        figsize=(2.9 * len(ratios), 2.3 * len(genes)), squeeze=False)
for r, g in enumerate(genes):
    for c, ratio in enumerate(ratios):
        ax = axs[r][c]
        R = int(round(g * ratio))
        sub = df[(df["n_genes"] == g) & (df["n_receptors"] == R)]
        if sub.empty:
            ax.set_axis_off()
            continue
        lo = sub.groupby("test_size")["kt_lower"].mean().sort_index()
        up = sub.groupby("test_size")["kt_upper"].mean().sort_index()
        ax.plot(lo.index, lo.values, "-o", color="tab:red", ms=3, lw=1.4, label="KT lower")
        ax.plot(up.index, up.values, "--s", color="tab:purple", ms=3, lw=1.1, mfc="none",
                label="KT upper")
        tb = pd.to_numeric(sub["train_batch"], errors="coerce").dropna()
        if not tb.empty:                       # current final test = 4 × train
            ax.axvline(4 * tb.mean(), color="0.5", ls=":", lw=1.0)
        ax.set_xscale("log", base=2)
        ax.set_title(f"G={g}, R={R}", fontsize=8)
        ax.tick_params(labelsize=6)
        if c == 0:
            ax.set_ylabel("KT [bits]", fontsize=7)
        if r == len(genes) - 1:
            ax.set_xlabel("test size", fontsize=7)
        if r == 0 and c == 0:
            ax.legend(fontsize=6, loc="lower right")

fig.suptitle("Test KT entropy vs test size (dotted = current 4×train) — plateau check",
             y=1.005, fontsize=11)
fig.tight_layout()
plt.show()
# %%
