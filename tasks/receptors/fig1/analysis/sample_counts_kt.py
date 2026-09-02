# %%
"""Training / test sample counts for the het_casc KT runs (the ones plotted by
impact_of_heteromerization_kt.py).

- TRAIN samples per step = resolved `batch_size` (auto → memory-max for KT).
- TEST samples (the FINAL measurement the figure bracket is read on) = 4 × batch_size,
  because het_casc runs with per_epoch_measure=False (the persisted `test_batch_size`
  is the *unused* per-epoch value, so do NOT read it — compute 4× instead).

Prints a per-(n_genes, R) table (mean over the environment runs) and draws train/test
samples vs R. Purely reads config — no GPU, no measurement.
"""
import sys
from pathlib import Path
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")  # exec dir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.plotlib import load_runs, latest_sweep

GOAL = "fig1"

df = latest_sweep(load_runs(GOAL, receptor_type="heteromer")).copy()
df["train_samples"] = pd.to_numeric(df["batch_size"], errors="coerce")
df["test_samples"]  = 4 * df["train_samples"]              # per_epoch_measure=False → final = 4×train
df["het_ratio"]     = (df["R"] / df["n_genes"]).round().astype(int)

# %% ── table (also the "array" you can copy) ─────────────────────────────────
tab = (df.groupby(["n_genes", "n_receptors"])
         .agg(train=("train_samples", "mean"),
              test=("test_samples", "mean"),
              n_runs=("train_samples", "size"))
         .round(0).astype({"train": int, "test": int})
         .reset_index())
pd.set_option("display.width", 120)
print(tab.to_string(index=False))
# as a raw numpy array [n_genes, R, train, test]:
arr = tab[["n_genes", "n_receptors", "train", "test"]].to_numpy()
# print("\narray [n_genes, R, train, test]:\n", arr)

# %% ── plot: samples vs R, train vs test ────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
genes = sorted(df["n_genes"].unique())
cmap = plt.colormaps["viridis"]
for i, g in enumerate(genes):
    s = tab[tab["n_genes"] == g].sort_values("n_receptors")
    color = cmap(i / max(1, len(genes) - 1))
    ax.plot(s["n_receptors"], s["train"], "-o", color=color, ms=5, label=f"G={g}")
    ax.plot(s["n_receptors"], s["test"],  "--s", color=color, ms=4, mfc="none")
ax.set_yscale("log", base=2)
ax.set_xlabel("R (n_receptors)")
ax.set_ylabel("samples")
ax.set_title("Sample counts — solid=train (batch), dashed=test (4×train)")
ax.legend(fontsize=8, title="n_genes", ncol=2)
ax.grid(True, which="both", alpha=0.2)
fig.tight_layout()
plt.show()
# %%
