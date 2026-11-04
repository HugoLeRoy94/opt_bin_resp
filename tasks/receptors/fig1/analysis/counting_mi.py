# %%
"""Counting MI for the het_casc runs: convergence in the test size, then Figure 1.

Reads <data>/fig1/ng*/test_counting.csv written by scripts/test_final.py (latest sweep
per n_genes). MI_MM = Miller-Madow counted response entropy H(Y) minus the analytic
H(Y|X). Missing mass = Good-Turing f1/n (f1 patterns seen once, n sniffs), the
probability of response patterns never sampled.

Cell 1: MI_MM and missing mass vs test size, one line per run, one column per n_genes.
Cell 2: MI vs n_genes per heteromerization index R/n_genes. A condition (n_genes, R)
uses only its runs measured within x2 of its largest test size: mean, band = std when
>1 run. Filled marker = the run's lowest missing mass is below MISSING_TARGET.
"""
import sys, glob
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")  # exec dir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.plotlib import DATA_ROOT
from src.testscaling import MISSING_TARGET

MI, MISS = "mutual_information_counting_mm", "response_counting_missing_mass"
RATIOS = range(1, 6)

csvs = sorted(glob.glob(str(DATA_ROOT / "fig1" / "ng*" / "test_counting.csv")))
if not csvs:
    raise SystemExit("no data/fig1/ng*/test_counting.csv: run scripts/test_final.py and sync first")
df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
df = df[df["sweep_folder"] == df.groupby("n_genes")["sweep_folder"].transform("max")].copy()
if MISS not in df:
    df[MISS] = np.nan
df["run_missing"] = df.groupby(["sweep_folder", "run_dir"])[MISS].transform("min")  # old rows lack it
df["het_ratio"] = (df["n_receptors"] / df["n_genes"]).round().astype(int)
genes = sorted(df["n_genes"].unique())
levels = sorted(r for r in df["het_ratio"].unique() if r in RATIOS)
cmap = plt.colormaps["viridis"]


def col(ratio):
    span = max(1, max(levels) - min(levels))
    return cmap(0.12 + 0.78 * (ratio - min(levels)) / span)


ratio_handles = [Line2D([0], [0], color=col(r), lw=2, label=fr"$R/n_g\approx{r}$") for r in levels]

# %% ── convergence: MI and missing mass vs test size ─────────────────────────
fig, axs = plt.subplots(2, len(genes), figsize=(2.6 * len(genes), 4.8),
                        sharex=True, squeeze=False)
for j, g in enumerate(genes):
    for (R, _), s in df[df["n_genes"] == g].groupby(["n_receptors", "run_dir"]):
        s = s.sort_values("test_size")
        c = col(round(R / g))
        axs[0][j].plot(s["test_size"], s[MI], "-o", color=c, ms=2.5, lw=1.0)
        axs[1][j].plot(s["test_size"], s[MISS], "-o", color=c, ms=2.5, lw=1.0)
    axs[0][j].set_title(f"G={g}", fontsize=9)
    axs[1][j].axhline(MISSING_TARGET, color="0.5", ls=":", lw=1.0)
    axs[1][j].set_xscale("log", base=2)
    axs[1][j].set_yscale("log")
    axs[1][j].set_xlabel("test size [sniffs]", fontsize=8)
axs[0][0].set_ylabel("MI_MM [bits]")
axs[1][0].set_ylabel("missing mass")
axs[0][-1].legend(handles=ratio_handles, fontsize=7, frameon=False,
                  loc="upper left", bbox_to_anchor=(1.02, 1.0))
fig.tight_layout()
plt.show()

# %% ── Figure 1: MI vs n_genes per heteromerization index ────────────────────
best = df.loc[df.groupby(["sweep_folder", "run_dir"])["test_size"].idxmax()]
best = best[2 * best["test_size"] >= best.groupby(["n_genes", "n_receptors"])["test_size"].transform("max")]
tab = (best.groupby(["n_genes", "n_receptors", "het_ratio"])
           .agg(n_runs=(MI, "size"), test_size=("test_size", "max"),
                mi=(MI, "mean"), mi_sd=(MI, "std"), missing=("run_missing", "max"))
           .reset_index())
print(tab.to_string(index=False))

fig, ax = plt.subplots(figsize=(5.6, 4.3))
ax.spines[["top", "right"]].set_visible(False)
ax.plot(np.arange(0, 16), np.arange(0, 16), color="0.6", ls=(0, (4, 3)), lw=1.0)
ax.text(max(genes), max(genes), " perfect\n homomers", fontsize=7, color="0.5", va="top")
for ratio in levels:
    t = tab[tab["het_ratio"] == ratio].sort_values("n_genes")
    c = col(ratio)
    sd = t["mi_sd"].fillna(0)
    ax.plot(t["n_genes"], t["mi"], color=c, lw=1.8)
    ax.fill_between(t["n_genes"], t["mi"] - sd, t["mi"] + sd, color=c, alpha=0.3, lw=0)
    ok = t["missing"] < MISSING_TARGET
    ax.scatter(t["n_genes"][ok], t["mi"][ok], color=c, s=20, zorder=3)
    ax.scatter(t["n_genes"][~ok], t["mi"][~ok], facecolor="white", edgecolor=c, s=20, zorder=3)
marker_handles = [Line2D([0], [0], color="0.35", lw=0, marker="o", label=f"missing mass < {MISSING_TARGET:g}"),
                  Line2D([0], [0], color="0.35", lw=0, marker="o", mfc="white", label="not converged")]
ax.legend(handles=ratio_handles + marker_handles, frameon=False, fontsize=7.5,
          loc="upper left", bbox_to_anchor=(1.02, 1.0))
ax.set_xlabel("$n_{\\mathrm{genes}}$")
ax.set_ylabel("MI  [bits]")
ax.set_xticks(genes)
ax.tick_params(direction="in")
fig.tight_layout()
plt.show()
# %%
