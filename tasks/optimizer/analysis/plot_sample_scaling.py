# %%
"""Sample-limit study — plots for the two questions, from test_scaling.csv.

Data: <data>/optimizer/sample_limit_*/test_scaling.csv, produced by
scripts/test_scaling.py (KT lower + upper re-measured on the frozen envs at a
ladder of test sizes, for each trained train_batch).

Figure 1 — "entropy vs test size" (Q2: what does a bigger test set buy?):
  per condition, KT_lower (solid) and KT_upper (dashed) vs test_size, one colour
  per train_batch. The grey diagonal is the resolvable-entropy ceiling
  log2(test_size): a curve can never cross it. Rising-then-flat = the expected
  plateau; the vertical gap to the diagonal is the sample headroom.

Figure 2 — "am I reaching the ceiling?" (Q1): achieved KT_lower vs its ceiling
  log2(sample_size), with the y=x line. Two points per (condition, train_batch):
  the TRAIN sample (test size nearest train_batch, ceiling log2(train_batch)) and
  the max TEST sample. On the line → sample-limited; below → optimization/physics-
  limited (extra samples won't help).
"""
import sys, glob, os
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")  # exec dir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.plotlib import DATA_ROOT

csvs = sorted(glob.glob(str(DATA_ROOT / "optimizer" / "sample_limit_*" / "test_scaling.csv")))
if not csvs:
    raise SystemExit("no test_scaling.csv found — run scripts/test_scaling.py and sync first")
df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
df = df.dropna(subset=["train_batch"])
df["train_batch"] = df["train_batch"].astype(int)

conditions = sorted(set(zip(df["n_genes"], df["n_receptors"])))
batches = sorted(df["train_batch"].unique())
cmap = plt.colormaps["viridis"]
col = {b: cmap(i / max(1, len(batches) - 1)) for i, b in enumerate(batches)}

# %% ── Figure 1: KT vs test size ────────────────────────────────────────────
fig1, axs = plt.subplots(1, len(conditions), figsize=(4.6 * len(conditions), 3.8),
                         squeeze=False)
for c, (ng, nr) in enumerate(conditions):
    ax = axs[0][c]
    sub = df[(df["n_genes"] == ng) & (df["n_receptors"] == nr)]
    xs = np.array(sorted(sub["test_size"].unique()), float)
    ax.plot(xs, np.log2(xs), color="0.6", ls="-", lw=1.2, zorder=1)  # log2(test) ceiling
    ax.text(xs[-1], np.log2(xs[-1]), " log2(test)", fontsize=7, va="bottom", color="0.5")
    for b in batches:
        d = sub[sub["train_batch"] == b].sort_values("test_size")
        if d.empty:
            continue
        ax.plot(d["test_size"], d["kt_lower"], "-o", color=col[b], ms=4, lw=1.6)
        ax.plot(d["test_size"], d["kt_upper"], "--", color=col[b], lw=1.2, alpha=0.8)
        ax.axvline(b, color=col[b], ls=":", lw=0.9, alpha=0.6)  # train-batch marker
    ax.set_xscale("log", base=2)
    #ax.set_ylim(0, nr)
    ax.set_title(f"G={ng}, R={nr}")
    ax.set_xlabel("test sample size")
    if c == 0:
        ax.set_ylabel("KT entropy [bits]")
handles = [Line2D([0], [0], color=col[b], marker="o", label=f"train={b}") for b in batches]
handles += [Line2D([0], [0], color="0.3", ls="-", label="KT lower"),
            Line2D([0], [0], color="0.3", ls="--", label="KT upper"),
            Line2D([0], [0], color="0.6", label="log2(test) ceiling")]
fig1.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=8,
            bbox_to_anchor=(0.5, 1.10))
fig1.suptitle("Entropy gain vs test sample size", y=1.14)
fig1.tight_layout()

# %% ── Figure 2: achieved vs ceiling ─────────────────────────────────────────
fig2, axs2 = plt.subplots(1, len(conditions), figsize=(4.2 * len(conditions), 3.8),
                          squeeze=False)
for c, (ng, nr) in enumerate(conditions):
    ax = axs2[0][c]
    sub = df[(df["n_genes"] == ng) & (df["n_receptors"] == nr)]
    lim = np.log2(sub["test_size"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="0.6", lw=1.0, zorder=1)  # y = x (at ceiling)
    for b in batches:
        d = sub[sub["train_batch"] == b].sort_values("test_size")
        if d.empty:
            continue
        # train sample: point whose test_size is nearest the train batch
        tr = d.iloc[(d["test_size"] - b).abs().argmin()]
        ax.scatter(np.log2(b), tr["kt_lower"], color=col[b], marker="s", s=55,
                   edgecolor="k", linewidth=0.5, zorder=3)          # train
        te = d.iloc[-1]                                              # max test
        ax.scatter(np.log2(te["test_size"]), te["kt_lower"], color=col[b], marker="o",
                   s=55, edgecolor="k", linewidth=0.5, zorder=3)    # test
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(f"G={ng}, R={nr}")
    ax.set_xlabel("ceiling  log2(sample size) [bits]")
    if c == 0:
        ax.set_ylabel("achieved KT lower [bits]")
handles2 = [Line2D([0], [0], color=col[b], marker="s", ls="none", label=f"train={b}") for b in batches]
handles2 += [Line2D([0], [0], color="0.3", marker="s", ls="none", label="train sample"),
             Line2D([0], [0], color="0.3", marker="o", ls="none", label="max test sample"),
             Line2D([0], [0], color="0.6", label="on ceiling (y=x)")]
fig2.legend(handles=handles2, loc="upper center", ncol=len(handles2), fontsize=8,
            bbox_to_anchor=(0.5, 1.10))
fig2.suptitle("Am I reaching the sample ceiling?  (on y=x → sample-limited)", y=1.14)
fig2.tight_layout()

# %% ── Figure 3: GPU memory / util traces ──────────────────────────────────
# nvidia-smi traces from monitor_gpu.sh. The old ones sit loose in the task root
# (pre finalize_run.sh, so unlinked to a sweep); newer runs land theirs inside the
# sweep folder. We just plot every trace we can find, one line per file, on a shared
# elapsed-time axis (each file zeroed at its own first sample — they are NOT aligned).
gpu_csvs = sorted(glob.glob(str(DATA_ROOT / "optimizer" / "gpu_mem_*.csv")) +
                  glob.glob(str(DATA_ROOT / "optimizer" / "*" / "gpu_mem_*.csv")))
if gpu_csvs:
    cmap3 = plt.colormaps["turbo"]
    fig3, (axm, axu) = plt.subplots(1, 2, figsize=(11, 4))
    for i, path in enumerate(gpu_csvs):
        g = pd.read_csv(path, skipinitialspace=True)
        t = pd.to_datetime(g["timestamp"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
        mins = (t - t.iloc[0]).dt.total_seconds() / 60.0
        lab = os.path.relpath(path, DATA_ROOT / "optimizer")[:-4].replace("gpu_mem_", "")
        c = cmap3(i / max(1, len(gpu_csvs) - 1))
        axm.plot(mins, g["mem_used_MiB"] / 1024.0, color=c, lw=1.3, label=lab)
        axu.plot(mins, g["gpu_util_pct"], color=c, lw=1.0)
    axm.set_xlabel("elapsed [min]"); axm.set_ylabel("GPU memory used [GiB]")
    axm.set_xscale('log')
    axu.set_xlabel("elapsed [min]"); axu.set_ylabel("GPU util [%]"); axu.set_ylim(0, 105)
    fig3.legend(loc="upper center", ncol=min(3, len(gpu_csvs)), fontsize=7,
                bbox_to_anchor=(0.5, 1.06))
    fig3.suptitle("GPU memory / utilisation traces", y=1.12)
    fig3.tight_layout()
else:
    print("no gpu_mem_*.csv found under data/optimizer/")

plt.show()
# %%
