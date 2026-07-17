# %%
"""Compare two optimizer sweeps measured at different sample sizes.

Per condition, overlays the KT bracket (lower + upper) vs epoch for two sweeps
(SWEEP_A vs SWEEP_B) and draws the sample-size resolvable-entropy ceilings
log2(train_batch) and log2(test_batch): the measured entropy cannot exceed these,
so a sweep with a bigger test batch plateaus higher.

We compare KT because it is the estimator that scales with the measurement sample
count (blocked/collision are read on the fixed eval chunk). It is also the reported
quantity per our decision: KT lower = optimizer target, KT upper = bracketing.

Ceilings need the RESOLVED batch sizes, which are persisted only for runs made after
the config re-save fix; for older 'auto' runs the ceilings are skipped (with a note).
"""
import sys
sys.path.append("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")  # exec dir

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.plotlib import load_runs, load_epochs

GOAL = "optimizer"
CMP_OPT = "kt"                     # optimizer column to compare on
SWEEP_A, SWEEP_B = None, None      # sweep_folder; None → earliest / latest with KT

KT = {"full_array_entropy_kt":       ("KT lower", "tab:red"),
      "full_array_entropy_kt_upper": ("KT upper", "tab:purple")}
CEIL = {"train": dict(color="0.55", ls="--"),   # log2(train batch)
        "test":  dict(color="0.75", ls=":")}    # log2(test batch)


def _num(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _ceilings(sub):
    """log2(sample-size) ceilings from the resolved batch columns, if numeric."""
    out = {}
    for col, lab in (("batch_size", "train"), ("test_batch_size", "test")):
        if col in sub.columns:
            vals = [v for v in (_num(x) for x in sub[col].unique()) if v]
            if vals:
                out[lab] = float(np.log2(max(vals)))
    return out


def _test_dot(ax, dsub, metric, color, x, filled):
    """Final test-set value (test_results.json → `<metric>_mean`) as a dot at the
    curve end. A = open dot, B = filled."""
    col = metric + "_mean"
    if col not in dsub.columns:
        return
    v = dsub[col].dropna()
    if v.empty:
        return
    ax.plot([x], [v.mean()], marker="o", color=color, ms=9 if filled else 7,
            mec="k", mew=0.6, mfc=color if filled else "none", ls="none", zorder=5)


df = load_runs(GOAL, complete=False)
df = df[df["sweep_folder"].str.startswith("optimizer_")]  # exclude sample_limit_*
ep = load_epochs(df)
_kt = "full_array_entropy_kt"
avail = sorted(ep.dropna(subset=[_kt])["sweep_folder"].unique()) if _kt in ep.columns else []
A = SWEEP_A or (avail[0]  if avail else None)
B = SWEEP_B or (avail[-1] if avail else None)
print("sweeps with KT:", avail)
print(f"comparing A={A}  vs  B={B}")

conditions = sorted(set(zip(df["n_genes"], df["n_receptors"])))
any_ceiling = False

# %%
fig, axs = plt.subplots(1, len(conditions), figsize=(4.2 * len(conditions), 3.4), squeeze=False)
for c, (ng, nr) in enumerate(conditions):
    ax = axs[0][c]
    for sweep, ls_curve, lw in ((A, "--", 1.3), (B, "-", 2.4)):
        s = ep[(ep["n_genes"] == ng) & (ep["n_receptors"] == nr) &
               (ep["entropy"] == CMP_OPT) & (ep["sweep_folder"] == sweep)]
        if s.empty:
            continue
        x_text = s["epoch"].max()
        d = df[(df["n_genes"] == ng) & (df["n_receptors"] == nr) &
               (df["entropy"] == CMP_OPT) & (df["sweep_folder"] == sweep)]
        for metric, (_, color) in KT.items():
            if metric not in s.columns:
                continue
            g = s.groupby("epoch")[metric].mean().dropna()
            if not g.empty:
                ax.plot(g.index.values, g.values, color=color, ls=ls_curve, lw=lw)
            # final test-set value at the curve end (open=A, filled=B)
            _test_dot(ax, d, metric, color, x_text, filled=(sweep == B))
        # sample-size ceilings for this sweep
        for lab, y in _ceilings(s).items():
            any_ceiling = True
            ax.axhline(y, lw=1.0, alpha=0.9, **CEIL[lab])
            ax.text(x_text, y, f" log2({lab})", fontsize=6, va="bottom", color=CEIL[lab]["color"])
    ax.set_ylim(0, nr)
    ax.set_title(f"G={ng}, R={nr}")
    ax.set_xlabel("epoch")
    if c == 0:
        ax.set_ylabel(f"KT entropy [bits] — optimizer = {CMP_OPT}")

if not any_ceiling:
    print("NOTE: no ceiling lines drawn — the compared sweeps store batch_size/"
          "test_batch_size as 'auto' (run made before the resolved-size persist fix).")

handles = [Line2D([0], [0], color=col, label=lab) for lab, col in KT.values()]
handles += [Line2D([0], [0], color="0.3", ls="--", lw=1.3, label=f"A: {A[-6:] if A else '?'} (fewer)"),
            Line2D([0], [0], color="0.3", ls="-",  lw=2.4, label=f"B: {B[-6:] if B else '?'} (more)"),
            Line2D([0], [0], label="log2(train batch)", lw=1.0, **CEIL["train"]),
            Line2D([0], [0], label="log2(test batch)",  lw=1.0, **CEIL["test"]),
            Line2D([0], [0], color="0.3", marker="o", mfc="none", mec="k", ls="none", label="final test A"),
            Line2D([0], [0], color="0.3", marker="o", mfc="0.3", mec="k", ls="none", label="final test B")]
fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.14))
fig.suptitle("KT bracket vs epoch — two sample sizes, with sample-size ceilings", y=1.22)
fig.tight_layout()
plt.show()

# %%
