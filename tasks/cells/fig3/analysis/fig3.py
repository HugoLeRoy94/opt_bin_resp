# %%
"""Does the MI maximum move with k_sub?

Left: MI against genes per cell, one curve per k_sub, peak marked. Right: the
receptor pool on the same axis, which is the thing k_sub actually changes.

The prediction from the pool-versus-dilution account is that the peak moves LEFT as
k_sub falls (g=1 at k=2, g=2 at k=3 and k=5), because lowering k_sub slows the pool
down as much as it slows dilution. At k=2 the pool is exhausted at g=2, so that arm
is a pure-dilution control: any decline past g=2 has zero pool gain behind it.

Points are the mean over independent worlds; the band is their min/max.
"""
import json
import sys
from pathlib import Path

ROOT = Path("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.IO import SweepLoader, SingleRunLoader

DATA = ROOT / "data/fig3"
SWEEPS = ["fig3_k2", "fig3_k3", "fig3_k5"]
TAIL = 0.8          # fraction of training after which the temperature is final
STYLE = {2: ("o-", "tab:blue"), 3: ("s-", "tab:green"), 5: ("^-", "tab:red")}
KEYS = {"MI":     "mutual_information_grouped_counting_mm",
        "H(Y|X)": "conditional_entropy_response_grouped_counting",
        "miss":   "grouped_counting_missing_mass"}

rows = []
for prefix in SWEEPS:
    for root in sorted(DATA.glob(f"{prefix}_*")):
        manifest = json.loads((root / "experiment.json").read_text())
        design = {r["cell_sampling_seed"]: r for r in manifest["rows"]}
        for cfg, run in SweepLoader(str(root)).iter_run_dirs():
            path = Path(run) / "test_results.json"
            if not path.exists():
                continue
            plan = design[cfg.cell_sampling_seed]
            result = json.loads(path.read_text())
            history = SingleRunLoader(run).load_history()["train_mutual_information"]
            tail = history.to_numpy()[int(TAIL * len(history)):]
            rows.append({"k_sub": manifest["k_sub"], "genes": plan["genes_per_cell"],
                         "replicate": plan["replicate"],
                         "R_pool": len(cfg.receptor_indices),
                         "MI kt": tail.mean(),
                         **{k: float(np.mean(result[v])) for k, v in KEYS.items()}})
if not rows:
    raise SystemExit(f"No finished fig3 run under {DATA}. Run the three scripts first.")

raw = pd.DataFrame(rows)
numeric = [c for c in raw.columns if c not in ("k_sub", "genes", "replicate")]
grouped = raw.groupby(["k_sub", "genes"])[numeric]
points = grouped.mean().reset_index()
for column in numeric:
    points[f"{column} lo"] = grouped[column].min().to_numpy()
    points[f"{column} hi"] = grouped[column].max().to_numpy()
print(points.round(3).to_string(index=False))
print("\npeak position of MI by k_sub:")
print(points.loc[points.groupby("k_sub")["MI"].idxmax(), ["k_sub", "genes", "MI"]]
      .to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for k_sub, curve in points.groupby("k_sub"):
    marker, colour = STYLE[k_sub]
    axes[0].plot(curve["genes"], curve["MI"], marker, color=colour, label=f"k_sub={k_sub}")
    axes[0].fill_between(curve["genes"], curve["MI lo"], curve["MI hi"],
                         color=colour, alpha=.15, linewidth=0)
    peak = curve.loc[curve["MI"].idxmax()]
    axes[0].axvline(peak["genes"], color=colour, lw=.8, ls=":")
    axes[1].plot(curve["genes"], curve["R_pool"], marker, color=colour,
                 label=f"k_sub={k_sub}")
axes[0].set(xlabel="genes expressed per cell", ylabel="MI [bits]",
            title="dotted line = peak position")
axes[1].set(xlabel="genes expressed per cell", ylabel="receptors in pool",
            yscale="log", title="what k_sub actually changes")
for ax in axes:
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle("G = 5, C = 50, complete coverage, heteromers")
plt.tight_layout()
plt.show()
