# %%
"""Dilution against heteromerisation, with everything else pinned.

Cell-type structure is identical across the three arms at every point (the same
shared_subsets pattern applied to three different pools), and the pool size is
fixed at 8. So a vertical gap between the curves is heteromerisation per se, and
the slope along x is dilution per se.

Read it as: does H(Y|X)/cell rise with receptors per cell the same way in all three
arms? If yes, dilution is a property of averaging and nothing to do with what the
receptors are. Does MI separate between arms at fixed x? If yes, a mixed-subunit
receptor is a better detector than a homomer, independently of dilution.
"""
import json
import sys
from pathlib import Path

ROOT = Path("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.IO import SweepLoader

DATA = ROOT / "data/dilution"
SWEEPS = ["dilution_homomers", "dilution_dimers", "dilution_full_heteromers"]
KEYS = {"MI":     "mutual_information_grouped_counting_mm",
        "H(Y)":   "response_entropy_grouped_counting_mm",
        "H(Y|X)": "conditional_entropy_response_grouped_counting"}
STYLE = {1: ("o-", "tab:blue", "homomers (1 gene)"),
         2: ("s-", "tab:green", "dimers (2 genes)"),
         5: ("^-", "tab:red", "full heteromers (5 genes)")}

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
            rows.append({"n_distinct": manifest["n_distinct"],
                         "per_cell": plan["receptors_per_cell"],
                         "types": plan["distinct_repertoires"],
                         "C": plan["n_cells"],
                         **{k: np.mean(result[v]) for k, v in KEYS.items()}})
points = pd.DataFrame(rows).sort_values(["n_distinct", "per_cell"])
points["H(Y|X)/cell"] = points["H(Y|X)"] / points["C"]
print(points.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for n_distinct, curve in points.groupby("n_distinct"):
    marker, colour, label = STYLE[n_distinct]
    for ax, column in zip(axes, ("MI", "H(Y|X)/cell", "H(Y)")):
        ax.plot(curve["per_cell"], curve[column], marker, color=colour, label=label)
for ax, title in zip(axes, ("MI [bits]", "H(Y|X) per cell [bits]", "H(Y) [bits]")):
    ax.set(xlabel="receptors per cell (dilution)", ylabel=title)
    ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle("Pool size 8, cell-type structure identical across arms")
plt.tight_layout()
plt.show()
