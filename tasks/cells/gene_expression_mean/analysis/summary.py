# %%
"""MI, H(response|input), H(response) and pattern count, against genes per cell.

One curve per (sweep, G). Replicates at the same genes per cell are averaged.
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

DATA = ROOT / "data/gene_expression"
SWEEPS = [
    "mean_heteromers_20261013_111933",
    "mean_homomers_20261013_111944",
]
KEYS = {"MI [bits]":                  "mutual_information_grouped",
        "H(response | input) [bits]": "conditional_entropy_response_grouped",
        "H(response) [bits]":         "response_entropy_grouped",
        "patterns":                   "grouped_n_states"}

rows = []
for sweep in SWEEPS:
    for cfg, run in SweepLoader(str(DATA / sweep)).iter_run_dirs():
        result = json.loads((Path(run) / "test_results.json").read_text())
        rows.append({"sweep": sweep, "G": cfg.n_genes,
                     "genes": len(cfg.cell_gene_sets[0]),
                     **{label: np.mean(result[key]) for label, key in KEYS.items()}})

points = pd.DataFrame(rows).groupby(["sweep", "G", "genes"]).mean().reset_index()

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for (sweep, n_genes), curve in points.groupby(["sweep", "G"]):
    for ax, label in zip(axes, KEYS):
        ax.plot(curve["genes"], curve[label], "o-", label=f"{sweep} G={n_genes}")
        ax.set(xlabel="genes per cell", ylabel=label)
axes[3].set_yscale("log")
axes[0].legend(fontsize=6)
plt.tight_layout()
plt.show()
