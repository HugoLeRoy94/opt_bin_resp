# %%
"""MI, H(response|input), H(response) and pattern count, against genes per cell.

One curve per (sweep, G). Replicates at the same target mean are averaged.
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

DATA = ROOT / "data/expression_law"
SWEEPS = [
    "expression_law_size_law_heteromers_exponential_uniform_20261013_094316",
    "expression_law_size_law_heteromers_uniform_uniform_20261013_094259",
]
KEYS = {"MI [bits]":                  "mutual_information_grouped_counting_mm",
        "H(response | input) [bits]": "conditional_entropy_response_grouped_counting",
        "H(response) [bits]":         "response_entropy_grouped_counting_mm",
        "patterns":                   "grouped_n_states"}

rows = []
for sweep in SWEEPS:
    # Replicates share a TARGET mean but land on different realised means, so the
    # manifest's target is what groups them; the realised mean never repeats.
    target = {r["cell_sampling_seed"]: r["target_mean_genes"] for r in
              json.loads((DATA / sweep / "experiment.json").read_text())["rows"]}
    for cfg, run in SweepLoader(str(DATA / sweep)).iter_run_dirs():
        result = json.loads((Path(run) / "test_results.json").read_text())
        rows.append({"sweep": sweep, "G": cfg.n_genes,
                     "genes": target[cfg.cell_sampling_seed],
                     **{label: np.mean(result[key]) for label, key in KEYS.items()}})

points = pd.DataFrame(rows).groupby(["sweep", "G", "genes"]).mean().reset_index()

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for (sweep, n_genes), curve in points.groupby(["sweep", "G"]):
    for ax, label in zip(axes, KEYS):
        ax.plot(curve["genes"], curve[label], "o-", label=f"{sweep} G={n_genes}")
        ax.set(xlabel="target mean genes per cell", ylabel=label)
axes[3].set_yscale("log")
axes[0].legend(fontsize=6)
plt.tight_layout()
plt.show()

# %%
