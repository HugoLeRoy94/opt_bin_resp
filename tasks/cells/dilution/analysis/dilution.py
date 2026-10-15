# %%
"""Dilution against heteromerisation, with everything else pinned.

Cell-type structure is identical across the three arms at every point (the same
shared_subsets pattern applied to three different pools) and the pool size is fixed
at 8. A vertical gap between curves is heteromerisation per se; the slope along x
is dilution per se.

MI COMES FROM THE KT OBJECTIVE, NOT FROM COUNTING. The sampled counting estimator
is unusable on this design: it estimates H(K) from B samples, so it cannot exceed
log2(B) = 20 bits, while the exact H(K|X) reaches 28. MI = H(K)hat - H(K|X) then
goes NEGATIVE by arithmetic. The right-hand panel plots both so the failure is
visible rather than silently averaged in.

What is still trustworthy from the counting run is H(Y|X): it is computed
analytically from the firing probabilities, never sampled, so it has no such
ceiling. That is the dilution diagnostic in the middle panel.

KT is read from the end of stats.csv. Receptor annealing finishes at 80% of
training, so the last fifth is at the final temperature; it is averaged over that
tail (std <= 0.05 bits) rather than taken from the single last step.
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

DATA = ROOT / "data/dilution"
SWEEPS = ["dilution_homomers", "dilution_dimers", "dilution_full_heteromers"]
TAIL = 0.8          # fraction of training after which the temperature is final
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
            history = SingleRunLoader(run).load_history()
            tail = history["train_mutual_information"].to_numpy()[int(TAIL * len(history)):]
            get = lambda key: float(np.mean(result[key]))
            rows.append({
                "n_distinct": manifest["n_distinct"],
                "per_cell": plan["receptors_per_cell"],
                "types": plan["distinct_repertoires"], "C": plan["n_cells"],
                "MI kt": tail.mean(), "MI kt sd": tail.std(),
                "MI counting": get("mutual_information_grouped_counting_mm"),
                "H(Y|X)": get("conditional_entropy_response_grouped_counting"),
                "H(K|X)": get("grouped_count_conditional_entropy_counting"),
                "log2B": get("grouped_counting_log2B")})
points = pd.DataFrame(rows).sort_values(["n_distinct", "per_cell"])
points["H(Y|X)/cell"] = points["H(Y|X)"] / points["C"]
print(points.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for n_distinct, curve in points.groupby("n_distinct"):
    marker, colour, label = STYLE[n_distinct]
    axes[0].errorbar(curve["per_cell"], curve["MI kt"], yerr=curve["MI kt sd"],
                     fmt=marker, color=colour, label=label, capsize=3)
    axes[1].plot(curve["per_cell"], curve["H(Y|X)/cell"], marker, color=colour,
                 label=label)
    axes[2].plot(curve["per_cell"], curve["MI kt"], marker, color=colour, label=f"{label} KT")
    axes[2].plot(curve["per_cell"], curve["MI counting"], ":", color=colour, alpha=.6)
axes[0].set(xlabel="receptors per cell (dilution)", ylabel="KT MI [bits]")
axes[1].set(xlabel="receptors per cell (dilution)", ylabel="H(Y|X) per cell [bits]")
axes[1].axhline(1.0, color="0.6", lw=.8, ls=":")      # a fair coin
axes[2].set(xlabel="receptors per cell (dilution)", ylabel="MI [bits]",
            title="KT (solid) vs sampled counting (dotted)")
axes[2].axhline(0.0, color="0.4", lw=.8)
for ax in axes:
    ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle("Pool size 8, cell-type structure identical across arms")
plt.tight_layout()
plt.show()
