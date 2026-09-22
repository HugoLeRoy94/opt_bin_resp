# %%
"""Input-budget convergence on fixed trained models, distinct from world variability."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data" / "gene_expression"
# Set explicit report paths to compare older evaluations. Otherwise latest per model.
REPORTS = None
paths = list(map(Path, REPORTS)) if REPORTS is not None else sorted(DATA.rglob("grouped_evaluation_budget_*.json"))
latest = {path.parent: path for path in paths}
rows = []
for path in latest.values():
    report = json.loads(path.read_text())
    cfg = json.loads((path.parent / "config.json").read_text())
    g = len(cfg["cell_gene_sets"][0])
    condition = "homomers" if cfg.get("cell_receptors") is not None else "heteromers"
    label = f"{condition}, G={cfg['n_genes']}, g={g}, {path.parent.name}"
    rows.extend(dict(row, run_dir=str(path.parent), label=label) for row in report["records"])
RESULTS = pd.DataFrame(rows)
if RESULTS.empty:
    print("No evaluation-budget reports found. Run scripts/evaluation_budget.py on saved run folders.")

# %%
if not RESULTS.empty:
    SUMMARY = RESULTS.groupby(["run_dir", "label", "samples"], as_index=False).agg(
        mi_mean=("mutual_information_grouped", "mean"), mi_sd=("mutual_information_grouped", "std"),
        repeats=("repeat", "size"), response_entropy=("response_entropy_grouped", "mean"))
    SUMMARY["mi_sem"] = SUMMARY.mi_sd / np.sqrt(SUMMARY.repeats)
    print(SUMMARY.to_string(index=False))
    print("Error bars are input-sampling SEM for fixed models, not independent-world uncertainty.")
    fig, ax = plt.subplots(figsize=(11, 6))
    for (_, label), part in SUMMARY.groupby(["run_dir", "label"]):
        part = part.sort_values("samples")
        ax.plot(part.samples, part.mi_mean, "o-", label=label)
        valid = part.mi_sem.notna()
        ax.errorbar(part.loc[valid, "samples"], part.loc[valid, "mi_mean"],
                    yerr=part.loc[valid, "mi_sem"], fmt="none", color=ax.lines[-1].get_color(), capsize=3)
    budgets = np.sort(SUMMARY.samples.unique())
    ax.plot(budgets, np.log2(budgets), "k:", label="log₂(input samples)")
    ax.set(xlabel="evaluation input samples", ylabel="total MI [bits]")
    ax.set_xscale("log", base=2)
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    # fig.savefig(Path(__file__).with_name("evaluation_budget.png"), dpi=180)
    plt.show()
