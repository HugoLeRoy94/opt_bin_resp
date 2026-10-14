"""Shared loader: every finished fig2 run as one DataFrame row.

The only thing the three analysis files have in common. Each run is joined back to
its design point through cell_sampling_seed, which the sweep script wrote into
experiment.json before training.
"""
import json
import sys
from pathlib import Path

ROOT = Path("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.IO import SweepLoader

DATA = ROOT / "data/fig2"
SWEEPS = [
    "fig2_complete_heteromers",
    "fig2_complete_homomers",
    "fig2_bernoulli_heteromers",
    "fig2_bernoulli_homomers",
]
# All from grouped_counting, so MI = H(Y) - H(Y|X) holds exactly: the label term
# cancels between H(Y) = H(K) + d and H(Y|X) = H(K|X) + d.
METRICS = {
    "MI":           "mutual_information_grouped_counting_mm",
    "H(Y)":         "response_entropy_grouped_counting_mm",
    "H(Y|X)":       "conditional_entropy_response_grouped_counting",
    "patterns":     "grouped_counting_K_hat",
    "missing mass": "grouped_counting_missing_mass",
}


def runs():
    """One row per finished run, with its design fields and its final metrics."""
    rows = []
    for prefix in SWEEPS:
        for root in sorted(DATA.glob(f"{prefix}_*")):
            design = {r["cell_sampling_seed"]: r for r in
                      json.loads((root / "experiment.json").read_text())["rows"]}
            for cfg, run_dir in SweepLoader(str(root)).iter_run_dirs():
                path = Path(run_dir) / "test_results.json"
                if not path.exists():
                    continue
                plan = design[cfg.cell_sampling_seed]
                result = json.loads(path.read_text())
                rows.append({
                    "sweep": root.name, "run": run_dir,
                    "strategy": plan["strategy"], "condition": plan["condition"],
                    "arm": f"{plan['strategy']}/{plan['condition']}",
                    "G": plan["n_genes"], "C": plan["n_cells"],
                    "genes": plan["realised_genes"], "target": plan["target_genes"],
                    "types": plan["distinct_gene_sets"],
                    **{label: np.mean(result[key]) for label, key in METRICS.items()
                       if key in result},
                })
    if not rows:
        raise SystemExit(f"No finished fig2 run under {DATA}. Run the four scripts first.")
    return pd.DataFrame(rows)


STYLE = {"complete/heteromers":  ("o-",  "tab:blue"),
         "complete/homomers":    ("s--", "tab:blue"),
         "bernoulli/heteromers": ("o-",  "tab:orange"),
         "bernoulli/homomers":   ("s--", "tab:orange")}
