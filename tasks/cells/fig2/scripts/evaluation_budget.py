#!/usr/bin/env python3
"""Re-measure every finished fig2 run at a ladder of evaluation budgets.

No retraining. Each run's saved model is reloaded and fresh sniffs are drawn, so
this isolates how much of a reported number is the ARRAY and how much is the
evaluation budget. Sampled counting is biased DOWNWARD when the response alphabet
is undersampled, and that bias falls only with the number of input samples. A curve
still rising at the largest budget is reporting the budget, not the array; the
Good-Turing missing mass alongside it says how much of the alphabet was never seen.

Budgets are NESTED within a repeat: the 2^18 point is the first 2^18 samples of
the 2^22 stream, so the curve is monotone in budget by construction and the only
scatter between repeats is evaluation noise.

Writes budgets.json next to each run. Re-run it and it overwrites.

    python3 tasks/cells/fig2/scripts/evaluation_budget.py
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.append('/app')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.cells import CellReadout, cell_activity
from src.grouped_loss import GroupedResponseCounter
from src.IO import SweepLoader, SingleRunLoader
from src.plotlib import load_model
from src.response_groups import CellGrouping

DATA = Path("/app/data/fig2")
SWEEPS = ("fig2_complete_heteromers", "fig2_complete_homomers",
          "fig2_bernoulli_heteromers", "fig2_bernoulli_homomers")
BUDGETS = (2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20, 2 ** 22)
REPEATS = 2
SEED = 7
CHUNK = 512
POOL_CHUNK = 128


@torch.no_grad()
def measure(env, physics, receptor_indices, readout, grouping, budgets, repeat_seed):
    """Grouped counting metrics at every budget, from one nested sample stream."""
    torch.manual_seed(repeat_seed)
    generator = torch.Generator(device=receptor_indices.device).manual_seed(repeat_seed + 104729)
    grouped = GroupedResponseCounter(grouping)
    n_have, records = 0, []
    for budget in sorted(budgets):
        while n_have < budget:
            n = min(CHUNK, budget - n_have)
            E, concentrations, _ = env.sample_batch(n, receptor_indices=receptor_indices)
            activity = cell_activity(
                physics, readout, E, concentrations, receptor_indices,
                pre_gathered=env.use_interface_model, composition=env.composition,
                chunk_size=POOL_CHUNK)
            grouped.update(activity, generator)
            n_have += n
        records.append(dict(
            samples=n_have, seed=repeat_seed, log2_samples=math.log2(n_have),
            **{key: (value.item() if isinstance(value, torch.Tensor) else value)
               for key, value in grouped.metrics().items()}))
    return records


def main():
    t0 = time.time()
    for sweep in SWEEPS:
        for root in sorted(DATA.glob(f"{sweep}_*")):
            for cfg, run_dir in SweepLoader(str(root)).iter_run_dirs():
                if not (Path(run_dir) / "best_model.pt").is_file():
                    continue
                env, physics, receptor_indices = load_model(run_dir=run_dir)
                checkpoint = SingleRunLoader(run_dir).load_checkpoint(map_location="cpu")
                readout = CellReadout(
                    checkpoint["readout_state"]["W"], mode=checkpoint["readout_mode"],
                    temperature=checkpoint["readout_temperature"], k_sub=cfg.k_sub,
                    learnable_threshold=cfg.cell_threshold_learnable)
                readout.load_state_dict(checkpoint["readout_state"])
                readout.eval()
                grouping = CellGrouping(readout.W)
                records = [record
                           for repeat in range(REPEATS)
                           for record in measure(env, physics, receptor_indices, readout,
                                                 grouping, BUDGETS, SEED + repeat)]
                (Path(run_dir) / "budgets.json").write_text(json.dumps(dict(
                    sweep=root.name, n_genes=cfg.n_genes,
                    n_cells=len(cfg.cell_gene_sets),
                    genes_per_cell=sum(map(len, cfg.cell_gene_sets)) / len(cfg.cell_gene_sets),
                    cell_sampling_seed=cfg.cell_sampling_seed,
                    records=records), indent=2))
                print(f"{Path(run_dir).name}: {len(records)} points")
    print(f"\nevaluation_budget finished in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
