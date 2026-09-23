#!/usr/bin/env python3
"""Re-evaluate saved cell models at increasing input budgets, without retraining."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.path.append('/app')
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import torch

from src.cells import CellReadout, cell_activity
from src.grouped_loss import GroupedCellMutualInformationLoss, GroupedResponseCounter
from src.response_groups import CellGrouping
from src.IO import SingleRunLoader
from src.plotlib import load_model


@torch.no_grad()
def evaluate_budgets(env, physics, ri, readout, estimator, budgets, repeats, seed,
                     chunk_size=512, pool_chunk=128):
    """Nested input samples within each repeat; repeat scatter is evaluation noise."""
    if not budgets or min(budgets) < 1 or min(repeats, chunk_size, pool_chunk) < 1:
        raise ValueError("Budgets, repeats, and chunk sizes must be positive.")
    budgets = sorted(set(budgets))
    records = []
    counting = isinstance(estimator, CellGrouping)
    for repeat in range(repeats):
        torch.manual_seed(seed + repeat)
        if counting:
            counter = GroupedResponseCounter(estimator)
            generator = torch.Generator(device=ri.device).manual_seed(seed + repeat + 104729)
        else:
            probability_sum = torch.zeros(estimator.n_states, dtype=torch.float64, device=ri.device)
            conditional_sum = torch.zeros((), dtype=torch.float64, device=ri.device)
        n_have = 0
        for budget in budgets:
            while n_have < budget:
                n = min(chunk_size, budget - n_have)
                E, concentrations, _ = env.sample_batch(n, receptor_indices=ri)
                activity = cell_activity(
                    physics, readout, E, concentrations, ri,
                    pre_gathered=env.use_interface_model, composition=env.composition,
                    chunk_size=pool_chunk)
                if counting:
                    counter.update(activity, generator)
                else:
                    p_sum, h_sum = estimator.sufficient_statistics(activity)
                    probability_sum += p_sum.double()
                    conditional_sum += h_sum.double()
                n_have += n
            metrics = (counter.metrics() if counting else
                       estimator.metrics_from_statistics(probability_sum, conditional_sum, n_have))
            records.append(dict(samples=n_have, repeat=repeat, seed=seed + repeat,
                                input_sample_ceiling=math.log2(n_have),
                                **{k: v.item() if isinstance(v, torch.Tensor) else v
                                   for k, v in metrics.items()}))
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[4096, 16384, 65536])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--pool_chunk", type=int, default=128)
    parser.add_argument("--max_states", type=int, default=None)
    parser.add_argument("--estimator", choices=("exact", "counting"), default="exact")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    for folder in args.run_dirs:
        run_dir = Path(folder).resolve()
        loader = SingleRunLoader(str(run_dir))
        cfg = loader.load_config()
        if not cfg.is_cell_mode():
            raise ValueError(f"This evaluation is cell-only: {run_dir}")
        env, physics, ri = load_model(run_dir=str(run_dir), device=device)
        checkpoint = loader.load_checkpoint(map_location=device)
        readout = CellReadout(
            checkpoint["readout_state"]["W"], mode=checkpoint["readout_mode"],
            temperature=checkpoint["readout_temperature"], k_sub=cfg.k_sub,
            learnable_threshold=cfg.cell_threshold_learnable).to(device)
        readout.load_state_dict(checkpoint["readout_state"])
        readout.eval()
        env.bind_receptors(ri)
        grouping = CellGrouping(readout.W).to(device)
        estimator = (grouping if args.estimator == "counting" else GroupedCellMutualInformationLoss(
            grouping, cfg.cell_grouped_max_states if args.max_states is None else args.max_states).to(device))
        records = evaluate_budgets(env, physics, ri, readout, estimator, args.budgets,
                                   args.repeats, args.seed, args.chunk_size, args.pool_chunk)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output = run_dir / f"grouped_evaluation_budget_{stamp}.json"
        output.write_text(json.dumps(dict(run_dir=str(run_dir), arguments=vars(args),
                                         n_groups=estimator.n_groups, n_states=estimator.n_states,
                                         records=records), indent=2))
        print(f"Saved {output}")
        mi_key = ("mutual_information_grouped_counting_plugin" if args.estimator == "counting"
                  else "mutual_information_grouped")
        for row in records:
            print(f"  B={row['samples']:>7} repeat={row['repeat']}: "
                  f"MI ({args.estimator})={row[mi_key]:.6f} bits")


if __name__ == "__main__":
    main()
