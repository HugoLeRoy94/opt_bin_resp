#!/usr/bin/env python3
"""Retrain with grouped_kt_mi at several TRAINING batch sizes, judged by the exact estimator.

Run estimator_crosscheck.py FIRST. It decomposes the exact-vs-KT gap without
retraining anything, and if it reports an optimization loss near zero then the KT
objective is already finding the same optimum and this sweep has nothing to find.

The point of this script is to remove the evaluation confound. Every run here is
measured with `grouped_information`, the exact enumerating estimator, whatever it
was trained on. So a difference between batch sizes is a difference in the models,
not in how they were read. `grouped_counting` is measured alongside purely as a
reference for how far the sampled estimator is off on the same model.

Exact evaluation caps the design: the joint count alphabet prod_j(n_j+1) reaches
about 4.6e5 for G=5, C=30 complete coverage, which fits, but nothing much larger
will. --dry_run prints the alphabet before committing.

    python3 tasks/cells/gene_expression/scripts/training_batch.py --dry_run
    python3 tasks/cells/gene_expression/scripts/training_batch.py --batch_sizes 1024 4096 16384
"""
import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.append('/app')
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch

from src.config import RunConfig
from src.run import SweepRunner
from tasks.cells.gene_expression._experiments import expression_sets

K_SUB = 5


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", choices=("heteromers", "homomers"), default="heteromers")
    p.add_argument("--coverage", choices=("random", "complete"), default="complete")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[1024, 4096, 16384],
                   help="The swept axis: gradient batch size during training.")
    p.add_argument("--genes_per_cell", type=int, nargs="+", default=[2, 3],
                   help="Expression levels to retrain. Default: the two with the largest gap.")
    p.add_argument("--n_genes", type=int, default=5)
    p.add_argument("--n_cells", type=int, default=30)
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--replicate_start", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--entropy", choices=("grouped_kt_mi", "grouped_mi"), default="grouped_kt_mi")
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--final_batch_size", type=int, default=65536)
    p.add_argument("--eval_chunk_size", type=int, default=256)
    p.add_argument("--max_states", type=int, default=1048576)
    p.add_argument("--base_folder", default="/app/data/gene_expression")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args(argv)


def design(args):
    """One row per planned run: replicate x expression level x training batch size.

    Gene sets reuse tasks/cells/gene_expression expression_sets with the same seed
    formula as replicates.py, so a row here is the SAME array as the corresponding
    replicates run and the two can be compared directly.
    """
    rows = []
    for replicate in range(args.replicate_start, args.replicate_start + args.replicates):
        seed = args.seed * 100000 + replicate
        for g in sorted(set(args.genes_per_cell)):
            sets = expression_sets(args.n_genes, args.n_cells, g, seed, args.coverage)
            states = math.prod(n + 1 for n in Counter(sets).values())
            if states > args.max_states:
                raise ValueError(
                    f"g={g}, replicate={replicate}: {states:,} count states exceed "
                    f"--max_states={args.max_states:,}. Exact evaluation is the whole "
                    "point of this sweep, so lower n_cells or g rather than the cap.")
            for batch in sorted(set(args.batch_sizes)):
                rows.append({"replicate": replicate, "cell_sampling_seed": seed,
                             "genes_per_cell": g, "batch_size": batch,
                             "cell_gene_sets": sets, "count_states": states})
    return rows


def build_config(args, rows):
    cells = ({"cell_gene_sets": [r["cell_gene_sets"] for r in rows]}
             if args.condition == "heteromers" else
             {"cell_receptors": [tuple(tuple((g,) * K_SUB for g in genes)
                                       for genes in r["cell_gene_sets"]) for r in rows]})
    # Both estimators on every run: `grouped_information` is the unbiased judge,
    # `grouped_counting` records what the sampled one would have said instead.
    measurements = ("grouped_information", "grouped_counting",
                    "conditional_entropy_response", "codeword_entropy")
    return RunConfig(
        n_families=5, n_ligands=100, latent_dim=6, family_spread=0.1,
        average_family_distance=1.0, environment_geometry="asymmetric",
        distribution_type="uniform", observation_noise_sigma=0.0,
        n_presence_blocks=1, mu_sources=1.0, mu_ligands_per_source=1e-6,
        conc_model_type="lognormal", conc_mean=(0.0,) * 100, conc_std=(1.0,) * 100,
        block_shared_conc_mean=False,

        n_genes=args.n_genes, k_sub=K_SUB, temperature=0.05, initial_temperature=3.0,
        affinity_kernel="gaussian", kernel_params=(1.0,), use_interface_model=True,

        **cells,
        n_cells=args.n_cells,
        cell_sampling_seed=[r["cell_sampling_seed"] for r in rows],
        cell_max_genes=[r["genes_per_cell"] for r in rows],
        cell_stoichiometry="multinomial", cell_readout="mean",
        cell_pool_chunk=128, recompute_backward=True,

        entropy=args.entropy, cell_grouped_max_states=args.max_states,
        epochs=args.epochs, lr=1e-2, use_scheduler=False,
        batch_size=[r["batch_size"] for r in rows],
        test_batch_size=[r["batch_size"] for r in rows],
        final_test_batch_size=args.final_batch_size,
        eval_chunk_size=args.eval_chunk_size, per_epoch_measure=False,
        measurement_fns=measurements, final_measurement_fns=measurements,

        sweep_name=f"training_batch_{args.condition}_{args.coverage}",
        base_folder=args.base_folder, warm_start=False,
    )


def main(argv=None):
    args = parse_args(argv)
    rows = design(args)
    seed_key = f"training_batch:{args.condition}:{args.coverage}:{args.seed}:{args.replicate_start}"
    world_seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:15], 16)

    print(f"training_batch: {args.condition}, {args.coverage} coverage, objective {args.entropy}")
    print(f"{len(rows)} independent optimizations "
          f"({args.replicates} replicates x {len(set(args.genes_per_cell))} levels "
          f"x {len(set(args.batch_sizes))} batch sizes)")
    print(f"World seed {world_seed}\n")
    print(f"{'g':>3} {'batch':>7} {'count states':>14} rep")
    for r in rows:
        print(f"{r['genes_per_cell']:>3} {r['batch_size']:>7} {r['count_states']:>14,} "
              f"{r['replicate']}")
    print(f"\nLargest count alphabet: {max(r['count_states'] for r in rows):,}")
    print(f"Exact enumeration holds a (eval_chunk_size, states) table: "
          f"{args.eval_chunk_size * max(r['count_states'] for r in rows) * 4 / 2**30:.2f} GiB")
    print("Every run is judged by the EXACT estimator, so differences between batch\n"
          "sizes are differences between models, not between measurements.")

    config = build_config(args, rows)
    manifest = {"schema": 1, "experiment": "training_batch", "condition": args.condition,
                "coverage": args.coverage, "world_seed": world_seed,
                "arguments": vars(args), "rows": rows}
    if args.dry_run:
        return manifest

    torch.manual_seed(world_seed)
    runner = SweepRunner(config)
    (Path(runner.master_logger.sweep_root) / "experiment.json").write_text(
        json.dumps(manifest, indent=2))
    start = time.time()
    runner.execute()
    print(f"\ntraining_batch finished in {(time.time() - start) / 3600:.2f} h")
    return manifest


if __name__ == "__main__":
    main()
