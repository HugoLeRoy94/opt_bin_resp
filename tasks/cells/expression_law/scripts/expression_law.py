#!/usr/bin/env python3
"""Sweep the MEAN number of genes per cell under an independent expression law.

Every cell draws its own gene set with no knowledge of any other cell: first how
many genes it expresses (`size family`), then which ones (`gene family`). Nothing
guarantees that a gene is expressed anywhere, which is the point. The previous
gene_expression task fixed the size at exactly g and forced every gene to appear
at least once, both of which require coordination a developing tissue cannot do.

The knob is the MEAN genes per cell, so laws of different shape are compared at
matched expression level. Sweeping n_genes alongside it tests where the MI peak
sits: the number of distinct gene sets a cell can have is C(G, g), maximal at
g = G/2, which predicts the peak moves right as G grows and does not move with
the number of cells.

    python3 tasks/cells/expression_law/scripts/expression_law.py --dry_run
    python3 tasks/cells/expression_law/scripts/expression_law.py --condition heteromers
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.append('/app')
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch

from src.cells import (expand_gene_set, gene_expression_probs, sample_gene_sets_by_size,
                       size_pmf_for_mean)
from src.config import RunConfig
from src.run import SweepRunner

K_SUB = 5


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", choices=("heteromers", "homomers"), default="heteromers")
    p.add_argument("--n_genes", type=int, nargs="+", default=[3, 5, 8],
                   help="Gene-pool sizes G. The peak of MI vs genes/cell should track G.")
    p.add_argument("--means", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0],
                   help="Target mean genes per cell. Points needing more than G genes are skipped.")
    p.add_argument("--n_cells", type=int, default=30)
    p.add_argument("--size_family", choices=("uniform", "exponential"), default="uniform",
                   help="Law for HOW MANY genes a cell expresses.")
    p.add_argument("--gene_family", choices=("uniform", "exponential"), default="uniform",
                   help="Law for WHICH genes are expressed.")
    p.add_argument("--gene_ratio", type=float, default=1.0,
                   help="P(first gene)/P(last gene) when gene_family='exponential'.")
    p.add_argument("--replicates", type=int, default=5)
    p.add_argument("--replicate_start", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--test_batch_size", type=int, default=4096)
    p.add_argument("--final_batch_size", type=int, default=65536,
                   help="Sampled counting needs a large final batch; see --dry_run output.")
    p.add_argument("--eval_chunk_size", type=int, default=512)
    p.add_argument("--max_pool", type=int, default=4096,
                   help="Refuse points whose receptor pool exceeds this; it is the memory bottleneck.")
    p.add_argument("--base_folder", default="/app/data/expression_law")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args(argv)


def design(args):
    """One row per planned run, with its gene sets already drawn.

    Sampling happens HERE, not inside SingleRunConfig, so the heteromer and homomer
    arms receive byte-identical arrays and differ only in what each cell assembles
    from them.
    """
    rows = []
    for replicate in range(args.replicate_start, args.replicate_start + args.replicates):
        for n_genes in sorted(set(args.n_genes)):
            gene_probs = gene_expression_probs(n_genes, args.gene_family, args.gene_ratio)
            for mean in sorted(set(args.means)):
                if args.size_family == "uniform" and 2 * mean - 1 > n_genes:
                    continue                      # flat support would exceed the pool
                if mean > n_genes:
                    continue
                pmf = size_pmf_for_mean(n_genes, mean, args.size_family)
                seed = args.seed * 1000000 + replicate * 1000 + n_genes * 10 + int(mean * 2)
                sets = sample_gene_sets_by_size(args.n_cells, n_genes, pmf,
                                                gene_probs=gene_probs, seed=seed)
                pool = {r for gs in sets for r in expand_gene_set(gs, K_SUB, True)}
                if len(pool) > args.max_pool:
                    raise ValueError(
                        f"G={n_genes}, mean={mean}: receptor pool {len(pool)} exceeds "
                        f"--max_pool={args.max_pool}. Lower the mean or raise the cap.")
                rows.append({
                    "replicate": replicate, "cell_sampling_seed": seed,
                    "n_genes": n_genes, "n_cells": args.n_cells,
                    "target_mean_genes": mean,
                    "realised_mean_genes": sum(map(len, sets)) / len(sets),
                    "size_family": args.size_family, "gene_family": args.gene_family,
                    "gene_ratio": args.gene_ratio,
                    "cell_gene_sets": sets,
                    "receptor_pool": len(pool),
                    "distinct_gene_sets": len(set(sets)),
                    "genes_represented": len({g for gs in sets for g in gs}),
                })
    if not rows:
        raise ValueError("No design points: every requested mean exceeds its gene pool.")
    return rows


def build_config(args, rows):
    """RunConfig with one zipped axis entry per row.

    entropy='grouped_kt_mi' with measurement 'grouped_counting' is not a style
    choice. Independent sampling makes nearly every cell its own type, so the joint
    count alphabet prod_j(n_j+1) reaches ~10^7 and exact enumeration is impossible.
    Both numbers are printed by --dry_run.
    """
    cells = ({"cell_gene_sets": [r["cell_gene_sets"] for r in rows]}
             if args.condition == "heteromers" else
             {"cell_receptors": [tuple(tuple((g,) * K_SUB for g in genes)
                                       for genes in r["cell_gene_sets"]) for r in rows]})
    measurements = ("grouped_counting", "conditional_entropy_response", "codeword_entropy")
    return RunConfig(
        # --- Environment: matched to the gene_expression task so curves are comparable
        n_families=5, n_ligands=100, latent_dim=6, family_spread=0.1,
        average_family_distance=1.0, environment_geometry="asymmetric",
        distribution_type="uniform", observation_noise_sigma=0.0,
        n_presence_blocks=1, mu_sources=1.0, mu_ligands_per_source=1e-6,
        conc_model_type="lognormal", conc_mean=(0.0,) * 100, conc_std=(1.0,) * 100,
        block_shared_conc_mean=False,

        # --- Physics
        n_genes=[r["n_genes"] for r in rows], k_sub=K_SUB,
        temperature=0.05, initial_temperature=3.0,
        affinity_kernel="gaussian", kernel_params=(1.0,), use_interface_model=True,

        # --- Cells
        **cells,
        n_cells=[r["n_cells"] for r in rows],
        cell_sampling_seed=[r["cell_sampling_seed"] for r in rows],
        cell_stoichiometry="multinomial", cell_readout="mean",
        cell_pool_chunk=128, recompute_backward=True,

        # --- Loss and measurement
        entropy="grouped_kt_mi",
        epochs=args.epochs, lr=1e-2, use_scheduler=False,
        batch_size=args.batch_size, test_batch_size=args.test_batch_size,
        final_test_batch_size=args.final_batch_size,
        eval_chunk_size=args.eval_chunk_size, per_epoch_measure=False,
        measurement_fns=measurements, final_measurement_fns=measurements,

        # --- Sweep
        sweep_name=f"expression_law_{args.condition}_{args.size_family}_{args.gene_family}",
        base_folder=args.base_folder, warm_start=False,
    )


def main(argv=None):
    args = parse_args(argv)
    rows = design(args)
    seed_key = (f"expression_law:{args.condition}:{args.size_family}:{args.gene_family}:"
                f"{args.gene_ratio}:{args.seed}:{args.replicate_start}")
    world_seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:15], 16)

    print(f"expression_law: {args.condition}, size law {args.size_family}, "
          f"gene law {args.gene_family} (ratio {args.gene_ratio})")
    print(f"{len(rows)} independent optimizations, {args.replicates} replicates per point")
    print(f"World seed {world_seed}; no warm start, fresh world per run\n")
    print(f"{'G':>3} {'target':>7} {'realised':>9} {'pool':>6} {'types':>6} {'genes seen':>11} rep")
    for r in rows:
        print(f"{r['n_genes']:>3} {r['target_mean_genes']:>7.2f} "
              f"{r['realised_mean_genes']:>9.2f} {r['receptor_pool']:>6} "
              f"{r['distinct_gene_sets']:>6} {r['genes_represented']:>7}/{r['n_genes']:<3} "
              f"{r['replicate']}")
    print(f"\nlargest receptor pool: {max(r['receptor_pool'] for r in rows)}")
    print(f"most distinct cell types: {max(r['distinct_gene_sets'] for r in rows)}"
          f" of {args.n_cells} cells")
    print("Independent sampling leaves few identical cells, so exact count enumeration\n"
          "is out of reach: training uses the KT bound and evaluation sampled counting.\n"
          f"Both are estimates. Check convergence at {args.final_batch_size} evaluation\n"
          "inputs with analysis/evaluation_budget before trusting small differences.")

    config = build_config(args, rows)
    manifest = {
        "schema": 1, "experiment": "expression_law", "condition": args.condition,
        "size_family": args.size_family, "gene_family": args.gene_family,
        "gene_ratio": args.gene_ratio, "world_seed": world_seed,
        "arguments": vars(args), "rows": rows,
    }
    if args.dry_run:
        return manifest

    torch.manual_seed(world_seed)
    runner = SweepRunner(config)
    (Path(runner.master_logger.sweep_root) / "experiment.json").write_text(
        json.dumps(manifest, indent=2))
    start = time.time()
    runner.execute()
    print(f"\nexpression_law finished in {(time.time() - start) / 3600:.2f} h")
    return manifest


if __name__ == "__main__":
    main()
