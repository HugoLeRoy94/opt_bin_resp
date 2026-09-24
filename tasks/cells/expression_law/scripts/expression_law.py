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
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.append('/app')
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch

from src.cells import (expand_gene_set, gene_expression_probs, sample_gene_sets_by_size,
                       size_pmf_for_mean)
from src.config import RunConfig
from src.run import SweepRunner

K_SUB = 5
# Measured on the G=5, C=30 arrays: H(Y|X)/C = 0.441 bits per cell, MI ~ 6 bits.
CELL_FIRING_P = 0.0905
TYPICAL_MI = 6.0
# pool-weighted forward inputs per second, from the measured KT sweep
CLUSTER_RATE = 1.08e8


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
    p.add_argument("--final_batch_size", type=int, default=16777216,
                   help="Inputs per final measurement. Sampled counting is biased DOWN and the "
                        "bias falls only with this number; --dry_run projects it.")
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


def _binomial_entropy(n, p):
    terms = [math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]
    return -sum(q * math.log2(q) for q in terms if q > 0)


def report_expected_bias(args, rows):
    """Project the counting bias per design point, before committing the sweep.

    tasks/cells/gene_expression/scripts/estimator_crosscheck.py measured the bias of
    sampled counting against exact enumeration over 5 expression levels x 3 budgets
    of the G=5, C=30 arrays. Across those 12 points,

        bias [bits] ~= 3.34 * (B / 2^H(K)) ** -0.92

    so it is the ratio of budget to EFFECTIVE ALPHABET 2^H(K) that matters, not the
    budget alone. H(K) = MI + H(K|X), and H(K|X) is the sum over groups of a
    binomial entropy, which this computes from the gene sets actually drawn. The
    per-cell firing noise and MI are taken from the measured G=5, C=30 arrays, so
    the projection is an extrapolation and not a guarantee: the dry run is for
    sizing the budget, the plotted Good-Turing missing mass is the real check.
    """
    print(f"\nFinal measurement: one pass over {args.final_batch_size:,} inputs per run.")
    print("Not repeated: repeats show evaluation noise and cannot reduce a bias that\n"
          "falls only with the batch size.\n")
    print(f"{'G':>3} {'mean':>5} {'groups':>7} {'H(K|X)':>8} {'H(K)~':>7} "
          f"{'bias at B':>11} {'B for 0.1 bits':>16}")
    worst = 0.0
    for row in rows:
        if row["replicate"] != rows[0]["replicate"]:
            continue                       # the design repeats across replicates
        sizes = Counter(row["cell_gene_sets"]).values()
        h_kx = sum(_binomial_entropy(n, CELL_FIRING_P) for n in sizes)
        h_k = TYPICAL_MI + h_kx
        bias = 3.34 * (args.final_batch_size / 2 ** h_k) ** -0.92
        worst = max(worst, bias)
        print(f"{row['n_genes']:>3} {row['target_mean_genes']:>5.1f} "
              f"{len(list(sizes)):>7} {h_kx:>8.1f} {h_k:>7.1f} {bias:>10.2f}b "
              f"{46 * 2 ** h_k:>16,.0f}")
    # Cost is the physics forward pass, linear in the receptor pool. CLUSTER_RATE is
    # calibrated on the measured gene_expression KT sweep: 25 runs in 1.26 h at mean
    # R_pool 331, 5000 epochs of batch 4096, final measurement 163,840 inputs. A
    # backward pass counts as two extra forwards.
    seconds = sum(
        r["receptor_pool"] * (args.epochs * args.batch_size * 3 + args.final_batch_size)
        for r in rows) / CLUSTER_RATE
    train_share = args.epochs * args.batch_size * 3
    print(f"\nProjected cost for this sweep: {seconds / 3600:.1f} h for one condition, "
          f"{2 * seconds / 3600:.1f} h for both.")
    print(f"  training is {100 * train_share / (train_share + args.final_batch_size):.0f}% "
          f"of it, the final measurement "
          f"{100 * args.final_batch_size / (train_share + args.final_batch_size):.0f}%.")
    print(f"  (calibrated on the measured gene_expression KT sweep, 3.0 min/run; "
          f"scale if the hardware differs)")
    print(f"Symbol counting is linear and negligible: "
          f"{args.final_batch_size * 8 / 2 ** 20:.0f} MiB of int64 keys per run.")
    # The other ceiling, which the missing mass says nothing about. KT measures
    # information about the EMPIRICAL distribution of the training batch, whose
    # entropy is log2(batch_size), so the objective cannot express more than that
    # however large the array is. Compare train_mutual_information in stats.csv
    # against this line after the run; approaching it means the TRAINING batch is
    # the limit, and no evaluation diagnostic will reveal it.
    print(f"\nTraining ceiling: the KT objective cannot exceed "
          f"log2(batch_size) = {math.log2(args.batch_size):.0f} bits. "
          f"Expect MI near {TYPICAL_MI:.0f}.")
    print(f"\nWorst projected bias: {worst:.2f} bits.")
    if worst > 0.2:
        print("Above ~0.2 bits the curve shape is distorted, because the bias differs\n"
              "between design points. Either raise --final_batch_size or lower --n_cells:\n"
              "H(K|X) is a sum over cells, so 2^H(K) grows close to exponentially in C\n"
              "while an affordable budget grows linearly. Grouping only blunts this once\n"
              "cells start duplicating, which needs C well above 2^G - 1.")


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
          "is out of reach: training uses the KT bound and evaluation sampled counting.")
    report_expected_bias(args, rows)

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
