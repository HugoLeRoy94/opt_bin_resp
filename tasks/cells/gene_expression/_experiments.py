"""Task-local expression designs and sweep manifests; RunConfig stays in each script."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from src.run import SweepRunner


MEASUREMENTS = ("grouped_information", "full_array_entropy",
                "conditional_entropy_response", "codeword_entropy")
BASE_ENVIRONMENT = dict(profile="base", n_ligands=100, latent_dim=6,
                        family_spread=.1, mu_ligands_per_source=1e-6)


def argument_parser(description, default_coverage="complete"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--condition", choices=("heteromers", "homomers"), default="heteromers")
    parser.add_argument("--coverage", choices=("random", "complete"), default=default_coverage)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--replicate_start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="Seed the sweep; recorded in experiment.json.")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--entropy", choices=("grouped_mi", "kt_mi"), default="grouped_mi",
                        help="Training objective; final grouped MI remains the common measurement.")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--test_batch_size", type=int, default=4096)
    parser.add_argument("--final_batch_size", type=int, default=16384)
    parser.add_argument("--eval_chunk_size", type=int, default=512)
    parser.add_argument("--max_states", type=int, default=262144,
                        help="Explicit count-alphabet guard; a (batch, states) table must also fit.")
    parser.add_argument("--base_folder", default="/app/data/gene_expression")
    parser.add_argument("--dry_run", action="store_true", help="Inspect the design without creating or training runs.")
    return parser


def expression_sets(n_genes, n_cells, genes_per_cell, seed, coverage):
    """Random subsets, optionally with one mandatory anchor gene in every cell.

    Anchors cover all genes, with near-balanced multiplicities at g=1. Other
    genes are chosen by random priorities. Reusing the seed makes sets nested
    across g, and identical across strategies, without sharing ligand worlds.
    """
    if not 1 <= genes_per_cell <= n_genes or n_cells < 1:
        raise ValueError("Require positive cells and 1 <= genes_per_cell <= n_genes.")
    if coverage not in {"random", "complete"}:
        raise ValueError("Unknown coverage design.")
    generator = torch.Generator().manual_seed(seed)
    priority = torch.rand(n_cells, n_genes, generator=generator)
    if coverage == "complete":
        if n_cells < n_genes:
            raise ValueError("Complete baseline coverage requires n_cells >= n_genes.")
        permutation = torch.randperm(n_genes, generator=generator)
        anchors = permutation[torch.arange(n_cells) % n_genes]
        anchors = anchors[torch.randperm(n_cells, generator=generator)]
        priority.scatter_(1, anchors[:, None], -1.)
    genes = priority.topk(genes_per_cell, dim=1, largest=False).indices.sort(dim=1).values
    return tuple(map(tuple, genes.tolist()))


def make_rows(args, gene_counts, environments, *, n_cells=None, cells_per_gene=3,
              expression_levels=None):
    """Expand metadata axes before passing their zipped lists to RunConfig."""
    if min(args.replicates, args.epochs, args.batch_size, args.test_batch_size,
           args.final_batch_size, args.eval_chunk_size, args.max_states) < 1:
        raise ValueError("Replicates, epochs, budgets, and max_states must be positive.")
    if min(args.seed, args.replicate_start) < 0:
        raise ValueError("Seeds and replicate_start must be nonnegative.")
    if len(set(gene_counts)) != len(gene_counts) or min(gene_counts) < 2:
        raise ValueError("Choose distinct gene counts >= 2.")
    rows = []
    # These loops construct experiment metadata, not mathematical tensor kernels.
    for replicate in range(args.replicate_start, args.replicate_start + args.replicates):
        seed = args.seed * 100000 + replicate
        for n_genes in gene_counts:
            cells = n_cells if n_cells is not None else cells_per_gene * n_genes
            if cells <= n_genes:
                raise ValueError("These experiments require n_cells > n_genes.")
            for environment in environments:
                for g in (range(1, n_genes + 1) if expression_levels is None else expression_levels):
                    sets = expression_sets(n_genes, cells, g, seed, args.coverage)
                    multiplicities = Counter(sets)
                    states = math.prod(n + 1 for n in multiplicities.values())
                    if states > args.max_states:
                        raise ValueError(f"G={n_genes}, C={cells}, g={g}, replicate={replicate}: "
                                         f"{states} count states exceed --max_states={args.max_states}.")
                    rows.append(dict(environment, replicate=replicate, cell_sampling_seed=seed,
                                     n_genes=n_genes, n_cells=cells, genes_per_cell=g,
                                     cell_gene_sets=sets, count_states=states,
                                     genes_represented=len(set().union(*map(set, sets)))))
    return rows


def cell_axes(rows, condition):
    if condition == "heteromers":
        return dict(cell_gene_sets=[r["cell_gene_sets"] for r in rows])
    return dict(cell_receptors=[tuple(tuple((gene,) * 5 for gene in genes)
                                       for genes in r["cell_gene_sets"]) for r in rows])


def run_key(config):
    """Stable join from persisted scalar configs to planned expression points."""
    g = config.get("genes_per_cell") or len(config["cell_gene_sets"][0])
    return (config["cell_sampling_seed"], config["n_genes"], config["n_cells"], g,
            config["n_ligands"], config["latent_dim"], config["family_spread"],
            config["mu_ligands_per_source"])


def launch(config, args, rows, experiment):
    """One SweepRunner invocation, with a task manifest for independent-run analysis."""
    seed_key = f"{experiment}:{args.condition}:{args.coverage}:{args.seed}:{args.replicate_start}"
    world_seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:15], 16)
    design_fields = ("n_genes", "n_cells", "genes_per_cell", "profile", "n_ligands",
                     "latent_dim", "family_spread", "mu_ligands_per_source")
    points = sorted({tuple(r[k] for k in design_fields) for r in rows})
    # Compare actual RunConfig values, including physics edits made in a script.
    # Remove only deliberate architecture/design differences and bookkeeping.
    ignored = {"cell_gene_sets", "cell_receptors", "cell_sampling_seed", "sweep_name",
               "base_folder", "curation_state", "curation_label"}
    settings = {}
    for key, value in config.to_dict().items():
        if key in ignored:
            continue
        # Repeating a grid with additional seeds does not change its protocol.
        settings[key] = (sorted({json.dumps(v, sort_keys=True) for v in value})
                         if isinstance(value, list) else value)
    protocol = dict(points=points, settings=settings)
    manifest = dict(schema=1, experiment=experiment, condition=args.condition,
                    coverage=args.coverage, world_seed=world_seed, arguments=vars(args),
                    protocol=protocol,
                    protocol_id=hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16],
                    rows=rows)
    print(f"{experiment}: {args.condition}, {args.coverage} coverage, {len(rows)} independent optimizations")
    print(f"World seed: {world_seed}; no warm start or matched worlds; {args.replicates} replicates/point")
    print(f"Largest count alphabet: {max(r['count_states'] for r in rows):,}; "
          f"largest single fp32 (batch,states) table: "
          f"{args.batch_size * max(r['count_states'] for r in rows) * 4 / 2**30:.2f} GiB before gradients")
    if args.dry_run:
        print("G  C  g  profile  count_states  genes_represented  replicate")
        for r in rows:
            print(*(r[k] for k in ("n_genes", "n_cells", "genes_per_cell", "profile",
                                   "count_states", "genes_represented", "replicate")))
        return manifest
    torch.manual_seed(world_seed)
    runner = SweepRunner(config)
    (Path(runner.master_logger.sweep_root) / "experiment.json").write_text(json.dumps(manifest, indent=2))
    start = time.time()
    runner.execute()
    print(f"\n{experiment} finished in {(time.time() - start) / 3600:.2f} h")
    return manifest
