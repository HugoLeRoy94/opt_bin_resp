#!/usr/bin/env python3
"""Split the exact-vs-KT gap into estimator bias and optimization loss. No retraining.

The two replicates sweeps differ in TWO things at once: what they trained on
(grouped_mi against grouped_kt_mi) and what they measured with (exact enumeration
against sampled counting). Comparing their reported MI therefore cannot say which
one is responsible.

This re-measures BOTH sets of saved checkpoints with BOTH estimators, filling in a
2x2 per design point:

                        eval exact        eval counting
    train exact           A                     B
    train KT              C                     D

    estimator bias    = A - B     what sampled counting loses at this budget
    optimization loss = A - C     what training on the KT bound gives up
    reported gap      = A - D     the number the two sweeps actually print

Both sweeps share a world seed, gene sets, batch size and epoch count, so the
comparison is paired: only the objective differs.

    python3 tasks/cells/gene_expression/scripts/estimator_crosscheck.py \\
        --exact_sweep  data/gene_expression/replicates_heteromers_complete_20260923_112537 \\
        --kt_sweep     data/gene_expression/replicates_heteromers_complete_20260923_135950 \\
        --budgets 16384 65536 262144
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append('/app')
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch

from src.IO import SweepLoader
from tasks.cells.gene_expression.scripts.evaluation_budget import (
    MI_KEY, evaluate_budgets, prepare_run)


def index_sweep(sweep_dir):
    """Map (replicate seed, genes per cell) -> run directory."""
    runs = {}
    for cfg, run_dir in SweepLoader(str(sweep_dir)).iter_run_dirs():
        if not (Path(run_dir) / "best_model.pt").exists():
            continue
        sizes = {len(genes) for genes in cfg.cell_gene_sets}
        if len(sizes) != 1:
            raise ValueError(f"Cells express different numbers of genes in {run_dir}")
        runs[(cfg.cell_sampling_seed, sizes.pop())] = run_dir
    return runs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exact_sweep", required=True, help="sweep trained with grouped_mi")
    p.add_argument("--kt_sweep", required=True, help="sweep trained with grouped_kt_mi")
    p.add_argument("--budgets", nargs="+", type=int, default=[16384, 65536, 262144],
                   help="Evaluation inputs. The exact estimator is unbiased in these; "
                        "only the counting columns should move.")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--chunk_size", type=int, default=256,
                   help="Exact enumeration holds a (chunk, n_states) table; lower this if it "
                        "does not fit. n_states reaches ~460k for this design.")
    p.add_argument("--pool_chunk", type=int, default=128)
    p.add_argument("--max_states", type=int, default=1048576)
    p.add_argument("--genes_per_cell", nargs="+", type=int, default=None,
                   help="Restrict to these expression levels (default: all present).")
    p.add_argument("--replicates", nargs="+", type=int, default=None,
                   help="Restrict to these cell_sampling_seed values.")
    p.add_argument("--device", choices=("cpu", "cuda"), default=None)
    p.add_argument("--out", default=None, help="Output JSON (default: next to the KT sweep).")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    trained = {"exact": index_sweep(args.exact_sweep), "KT": index_sweep(args.kt_sweep)}
    shared = sorted(set(trained["exact"]) & set(trained["KT"]))
    if args.genes_per_cell:
        shared = [k for k in shared if k[1] in set(args.genes_per_cell)]
    if args.replicates:
        shared = [k for k in shared if k[0] in set(args.replicates)]
    if not shared:
        raise SystemExit("No design point is present in BOTH sweeps with a saved model.")

    only_exact = len(set(trained["exact"]) - set(trained["KT"]))
    only_kt = len(set(trained["KT"]) - set(trained["exact"]))
    print(f"{len(shared)} matched design points "
          f"({only_exact} only in the exact sweep, {only_kt} only in the KT sweep)")
    print(f"4 measurements each (2 training objectives x 2 estimators), "
          f"budgets {args.budgets}, {args.repeats} repeats -> "
          f"{len(shared) * 4 * args.repeats} evaluations on {device}\n")
    if args.dry_run:
        for seed, g in shared:
            print(f"  seed={seed} genes/cell={g}")
        return []

    records = []
    for seed, g in shared:
        for training, runs in trained.items():
            for estimator in ("exact", "counting"):
                env, physics, ri, readout, est, cfg = prepare_run(
                    runs[(seed, g)], device, estimator, args.max_states)
                rows = evaluate_budgets(env, physics, ri, readout, est, args.budgets,
                                        args.repeats, args.seed, args.chunk_size,
                                        args.pool_chunk)
                for row in rows:
                    records.append(dict(
                        cell_sampling_seed=seed, genes_per_cell=g,
                        trained_on=training, evaluated_with=estimator,
                        mi=row[MI_KEY[estimator]], samples=row["samples"],
                        repeat=row["repeat"], run_dir=str(runs[(seed, g)]),
                        unique_fraction=row.get("grouped_counting_unique_fraction")))
                print(f"  seed={seed} g={g} trained={training:5} eval={estimator:8} "
                      + "  ".join(f"B={r['samples']}: {r[MI_KEY[estimator]]:.3f}" for r in rows))
                del env, physics, readout, est
                if device == "cuda":
                    torch.cuda.empty_cache()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else Path(args.kt_sweep) / f"estimator_crosscheck_{stamp}.json"
    out.write_text(json.dumps(dict(
        schema=1, arguments=vars(args),
        exact_sweep=str(args.exact_sweep), kt_sweep=str(args.kt_sweep),
        records=records), indent=2))
    print(f"\nSaved {out}")
    print("Read it with analysis/estimator_crosscheck.py")
    return records


if __name__ == "__main__":
    main()
