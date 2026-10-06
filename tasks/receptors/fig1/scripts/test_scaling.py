#!/usr/bin/env python3
"""Post-hoc test-size scaling for the fig1 het_casc runs — plateau check.

Reloads each run's frozen env (best_model.pt) and re-measures ONE estimator at a ladder
of test sizes, to see whether its value has stopped moving with the budget. Writes
<sweep>/test_scaling.csv (KT, read by analysis/test_scaling_kt.py) or
<sweep>/test_counting.csv (counting). All logic is in src.testscaling; this only sets
fig1 defaults.

KT (default): sizes = multiples of that run's TRAIN batch B (1,2,4,8,16 -> B..16B; 4B is
what the figure's final bracket uses). The ladder stops at the KT memory cap, since KT
holds a (tile, B) pairwise buffer on the GPU and costs O(B^2).

Counting: the ladder is ABSOLUTE and NESTED, x4 rungs upwards from --start_samples
(default 2^20), because the counting budget has nothing to do with the train batch and
a rung reuses the sniffs of the one below it. Every requested rung is measured, with no
early stop: this is the diagnostic you run to SEE the curve, while test_final.py grows
the same stream adaptively and stops when it has converged.
  ../../run_remote.sh receptors/fig1 test_scaling.py 0 -- --measurement counting \
      --sweep_glob 'ng10_*' --per_condition --max_samples 268435456
Read the curve together with `response_counting_missing_mass`: a rung whose entropy is
still climbing and whose missing mass is large is reporting its budget, not the array.

Parallelise per n_genes (each n_genes is its own sweep folder ng{G}_*):
  ../../run_remote.sh receptors/fig1 test_scaling.py 0 -- --sweep_glob 'ng2_*'
  ../../run_remote.sh receptors/fig1 test_scaling.py 1 -- --sweep_glob 'ng3_*'
  ../../run_remote.sh receptors/fig1 test_scaling.py 2 -- --sweep_glob 'ng5_*'
  ../../run_remote.sh receptors/fig1 test_scaling.py 3 -- --sweep_glob 'ng7_*'
  ../../run_remote.sh receptors/fig1 test_scaling.py 0 -- --sweep_glob 'ng10_*'
Add --per_condition to measure one env per (n_genes, R) (much faster); override sizes
with --mult or --test_sizes.
"""
import sys
sys.path.append('/app')

from src import testscaling as ts


def main():
    p = ts.build_parser(__doc__, data_default="/app/data/fig1", sweep_default="ng*")
    p.set_defaults(seed=0)
    args = p.parse_args()
    if args.measurement == "kt" and not args.mult and not args.test_sizes:
        args.mult = [1, 2, 4, 8, 16]      # counting instead uses the absolute ladder
    # No stop= : every requested rung is measured. Seeing the curve IS the point.
    ts.run(args.data, args.sweep_glob, ts.sizes_from_args(args),
           n_receptors=args.n_receptors, per_condition=args.per_condition,
           measurement=args.measurement, fwd_chunk=args.fwd_chunk, seed=args.seed)


if __name__ == "__main__":
    main()
