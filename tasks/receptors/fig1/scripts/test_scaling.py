#!/usr/bin/env python3
"""Post-hoc test-size scaling for the fig1 het_casc KT runs — plateau check.

Reloads each run's frozen env (best_model.pt) and re-measures the KT bracket at test
sizes = multiples of that run's TRAIN batch B (default 1,2,4,8,16 → B,2B,4B,8B,16B; 4B is
the size the figure's final bracket already uses). Writes <sweep>/test_scaling.csv, read
by analysis/test_scaling_kt.py. All logic is in src.testscaling; this only sets fig1
defaults.

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
    p = ts.build_parser(__doc__, data_default="/app/data/fig1",
                        sweep_default="ng*", mult_default=[1, 2, 4, 8, 16])
    args = p.parse_args()
    ts.run(args.data, args.sweep_glob, ts.sizes_from_args(args),
           n_receptors=args.n_receptors, per_condition=args.per_condition,
           measurement=args.measurement)


if __name__ == "__main__":
    main()
