#!/usr/bin/env python3
"""Post-hoc test-size scaling — re-measure the KT bracket at chosen TEST sizes.

Reloads a sweep's frozen envs (best_model.pt) and re-evaluates the KT lower+upper entropy
on a set of test batch sizes, WITHOUT retraining — to check the test-entropy-vs-batch-size
curve plateaus. All logic is in src.testscaling; this only sets optimizer/sample_limit
defaults. Task-agnostic via --data/--sweep_glob (fig1 has its own tasks/fig1/scripts version).

Sizes: --test_sizes (explicit), --mult (per-run multiples of train B), else an auto ×4
ladder of --n_test points. Re-running MERGES the CSV, so you can add sizes incrementally.

Run on the cluster (GPU):
  ../../run_remote.sh receptors/optimizer test_scaling.py 0
  ../../run_remote.sh receptors/optimizer test_scaling.py 0 -- --sweep_glob 'sample_limit_*' --n_test 6
"""
import sys
sys.path.append('/app')

from src import testscaling as ts


def main():
    p = ts.build_parser(__doc__, data_default="/app/data/optimizer",
                        sweep_default="sample_limit_*")
    args = p.parse_args()
    ts.run(args.data, args.sweep_glob, ts.sizes_from_args(args),
           n_receptors=args.n_receptors, per_condition=args.per_condition)


if __name__ == "__main__":
    main()
