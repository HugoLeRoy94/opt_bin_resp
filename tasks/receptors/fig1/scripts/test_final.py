#!/usr/bin/env python3
"""Final test measurement for the fig1 het_casc figure — EVERY environment at ONE
(largest feasible) test batch, for the figure's mean ± std at the converged point.

Unlike test_scaling.py (which sweeps sizes on one env per condition to CHECK the plateau),
this measures ALL environment runs at a single size = min(--largest × B, memory cap) — the
largest test batch that fits. Writes into the same <sweep>/test_scaling.csv (merged), so
impact_bracket_variants.py / impact_of_heteromerization_kt.py pick it up as each run's
largest measured size. All logic is in src.testscaling; this only sets fig1 defaults.

Parallelise per n_genes (each is its own sweep folder ng{G}_*):
  ../../run_remote.sh receptors/fig1 test_final.py 0 -- --sweep_glob 'ng2_*'
  ../../run_remote.sh receptors/fig1 test_final.py 1 -- --sweep_glob 'ng3_*'
  ../../run_remote.sh receptors/fig1 test_final.py 2 -- --sweep_glob 'ng5_*'
  ../../run_remote.sh receptors/fig1 test_final.py 3 -- --sweep_glob 'ng7_*'
  ../../run_remote.sh receptors/fig1 test_final.py 0 -- --sweep_glob 'ng10_*'   # then ng15_* ...

--largest defaults to 16 (=16×B, the top of the plateau sweep). The low-ratio runs have the
LARGEST B (batch ∝ 1/√R), so 16×B is biggest and slowest there (KT is O(B²)); lower --largest
to the size where YOUR plateau check flattened if this is too slow.
"""
import sys
sys.path.append('/app')

from src import testscaling as ts


def main():
    p = ts.build_parser(__doc__, data_default="/app/data/fig1", sweep_default="ng*")
    p.set_defaults(largest=16.0)          # single largest feasible size, all envs
    args = p.parse_args()
    ts.run(args.data, args.sweep_glob, ts.sizes_from_args(args),
           n_receptors=args.n_receptors, per_condition=args.per_condition)


if __name__ == "__main__":
    main()
