#!/usr/bin/env python3
"""Final stochastic-response test for fig1 het_casc — EVERY environment at one large
batch, reporting both the response entropy and mutual information at convergence.

Unlike test_scaling.py (which keeps the KT bracket sweep for plateau checks), this
streams ALL environment runs at one size = --largest × B. It samples one binary response
per sniff, counts joint response codes, and writes `<sweep>/test_counting.csv`.  The CSV
contains `response_entropy_plugin` / `response_entropy_mm` and the matching
`mutual_information_counting_plugin` / `mutual_information_counting_mm`, as well as
H(Y|X) and code-coverage diagnostics.  Counting does not use the KT memory cap or its
quadratic pairwise work; outputs are retained on CPU for the final unique-row count.

Use `--measurement kt` only when intentionally rerunning the legacy KT final test; it
writes `test_scaling.csv` so existing KT analysis continues to work unchanged.

Parallelise per n_genes (each is its own sweep folder ng{G}_*):
  ../../run_remote.sh receptors/fig1 test_final.py 0 -- --sweep_glob 'ng2_*'
  ../../run_remote.sh receptors/fig1 test_final.py 1 -- --sweep_glob 'ng3_*'
  ../../run_remote.sh receptors/fig1 test_final.py 2 -- --sweep_glob 'ng5_*'
  ../../run_remote.sh receptors/fig1 test_final.py 3 -- --sweep_glob 'ng7_*'
  ../../run_remote.sh receptors/fig1 test_final.py 0 -- --sweep_glob 'ng10_*'   # then ng15_* ...

--largest defaults to 16 (=16×B). Increase it for counting after checking the
Miller–Madow estimate and distinct-code fraction have stabilized; it is no longer
limited by KT's quadratic cost, though CPU memory still scales with samples × receptors.
"""
import sys
sys.path.append('/app')

from src import testscaling as ts


def main():
    p = ts.build_parser(__doc__, data_default="/app/data/fig1", sweep_default="ng*")
    p.set_defaults(largest=16.0, measurement="counting")
    args = p.parse_args()
    ts.run(args.data, args.sweep_glob, ts.sizes_from_args(args),
           n_receptors=args.n_receptors, per_condition=args.per_condition,
           measurement=args.measurement)


if __name__ == "__main__":
    main()
