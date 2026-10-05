#!/usr/bin/env python3
"""Final counting test for fig1 het_casc — EVERY environment at one VERY large batch,
reporting the response entropy and mutual information the array reaches at convergence.

Why not the KT bracket: KT holds a (tile, B) pairwise buffer on the GPU and costs
O(B^2), so its test size is capped near 1e5-4e5 sniffs; at R >= 20 the fig1 bracket is
still wide there (e.g. ng10 R50: 20.5 - 21.3 bits). Counting samples ONE binary response
per sniff, packs the R bits into ceil(R/62) int64 keys and merges them into a CPU
frequency table, so memory is 8-16 bytes per sniff and the cost is linear: the budget is
set by the forward pass and by the log2(n) ceiling on a measurable entropy, not by the
GPU. MI = H(Y) - H(Y|X), with H(Y) counted (plug-in and Miller-Madow) and H(Y|X) the
analytic sum of Bernoulli entropies, averaged over the same sniffs.

Budget per run: min(--max_samples, 64 x 2^R), floored at 2^18 sniffs. 64 x the binary
alphabet is where the counting bias measured against exact enumeration
(~3.34 (n / 2^H)^-0.92 bits) drops under 0.1 bit, so small R stays cheap and only the
large arrays pay the cap. The cap defaults to 2**26 = 67,108,864 sniffs, i.e. entropies
up to 26 bits measurable (the widest fig1 KT upper bound is 25.0 at ng15 R75), which
costs under ~9 min of forward pass and ~9 GB of CPU counting memory per run.
Writes <sweep>/test_counting.csv, MERGING with whatever is already there, so the
earlier small-batch rows stay as the convergence trail.

ALWAYS read `response_counting_missing_mass` (Good-Turing f1/n, the mass never sampled)
and `response_counting_log2B` next to the MI: an estimate taken where the missing mass
is large measures the budget, not the array. If it is still high at 2**26, raise
--max_samples (or pass --test_sizes) rather than quoting the number.

Parallelise per n_genes (each is its own sweep folder ng{G}_*):
  ../../run_remote.sh receptors/fig1 test_final.py 0 -- --sweep_glob 'ng2_*'
  ../../run_remote.sh receptors/fig1 test_final.py 1 -- --sweep_glob 'ng3_*'
  ../../run_remote.sh receptors/fig1 test_final.py 2 -- --sweep_glob 'ng5_*'
  ../../run_remote.sh receptors/fig1 test_final.py 3 -- --sweep_glob 'ng7_*'
  ../../run_remote.sh receptors/fig1 test_final.py 0 -- --sweep_glob 'ng10_*'   # then ng15_*
A bigger budget on the sparse, expensive points only:
  ../../run_remote.sh receptors/fig1 test_final.py 0 -- --sweep_glob 'ng15_*' --max_samples 268435456
Use `--measurement kt --largest 16` only to rerun the legacy KT final test (one size =
16 x the run's train batch B, clamped to the KT memory cap; writes test_scaling.csv).
"""
import sys
sys.path.append('/app')

from src import testscaling as ts


def main():
    p = ts.build_parser(__doc__, data_default="/app/data/fig1", sweep_default="ng*")
    # n_test=1 -> the auto ladder degenerates to the single per-run budget, while
    # --test_sizes / --largest / --mult still override it as usual.
    p.set_defaults(measurement="counting", n_test=1, largest=None, seed=0)
    args = p.parse_args()
    ts.run(args.data, args.sweep_glob, ts.sizes_from_args(args),
           n_receptors=args.n_receptors, per_condition=args.per_condition,
           measurement=args.measurement, fwd_chunk=args.fwd_chunk, seed=args.seed)


if __name__ == "__main__":
    main()
