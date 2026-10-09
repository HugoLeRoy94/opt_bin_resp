#!/usr/bin/env python3
"""Final counting test for fig1 het_casc — grow the batch per environment until the
response entropy and mutual information stop depending on it.

Why not the KT bracket: KT holds a (tile, B) pairwise buffer on the GPU and costs
O(B^2), so its test size is capped near 1e6; measured on ng10 at 2^26 sniffs, counting
MI already beats the KT lower bound at every R (16.23 vs 15.09 at R20, 19.02 vs 17.18
at R30, 18.22 vs 17.20 at R40, 17.32 vs 17.12 at R50). Counting samples ONE binary
response per sniff, packs the R bits into ceil(R/62) int64 keys and merges them into a
CPU frequency table, so the budget is set by the forward pass, not by the GPU.
MI = H(Y) - H(Y|X), with H(Y) counted (plug-in and Miller-Madow) and H(Y|X) the analytic
sum of Bernoulli entropies over the same sniffs.

ADAPTIVE BUDGET. There is no budget that fits every condition: counting needs about
46 x 2^H(Y) sniffs, and H(Y) = MI + H(Y|X) is what we are trying to measure. R=10
converges at 16k sniffs (it cannot exceed 10 bits), while R=50 at ng10 was still
climbing 1 bit per 4x at 2^26. So the script grows one stream in x4 batches from
--start_samples and stops that run when

  * the Good-Turing missing mass drops below --missing_target (converged), or
  * a x4 batch buys less than 0.02 bit (converged), or
  * the next batch's counting table is projected past --max_memory_gb (refused), or
  * --max_samples is reached.

The batches are nested, so the whole ladder costs its last batch. Every batch is written to
<sweep>/test_counting.csv, giving the convergence trail for free; the value to plot is
the largest test_size per run, and its `response_counting_missing_mass` says how much
to trust it.

Defaults are sized from the measured throughput of about 2e5 sniffs/s: --max_samples
2**30 (1.07e9) is ~1.5 h for one unconverged run, and --max_memory_gb 32 will in
practice stop R>=40 one or two batches earlier. Check the node with `free -g` and raise
--max_memory_gb if it has the room, that is the knob that binds first.

Starts in --per_condition mode (the FIRST environment of each (n_genes, R)), which is
the cheap survey; pass --all_runs for the 5 repeat environments once the budget per
condition is known.

  ../../run_remote.sh receptors/fig1 test_final.py 0 -- --sweep_glob 'ng2_*'
  ../../run_remote.sh receptors/fig1 test_final.py 1 -- --sweep_glob 'ng3_*'
  ../../run_remote.sh receptors/fig1 test_final.py 2 -- --sweep_glob 'ng5_*'
  ../../run_remote.sh receptors/fig1 test_final.py 3 -- --sweep_glob 'ng7_*'
  ../../run_remote.sh receptors/fig1 test_final.py 0 -- --sweep_glob 'ng10_*'   # then ng15_*
Only the expensive arrays, with more room:
  ../../run_remote.sh receptors/fig1 test_final.py 0 -- --sweep_glob 'ng15_*' \
      --n_receptors 45 60 75 --max_memory_gb 128 --max_samples 4294967296
Use `--measurement kt --largest 16` to rerun the legacy KT final test (writes
test_scaling.csv).
"""
import sys
sys.path.append('/app')

from src import testscaling as ts


def main():
    p = ts.build_parser(__doc__, data_default="/app/data/fig1", sweep_default="ng*")
    p.add_argument("--all_runs", action="store_true",
                   help="measure every repeat environment, not just one per (n_genes, R)")
    p.set_defaults(measurement="counting", max_samples=1 << 30, seed=0)
    args = p.parse_args()
    # Explicit sizes mean the caller wants exactly those, so the stop rule is dropped.
    explicit = bool(args.test_sizes or args.mult or args.largest)
    ts.run(args.data, args.sweep_glob, ts.sizes_from_args(args),
           n_receptors=args.n_receptors, per_condition=not args.all_runs,
           measurement=args.measurement, fwd_chunk=args.fwd_chunk, seed=args.seed,
           stop=None if explicit or args.measurement == "kt" else
           ts.convergence_stop(args.missing_target, memory_gb=args.max_memory_gb))


if __name__ == "__main__":
    main()
