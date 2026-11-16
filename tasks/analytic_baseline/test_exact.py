#!/usr/bin/env python3
"""analytic_baseline measurement: exact codeword entropy for an infinite ligand cloud.

The trained receptor positions are evaluated against the continuous Gaussian
cloud the ligands were drawn from, not the finite training set, so the
log2(n_ligands) + log2(R+1) ceiling of that set does not apply. Each drawn
codeword's probability is integrated rather than counted: the only error is
statistical, entropy_exact_sem, and it does not grow with H. Hard threshold,
so H(Y|X) = 0 and the entropy is the MI. See estimate/exact.py.

    python -m tasks.analytic_baseline.test_exact
"""
import argparse

from tasks._lib.measure import measure_exact_goal

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="data/analytic_baseline")
    p.add_argument("--prefix", nargs="+", default=["homomer_R"])
    p.add_argument("--n_samples", type=int, default=4096, help="codewords averaged")
    p.add_argument("--n_chains", type=int, default=512, help="samples per codeword integral")
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    measure_exact_goal(a.data_root, a.prefix, a.n_samples, a.n_chains, a.device, a.overwrite)
