#!/usr/bin/env python3
"""analytic_baseline measurement: how the information splits between ligand and concentration.

Hard threshold and no observation noise, so H(Y | l, s) = 0 and H(Y) = I(Y; l, s),
with l the ligand position and s = ln c. This pass measures what is left once
one of the two is known:

    entropy_given_ligand   H(Y | l), closed form per l, Monte Carlo over l
    entropy_given_conc     H(Y | s), subset simulation as in test_exact

and the figure forms I(Y; l) = H(Y) - H(Y | l) and I(Y; s) = H(Y) - H(Y | s)
with entropy_exact from test_exact.py. Continuous ligand cloud, so "ligand
identity" is the position l, not an index into the finite training pool.
See estimate/exact.py. Default prefixes: the curves of figure_ratio.py.

    python -m tasks.analytic_baseline.test_channels
"""
import argparse

from tasks._lib.measure import measure_channels_goal

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="data/analytic_baseline")
    p.add_argument("--prefix", nargs="+", default=["homomer_R", "ratio"])
    p.add_argument("--n_ligand", type=int, default=2 ** 16, help="ligand positions averaged")
    p.add_argument("--n_conc", type=int, default=4096, help="codewords averaged for H(Y|s)")
    p.add_argument("--n_chains", type=int, default=512, help="samples per codeword integral")
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    measure_channels_goal(a.data_root, a.prefix, a.n_ligand, a.n_conc, a.n_chains,
                          a.device, a.overwrite)
