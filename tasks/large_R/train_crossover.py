#!/usr/bin/env python3
"""large_R crossover: one dimension, R from well below D to well above it.

Resolves both regimes on one curve: MI = R eta(nu) for R < D (the closed form
of analytic_baseline), and the counting regime ~ D log2 R for R >> D. R is
stepped by 1 up to past the crossover, then roughly geometrically, since
training cost grows with R.

D = 5 is the compromise between two walls:
  below D, enough points to see the linear regime (R = 2, 3, 4, 5)
  above D, the MI must stay under log2(n_ligands) + log2(R+1), the most a
  finite training set can carry. At D = 5, R = 40 the shell ceiling is about
  20 bits against 21.9 for 10^5 ligands. At D = 6 a factor 8 above D would
  already exceed it.

Same setting as shell_R (train_sweep.py): one Gaussian family, gaussian kernel,
one ligand per sniff, homomers, shared energies with E_max fixed and E_base
learned. lambda = lam_for(D) keeps the geometry ratio of the other sweeps.

Training batch: one ligand per sniff, so the physics tensor carries one ligand
slot and suggest() is called with n_ligands=1. What does scale with n_ligands
is the (B, n_ligands) presence mask and its sampling, a few tens of bytes per
ligand per sample, hence --batch_cap. KT training cannot resolve much more
than log2(batch) bits per pair of samples, so keep the batch large.

Trains and checkpoints only. Measure with test_exact.py.

    python -m tasks.large_R.train_crossover --dry_run
    python -m tasks.large_R.train_crossover
"""
import argparse
import math
import time

import numpy as np

from tasks._lib import large_r as LR
from tasks._lib import theory as T
from train.batching import suggest
from train.config import sweep
from train.sweep import Sweep

SIGMA = 1.0
EMAX = 10.0
CONC_MEAN = 0.0
TEMPERATURE = 0.005
N_LIGANDS = 100_000
K_SUB = 5
D_DEFAULT = 5
R_DEFAULT = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 40]
CONC_STD_DEFAULT = [0.1086, 0.4343, 1.7372]
BATCH_CAP = 8192            # presence mask at 10^5 ligands: roughly 30-40 GB


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-D", "--latent_dim", type=int, default=D_DEFAULT)
    p.add_argument("--R", type=int, nargs="+", default=R_DEFAULT)
    p.add_argument("--conc_std", type=float, nargs="+", default=CONC_STD_DEFAULT)
    p.add_argument("--n_ligands", type=int, default=N_LIGANDS)
    p.add_argument("--emax", type=float, default=EMAX)
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--epochs_per_R", type=int, default=600)
    p.add_argument("--epochs_base", type=int, default=2000)
    p.add_argument("--batch_cap", type=int, default=BATCH_CAP)
    p.add_argument("--data_root", default="data/large_R")
    p.add_argument("--gpu_gb", type=float, default=80.0)
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    D, Rs, stds = a.latent_dim, a.R, a.conc_std
    lam = LR.lam_for(D, SIGMA)
    blk_R = Rs * len(stds)
    blk_cs = np.repeat(stds, len(Rs)).tolist()
    blk_ep = [int(a.epochs_per_R * R + a.epochs_base) for R in blk_R]
    rep = a.replicates
    tiled_R = blk_R * rep
    budget = int(a.gpu_gb * (1 << 30))
    batches = [suggest(R, "kt", n_ligands=1, k_sub=K_SUB, mem_budget_bytes=budget)
               for R in tiled_R]

    print(f"D = {D}, lambda = {lam:.3f}, {a.n_ligands} ligands: "
          f"{len(stds)} conc_std x {len(Rs)} R x {rep} replicates = {len(tiled_R)} runs")
    for cs in stds:
        _, sigma_c = T.from_log10(CONC_MEAN, cs)
        nu = T.lambert_optimum(T.Params(D=D, sigma=SIGMA, lam=lam, Emax=a.emax,
                                        sigma_c=sigma_c))["nu_min"]
        print(f"  conc_std {cs:.4f}  ->  nu = {nu:.3f}")

    runs = sweep(
        dict(
            n_families=1, n_ligands=a.n_ligands, latent_dim=D, family_spread=SIGMA,
            average_family_distance=1.0, distribution_type="gaussian",
            observation_noise_sigma=0.0, initial_temperature="auto",
            n_presence_blocks=1, mu_sources=1, mu_ligands_per_source=1e-6,
            block_shared_conc_mean=False,            shared_unit_energies=True, max_energy=a.emax,
            conc_mean=CONC_MEAN, k_sub=K_SUB, temperature=TEMPERATURE,
            kernel_params=(lam,),
            mi="kt", grouped=False, lr=0.01, per_epoch_measure=False,
            eval_chunk_size=4096,
            measurement_fns=("codeword_entropy",),
        ),
        n_genes=tiled_R,                       # homomers: R = n_genes
        conc_std=blk_cs * rep,
        epochs=blk_ep * rep,
        batch_size=[min(b["batch_size"], a.batch_cap) for b in batches],
        test_batch_size=[min(b["test_batch_size"], a.batch_cap) for b in batches],
    )

    if a.dry_run:
        ligand_cap = math.log2(a.n_ligands)
        print(f"\n  {'R':>3} {'conc_std':>9} {'epochs':>7} {'batch':>6} "
              f"{'shell':>6} {'ladder':>7} {'ligand cap':>11}")
        for run in runs[:len(blk_R)]:
            R = run.n_genes
            print(f"  {R:>3} {run.conc_std:>9.4f} {run.epochs:>7} {run.batch_size:>6} "
                  f"{LR.log2_shell_bound(R, D):>6.1f} {LR.log2_ladder_bound(R, D):>7.1f} "
                  f"{ligand_cap + math.log2(R + 1):>11.1f}")
        print(f"  ... and {len(runs) - len(blk_R)} more (replicates)")
        return

    start = time.time()
    root = Sweep(runs, name="crossover_R", data_root=a.data_root).execute()
    hours, rest = divmod(time.time() - start, 3600)
    minutes, seconds = divmod(rest, 60)
    print(f"\nlarge_R crossover: {len(runs)} runs in {int(hours)}h {int(minutes)}m "
          f"{seconds:.0f}s -> {root}")


if __name__ == "__main__":
    main()
