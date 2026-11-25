#!/usr/bin/env python3
"""analytic_baseline, heteromer arm: the homomer R sweep with mixed receptors.

Everything is train_sweep_R.py except the array. Each receptor uses exactly m
distinct genes (m = heteromerization index, m = 1 is the homomer sweep), and
there are as many genes as receptors, n_genes = R. Only the mixing changes, so
the curves overlay on the homomer one at the same nu.

Receptor draw: uniform over the rings of k_sub subunits (rotations identified,
reflections distinct) that contain exactly m distinct genes, without
replacement. Equivalently m genes at random, then one of the n_m ring patterns
on m letters at random. A fresh draw per run, never a shared seed. The array is
stored in config.json as receptor_indices.

The pool C(R, m) n_m can be smaller than R at small R. Those points are skipped
and listed. Where pool = R the array is forced.

One process per m, so the four arms can run in parallel:

    python -m tasks.analytic_baseline.train_sweep_R_heteromer --m 2 --dry_run
    python -m tasks.analytic_baseline.train_sweep_R_heteromer --m 2
    python -m tasks.analytic_baseline.test_exact --prefix heteromer_m2_R
"""
import argparse
import itertools
import math
import time

import numpy as np

from model.arrays import _canonical_rotation, _tier_canonical_forms
from tasks._lib import theory as T
from train.batching import suggest
from train.config import sweep
from train.sweep import Sweep

from tasks.analytic_baseline.train_sweep_R import (
    CONC_MEAN, CONC_STD_DEFAULT, D_DEFAULT, EMAX, K_SUB, LAMBDA, N_LIGANDS,
    R_DEFAULT, SIGMA, TEMPERATURE, theory_table)


def patterns(k_sub, m):
    """Canonical rings over the letters 0..m-1 that use every letter."""
    return sorted(_tier_canonical_forms(m, k_sub, m))


def pool(n_genes, k_sub, m):
    return math.comb(n_genes, m) * len(patterns(k_sub, m))


def draw_array(n_genes, k_sub, m, R, rng):
    """R distinct receptors, uniform over the exactly-m tier.

    Every gene set has the same patterns, so (m genes, one pattern) drawn
    uniformly is uniform over the tier. Distinct draws by rejection, or the
    whole tier when it is no larger than R.
    """
    pats = patterns(k_sub, m)
    if pool(n_genes, k_sub, m) <= R:
        return [list(_canonical_rotation(tuple(genes[c] for c in pat)))
                for genes in itertools.combinations(range(n_genes), m) for pat in pats]
    chosen = set()
    while len(chosen) < R:
        genes = sorted(rng.choice(n_genes, m, replace=False).tolist())
        pat = pats[rng.integers(len(pats))]
        chosen.add(_canonical_rotation(tuple(genes[c] for c in pat)))
    return [list(r) for r in sorted(chosen)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, required=True, help="distinct genes per receptor")
    p.add_argument("-D", "--latent_dim", type=int, default=D_DEFAULT)
    p.add_argument("--lam", type=float, default=LAMBDA)
    p.add_argument("--R", type=int, nargs="+", default=R_DEFAULT)
    p.add_argument("--conc_std", type=float, nargs="+", default=CONC_STD_DEFAULT)
    p.add_argument("--replicates", type=int, default=1,
                   help="independent environments AND arrays per design point")
    p.add_argument("--epochs_per_R", type=int, default=600)
    p.add_argument("--epochs_base", type=int, default=2000)
    p.add_argument("--emax", type=float, default=EMAX)
    p.add_argument("--data_root", default="data/analytic_baseline")
    p.add_argument("--gpu_gb", type=float, default=80.0)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if not 2 <= a.m <= K_SUB:
        raise ValueError(f"--m must lie in [2, {K_SUB}]; m = 1 is train_sweep_R.py")
    D, lam, stds = a.latent_dim, a.lam, a.conc_std
    skipped = [R for R in a.R if pool(R, K_SUB, a.m) < R]
    Rs = [R for R in a.R if R not in skipped]
    if not Rs:
        raise ValueError(f"no R in {a.R} has a pool >= R at m = {a.m}")
    print(f"m = {a.m}, n_genes = R, k_sub = {K_SUB}")
    print(f"grid: {len(Rs)} R x {len(stds)} conc_std x {a.replicates} replicates "
          f"= {len(Rs) * len(stds) * a.replicates} runs")
    if skipped:
        print(f"skipped, pool smaller than R: R = {skipped}")
    print("  " + "  ".join(f"R={R}: pool {pool(R, K_SUB, a.m)}" for R in Rs))
    theory_table(D, lam, stds, a.emax)

    # Same layout as train_sweep_R.py: R inner, conc_std outer, then replicated.
    # Every run, replicates included, gets its own array draw.
    blk_R = Rs * len(stds)
    blk_cs = np.repeat(stds, len(Rs)).tolist()
    blk_ep = [int(a.epochs_per_R * r + a.epochs_base) for r in Rs] * len(stds)
    tiled_R = blk_R * a.replicates
    rng = np.random.default_rng()
    arrays = [draw_array(R, K_SUB, a.m, R, rng) for R in tiled_R]
    budget = int(a.gpu_gb * (1 << 30))
    batches = [suggest(R, "kt", n_ligands=N_LIGANDS, k_sub=K_SUB,
                       mem_budget_bytes=budget) for R in tiled_R]

    runs = sweep(
        dict(
            n_families=1, n_ligands=N_LIGANDS, latent_dim=D,
            family_spread=SIGMA, average_family_distance=1.0,
            distribution_type="gaussian",
            observation_noise_sigma=0.0,
            initial_temperature="auto",
            n_presence_blocks=1, mu_sources=1, mu_ligands_per_source=1e-6,
            block_shared_conc_mean=False,
            shared_unit_energies=True,
            max_energy=a.emax,
            conc_mean=CONC_MEAN,
            k_sub=K_SUB, temperature=TEMPERATURE, kernel_params=(lam,),
            mi="kt", grouped=False, lr=0.01, per_epoch_measure=False,
            eval_chunk_size=4096,
            measurement_fns=("codeword_entropy",),
        ),
        n_genes=tiled_R,
        receptor_indices=arrays,
        conc_std=blk_cs * a.replicates,
        epochs=blk_ep * a.replicates,
        batch_size=[b["batch_size"] for b in batches],
        test_batch_size=[b["test_batch_size"] for b in batches],
    )

    if a.dry_run:
        print(f"\n{len(runs)} runs\n")
        print(f"  {'R':>3} {'conc_std':>9} {'nu':>7} {'epochs':>7} {'batch':>8}  array")
        for run in runs[:len(Rs) * len(stds)]:
            nu = T.nu_from_columns(run.latent_dim, run.family_spread,
                                   run.kernel_params[0], run.max_energy,
                                   run.conc_std)
            print(f"  {run.n_genes:>3} {run.conc_std:>9.4f} {nu:>7.3f} "
                  f"{run.epochs:>7} {run.batch_size:>8}  {run.receptor_indices}")
        print(f"  ... and {len(runs) - len(Rs) * len(stds)} more (replicates)")
        return

    start = time.time()
    root = Sweep(runs, name=f"heteromer_m{a.m}_R", data_root=a.data_root).execute()
    hours, rest = divmod(time.time() - start, 3600)
    minutes, seconds = divmod(rest, 60)
    print(f"\nanalytic_baseline m={a.m}: {len(runs)} runs in {int(hours)}h "
          f"{int(minutes)}m {seconds:.0f}s -> {root}")


if __name__ == "__main__":
    main()
