#!/usr/bin/env python3
"""analytic_baseline, fig1-style arm: more receptors from a fixed gene budget.

Setting of train_sweep_R.py (one Gaussian family, gaussian kernel, one ligand
per sniff, shared E_base, fixed E_max, near-hard readout). One invocation fixes
the ratio R / G and sweeps G = n_genes. Run once per ratio, in parallel. R / G = 1
is the homomer array, already in the homomer_R sweep, which figure_ratio.py reads:

    python -m tasks.analytic_baseline.train_sweep_ratio --ratio 2 --dry_run
    python -m tasks.analytic_baseline.train_sweep_ratio --ratio 2
    python -m tasks.analytic_baseline.test_exact    --prefix ratio
    python -m tasks.analytic_baseline.test_counting --prefix ratio

Array: strategy "cascading", as in fig1. Tiers are filled by number of distinct
genes, homomers first, so R / G = 1 is exactly the homomer array and higher
ratios add 2-gene receptors, then 3-gene ones. Within a tier the receptors are
shuffled with NO seed, so every run, replicates included, draws its own
stoichiometries and gene combinations. Replicates therefore average over the
environment, the stoichiometry and the repertoire. The array is stored in
config.json as receptor_indices.

G values whose R exceeds the pool C(G + k_sub - 1, k_sub) are skipped.
"""
import argparse
import time

import numpy as np

from model.arrays import pool_size
from tasks._lib import theory as T
from tasks.analytic_baseline.train_sweep_R import (
    CONC_MEAN, CONC_STD_DEFAULT, D_DEFAULT, EMAX, K_SUB, LAMBDA, N_LIGANDS,
    SIGMA, TEMPERATURE, theory_table)
from train.batching import suggest
from train.config import sweep
from train.sweep import Sweep

G_DEFAULT = [2, 3, 4, 5, 6, 8]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ratio", type=int, required=True, help="R / n_genes")
    p.add_argument("--G", type=int, nargs="+", default=G_DEFAULT, help="gene counts")
    p.add_argument("-D", "--latent_dim", type=int, default=D_DEFAULT)
    p.add_argument("--lam", type=float, default=LAMBDA)
    p.add_argument("--conc_std", type=float, nargs="+", default=CONC_STD_DEFAULT)
    p.add_argument("--replicates", type=int, default=3,
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
    r, D, lam, stds = a.ratio, a.latent_dim, a.lam, a.conc_std
    Gs = [G for G in a.G if r * G <= pool_size(G, K_SUB)]
    dropped = [G for G in a.G if G not in Gs]
    if not Gs:
        raise SystemExit(f"No G in {a.G} has a pool >= {r} G.")
    print(f"R / G = {r}, G = {Gs}, R = {[r * G for G in Gs]}, k_sub = {K_SUB}")
    if dropped:
        print(f"skipped, pool smaller than R: G = {dropped}")
    print(f"grid: {len(Gs)} G x {len(stds)} conc_std x {a.replicates} replicates "
          f"= {len(Gs) * len(stds) * a.replicates} runs")
    theory_table(D, lam, stds, a.emax)

    # G inner, conc_std outer, then replicated.
    blk_G = Gs * len(stds)
    blk_cs = np.repeat(stds, len(Gs)).tolist()
    tiled_G = blk_G * a.replicates
    tiled_R = [r * G for G in tiled_G]
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
            use_interface_model=False,
            shared_unit_energies=True,
            fixed_max_energy=a.emax,
            conc_mean=CONC_MEAN,
            k_sub=K_SUB, temperature=TEMPERATURE, kernel_params=(lam,),
            receptor_sampling_strategy="cascading",
            receptor_sampling_seed=None,          # fresh array every run
            mi="kt", grouped=False, lr=0.01, per_epoch_measure=False,
            eval_chunk_size=4096,
            measurement_fns=("codeword_entropy",),
        ),
        n_genes=tiled_G,
        n_receptors=tiled_R,
        conc_std=blk_cs * a.replicates,
        epochs=[int(a.epochs_per_R * R + a.epochs_base) for R in tiled_R],
        batch_size=[b["batch_size"] for b in batches],
        test_batch_size=[b["test_batch_size"] for b in batches],
    )

    if a.dry_run:
        print(f"\n{len(runs)} runs\n")
        print(f"  {'G':>3} {'R':>3} {'conc_std':>9} {'nu':>7} {'epochs':>7} "
              f"{'batch':>8}  array")
        for run in runs[:len(Gs) * len(stds)]:
            nu = T.nu_from_columns(run.latent_dim, run.family_spread,
                                   run.kernel_params[0], run.fixed_max_energy,
                                   run.conc_std)
            print(f"  {run.n_genes:>3} {run.n_receptors:>3} "
                  f"{run.conc_std:>9.4f} {nu:>7.3f} {run.epochs:>7} "
                  f"{run.batch_size:>8}  {run.receptor_indices}")
        print(f"  ... and {len(runs) - len(Gs) * len(stds)} more (replicates)")
        return

    start = time.time()
    root = Sweep(runs, name=f"ratio{r}", data_root=a.data_root).execute()
    hours, rest = divmod(time.time() - start, 3600)
    minutes, seconds = divmod(rest, 60)
    print(f"\nanalytic_baseline R/G={r}: {len(runs)} runs in {int(hours)}h "
          f"{int(minutes)}m {seconds:.0f}s -> {root}")


if __name__ == "__main__":
    main()
