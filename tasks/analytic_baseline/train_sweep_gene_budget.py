#!/usr/bin/env python3
"""analytic_baseline, gene budget arm: does heteromerization raise the information at a fixed number of genes?

Same setting as analytic_baseline (one Gaussian family, gaussian kernel, one
ligand per sniff, shared E_base, fixed E_max, near-hard readout). The budget is
G = n_genes. Homomers give R = G receptors. Heteromers give R = c G receptors
from the same G genes, each receptor using exactly m distinct genes.

    G   genes, the x axis
    c   receptors per gene, --receptors_per_gene
    m   distinct genes per receptor (heteromerization index), one process per m

The homomer reference (m = 1, R = G) is the analytic_baseline homomer_R sweep,
which figure_gene_budget.py reads directly. Its R grid also gives homomers
with c G genes, the unconstrained array of the same size.

Receptor draw: R distinct receptors, uniform over the multisets of k_sub
subunits with exactly m distinct genes. Each run, replicates included, gets a
fresh environment AND a fresh array, so replicates average over the environment,
the stoichiometry and which genes each receptor uses. The array is stored in
config.json as receptor_indices.

G values whose pool C(G, m) C(k_sub-1, m-1) is smaller than c G are skipped.

    python -m tasks.analytic_baseline.train_sweep_gene_budget --m 2 --dry_run
    python -m tasks.analytic_baseline.train_sweep_gene_budget --m 2
    python -m tasks.analytic_baseline.test_exact    --prefix gene_budget
    python -m tasks.analytic_baseline.test_counting --prefix gene_budget
"""
import argparse
import time

import numpy as np

from tasks._lib import theory as T
from tasks.analytic_baseline.train_sweep_R import (
    CONC_MEAN, CONC_STD_DEFAULT, D_DEFAULT, EMAX, K_SUB, LAMBDA, N_LIGANDS,
    SIGMA, TEMPERATURE, theory_table)
from tasks.analytic_baseline.train_sweep_R_heteromer import draw_array, pool
from train.batching import suggest
from train.config import sweep
from train.sweep import Sweep

G_DEFAULT = [2, 3, 4, 5, 6, 8]
C_DEFAULT = 2.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, required=True, help="distinct genes per receptor")
    p.add_argument("--G", type=int, nargs="+", default=G_DEFAULT, help="gene budgets")
    p.add_argument("--receptors_per_gene", type=float, default=C_DEFAULT,
                   help="c, so that R = round(c G)")
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
    if not 2 <= a.m <= K_SUB:
        raise ValueError(f"--m must lie in [2, {K_SUB}]; m = 1 is analytic_baseline")
    c, D, lam, stds = a.receptors_per_gene, a.latent_dim, a.lam, a.conc_std
    R_of = {G: int(round(c * G)) for G in a.G}
    skipped = [G for G in a.G if pool(G, K_SUB, a.m) < R_of[G]]
    Gs = [G for G in a.G if G not in skipped]
    if not Gs:
        raise ValueError(f"no G in {a.G} has a pool >= {c} G at m = {a.m}")
    print(f"m = {a.m}, R = round({c:g} G), k_sub = {K_SUB}")
    print(f"grid: {len(Gs)} G x {len(stds)} conc_std x {a.replicates} replicates "
          f"= {len(Gs) * len(stds) * a.replicates} runs")
    if skipped:
        print(f"skipped, pool smaller than R: G = {skipped}")
    print("  " + "  ".join(f"G={G}: R {R_of[G]}, pool {pool(G, K_SUB, a.m)}" for G in Gs))
    theory_table(D, lam, stds, a.emax)

    # G inner, conc_std outer, then replicated. Every run gets its own array draw.
    blk_G = Gs * len(stds)
    blk_cs = np.repeat(stds, len(Gs)).tolist()
    tiled_G = blk_G * a.replicates
    tiled_R = [R_of[G] for G in tiled_G]
    # Epochs and batch follow R, as in analytic_baseline: the array size sets the
    # cost and, conservatively, the convergence time.
    epochs = [int(a.epochs_per_R * R + a.epochs_base) for R in tiled_R]
    rng = np.random.default_rng()
    arrays = [draw_array(G, K_SUB, a.m, R_of[G], rng) for G in tiled_G]
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
            mi="kt", grouped=False, lr=0.01, per_epoch_measure=False,
            eval_chunk_size=4096,
            measurement_fns=("codeword_entropy",),
        ),
        n_genes=tiled_G,
        receptor_indices=arrays,
        conc_std=blk_cs * a.replicates,
        epochs=epochs,
        batch_size=[b["batch_size"] for b in batches],
        test_batch_size=[b["test_batch_size"] for b in batches],
    )

    if a.dry_run:
        print(f"\n{len(runs)} runs\n")
        print(f"  {'G':>3} {'R':>3} {'conc_std':>9} {'nu':>7} {'epochs':>7} {'batch':>8}  array")
        for run in runs[:len(Gs) * len(stds)]:
            nu = T.nu_from_columns(run.latent_dim, run.family_spread,
                                   run.kernel_params[0], run.fixed_max_energy,
                                   run.conc_std)
            print(f"  {run.n_genes:>3} {len(run.receptor_indices):>3} "
                  f"{run.conc_std:>9.4f} {nu:>7.3f} {run.epochs:>7} "
                  f"{run.batch_size:>8}  {run.receptor_indices}")
        print(f"  ... and {len(runs) - len(Gs) * len(stds)} more (replicates)")
        return

    start = time.time()
    root = Sweep(runs, name=f"gene_budget_m{a.m}_c{c:g}", data_root=a.data_root).execute()
    hours, rest = divmod(time.time() - start, 3600)
    minutes, seconds = divmod(rest, 60)
    print(f"\nanalytic_baseline gene budget m={a.m}: {len(runs)} runs in {int(hours)}h "
          f"{int(minutes)}m {seconds:.0f}s -> {root}")


if __name__ == "__main__":
    main()
