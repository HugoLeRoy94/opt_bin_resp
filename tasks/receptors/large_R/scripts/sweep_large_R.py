#!/usr/bin/env python3
"""large_R: homomer arrays with R >> D, at small D.

Tests the large-R expansion of theory.mi_large_R and the counting bounds:
equal norms give at most about D log2 R bits, unequal norms (a ladder) up to
about (D+1) log2 R. The expansion needs R >> D^2, reachable only at small D.

Setting as analytic_baseline: one Gaussian family, gaussian kernel, one ligand
per sniff, homomers, shared and FIXED energies (Emax fixed, Ebase learned),
near-hard readout. lambda is set per D so that rho = sigma sqrt(D)/lambda is the
same at every D.

Axes, zipped: D x conc_std x R x replicates. Cold start only: a warm-start chain
would share one environment across R.

    python3 tasks/receptors/large_R/scripts/sweep_large_R.py --dry-run
    ./run_remote.sh receptors/large_R sweep_large_R.py 0
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

_TASK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TASK / "_lib"))
sys.path.append("/app")
sys.path.append(str(_TASK.parent.parent.parent))   # repo root, for native runs

import large_r as LR                                           # noqa: E402

SIGMA = 1.0
EMAX = 10.0
CONC_MEAN = 0.0
TEMPERATURE = 0.05
N_LIGANDS = 50000
TEST_BATCH = 2 ** 20
D_DEFAULT = [2, 3]
R_DEFAULT = [2, 4, 8, 16, 32, 64]
CONC_STD_DEFAULT = [0.1086, 0.4343, 1.7372]
EPOCHS_PER_SEC = 17.0         # measured on the 2026-10-19 analytic_baseline sweep


def theory_table(Ds, conc_stds, Rs, emax):
    T = LR.T
    for D in Ds:
        lam = LR.lam_for(D, SIGMA)
        print(f"\nD={D}  lambda={lam:.3f}  rho={LR.RHO:.3f}  Emax={emax}  "
              f"kappa_D={T.cell_kappa(D):.2f}")
        print(f"  {'conc_std':>8} {'nu':>6} {'U*':>6} {'sat':>6}   "
              + "".join(f"R={R:<5}" for R in Rs))
        for cs in conc_stds:
            m, sc = T.from_log10(CONC_MEAN, cs)
            p = T.Params(D=D, sigma=SIGMA, lam=lam, Emax=emax, sigma_c=sc, m=m)
            o = T.lambert_optimum(p)
            exp_ = T.mi_large_R(p, o["U_star"], o["Ebase"], np.array(Rs))
            print(f"  {cs:8.4f} {o['nu_min']:6.2f} {o['U_star']:6.2f} {o['sat_prob']:6.3f}   "
                  + "".join(f"{v:<7.1f}" for v in exp_) + "  (large-R expansion)")
        print("  shell bound          " + " " * 6
              + "".join(f"{LR.log2_shell_bound(R, D):<7.1f}" for R in Rs))
        print("  ladder bound         " + " " * 6
              + "".join(f"{LR.log2_ladder_bound(R, D):<7.1f}" for R in Rs))
    print(f"\n  counting ceiling log2 B = {math.log2(TEST_BATCH):.0f} bits, "
          f"ligand pool cap log2(L(R+1)) = {math.log2(N_LIGANDS * (max(Rs) + 1)):.1f} bits")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-D", "--latent-dim", type=int, nargs="+", default=D_DEFAULT)
    ap.add_argument("--R", type=int, nargs="+", default=R_DEFAULT)
    ap.add_argument("--conc-std", type=float, nargs="+", default=CONC_STD_DEFAULT)
    ap.add_argument("--emax", type=float, default=EMAX)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--epochs-per-R", type=int, default=600)
    ap.add_argument("--epochs-base", type=int, default=2000)
    ap.add_argument("--base-folder", default="/app/data/large_R")
    args = ap.parse_args()

    Ds, Rs, stds = args.latent_dim, args.R, args.conc_std
    blk_D = np.repeat(Ds, len(stds) * len(Rs)).tolist()
    blk_cs = np.tile(np.repeat(stds, len(Rs)), len(Ds)).tolist()
    blk_R = Rs * (len(Ds) * len(stds))
    blk_ep = [int(args.epochs_per_R * r + args.epochs_base) for r in blk_R]
    blk_lam = [(LR.lam_for(D, SIGMA),) for D in blk_D]
    n = len(blk_R) * args.replicates
    total = sum(blk_ep) * args.replicates
    print(f"grid: {len(Ds)} D x {len(stds)} conc_std x {len(Rs)} R x "
          f"{args.replicates} replicates = {n} runs, {total} epochs "
          f"(~{total / EPOCHS_PER_SEC / 3600:.0f} h at {EPOCHS_PER_SEC:.0f} epochs/s, "
          f"rough: large R is slower per epoch)")

    try:
        theory_table(Ds, stds, Rs, args.emax)
    except Exception as exc:                                   # pragma: no cover
        print(f"  (theory table unavailable: {exc})")

    if args.dry_run:
        print("\ndry run, nothing launched")
        return

    from src.config import RunConfig                            # noqa: E402
    from src.run import SweepRunner                             # noqa: E402

    rep = args.replicates
    config = RunConfig(
        n_families              = 1,
        n_ligands               = N_LIGANDS,
        latent_dim              = blk_D * rep,
        family_spread           = SIGMA,
        average_family_distance = 1.0,
        environment_geometry    = "asymmetric",
        distribution_type       = "gaussian",
        observation_noise_sigma = 0.0,
        initial_temperature     = "auto",

        n_presence_blocks      = 1,
        mu_sources             = 1,
        mu_ligands_per_source  = 1e-6,
        block_shared_conc_mean = False,

        use_interface_model  = False,
        shared_unit_energies = True,
        fixed_max_energy     = args.emax,

        conc_model_type = "lognormal",
        conc_mean       = CONC_MEAN,
        conc_std        = blk_cs * rep,

        k_sub           = 5,
        temperature     = TEMPERATURE,
        affinity_kernel = "gaussian",
        kernel_params   = blk_lam * rep,

        entropy       = "annealed",
        epochs        = blk_ep * rep,
        lr            = 0.01,
        use_scheduler = False,
        batch_size      = "auto",
        test_batch_size = "auto",
        block_size      = 10,
        measurement_fns = ("full_array_entropy", "codeword_entropy"),
        final_measurement_fns = ("full_array_entropy", "codeword_entropy",
                                 "mutual_information_ligand",
                                 "mutual_information_concentration"),
        final_test_batch_size = TEST_BATCH,
        eval_chunk_size = 4096,

        n_genes     = blk_R * rep,
        sweep_name  = "shell_R",
        base_folder = args.base_folder,
        warm_start  = False,
    )

    print(config)
    t0 = time.time()
    SweepRunner(config).execute()
    h, rem = divmod(time.time() - t0, 3600)
    m, s = divmod(rem, 60)
    print(f"\nlarge_R sweep complete  {int(h)}h {int(m)}m {s:.0f}s")


if __name__ == "__main__":
    main()
