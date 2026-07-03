#!/usr/bin/env python3
"""Heteromers, cascading receptor-sampling strategy — parametrized over n_genes.

Replaces the het_casc_ng{2,3,5,7,10,15,20,25,30,35}.py family: the environment,
physics and concentration blocks are identical across all of them, so only the
per-task knobs below are exposed as CLI arguments. Each invocation draws random
environments within the high-entropy regime (rho in [0.2,1], d_fam/lambda in
[0.5,1.5]) and runs one sweep over n_receptors at fixed n_genes (no warm-start).

Two typical modes:
  ratio scan (default) : R = n_genes * [1..5], 5 random environments
      python3 het_casc.py --n_genes 7
  dense scan (top end) : explicit receptor list, 1 environment
      python3 het_casc.py --n_genes 25 --n_receptors $(seq 25 2 49) \\
              --n_runs 1 --entropy blocked_to_corrected --mu 10 20

Run on the cluster:
  ../run_remote.sh fig1 het_casc.py 0 -- --n_genes 7
"""
import argparse
import time
import numpy as np
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_genes", type=int, required=True,
                   help="number of genes (subunit types); fixed for the sweep.")
    p.add_argument("--n_receptors", type=int, nargs="+", default=None,
                   help="receptor counts to sweep. Default: n_genes*[1,2,3,4,5].")
    p.add_argument("--n_runs", type=int, default=5,
                   help="number of random environments drawn (default 5).")
    p.add_argument("--mu", type=int, nargs=2, metavar=("MIN", "MAX"),
                   default=(20, 30),
                   help="randint range for mu_ligands_per_source (default 20 30).")
    p.add_argument("--entropy", default="annealed",
                   help="entropy loss type (default annealed).")
    p.add_argument("--block_size", type=int, default=None,
                   help="blocked-entropy block size (only set when needed, e.g. 18).")
    p.add_argument("--base_folder", default="/app/data/fig1",
                   help="output root (default /app/data/fig1).")
    return p.parse_args()


def main():
    args = parse_args()

    N_RUNS = args.n_runs
    sweep = args.n_receptors if args.n_receptors is not None \
        else [args.n_genes * k for k in range(1, 6)]
    _NS = len(sweep)

    _D_r = np.random.randint(5, 16, N_RUNS)
    _N_r = np.random.randint(150, 301, N_RUNS)
    _D   = np.repeat(_D_r, _NS)
    _N   = np.repeat(_N_r, _NS)

    entropy_kw = {} if args.block_size is None else {"block_size": args.block_size}

    config = RunConfig(
        # --- Environment ---
        n_families              = np.repeat(np.random.randint(5, 11, N_RUNS), _NS).tolist(),
        n_ligands               = _N.tolist(),
        latent_dim              = _D.tolist(),
        family_spread           = np.repeat(np.random.uniform(0.2, 1.0, N_RUNS) / np.sqrt(_D_r), _NS).tolist(),
        average_family_distance = np.repeat(np.random.uniform(0.5, 1.5, N_RUNS), _NS).tolist(),
        environment_geometry    = "asymmetric",
        distribution_type       = "gaussian",
        observation_noise_sigma = 0.01,

        # --- Presence (hierarchical sampler) ---
        n_presence_blocks      = 1,
        mu_sources             = 1,
        mu_ligands_per_source  = np.repeat(np.random.randint(args.mu[0], args.mu[1], N_RUNS), _NS).tolist(),
        block_shared_conc_mean = False,

        # --- Interface model ---
        use_interface_model = True,

        # --- Concentration ---
        conc_model_type = "lognormal",
        conc_mean       = [cm for cm in [tuple(np.random.uniform(-8.0, -3.0, n)) for n in _N_r] for _ in range(_NS)],
        conc_std        = [cs for cs in [(1.0,) * int(n) for n in _N_r] for _ in range(_NS)],

        # --- Physics ---
        k_sub=5, temperature=0.1, affinity_kernel="gaussian", kernel_params=(1.0,),

        # --- Loss ---
        entropy=args.entropy, **entropy_kw,

        # --- Training ---
        epochs=[int(170 * n + 500) for n in sweep] * N_RUNS,
        lr=0.05, use_scheduler=False,
        batch_size="auto", test_batch_size="auto",
        measurement_fns=("full_array_entropy",),

        # --- Sweep ---
        n_genes                    = args.n_genes,
        n_receptors                = sweep * N_RUNS,
        receptor_sampling_strategy = "cascading",
        receptor_sampling_seed     = 0,
        sweep_name                 = f"ng{args.n_genes}",
        base_folder                = args.base_folder,
        warm_start                 = False,
    )

    print(config)
    t0 = time.time()
    SweepRunner(config).execute()
    h, rem = divmod(time.time() - t0, 3600)
    m, s = divmod(rem, 60)
    print(f"\nHeteromer cascading ng={args.n_genes} sweep complete!  "
          f"{int(h)}h {int(m)}m {s:.0f}s")


if __name__ == "__main__":
    main()
