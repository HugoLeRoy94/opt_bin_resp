#!/usr/bin/env python3
"""Optimizer-comparison task — fix (n_genes, n_receptors), vary the loss type.

Same environment / physics / concentration regime as fig1's het_casc, but instead
of sweeping receptor count we fix three (n_genes, n_receptors) conditions and sweep
the optimization loss (entropy estimator). For each condition we draw N_RUNS random
environments and reuse each one across all losses, so the losses are compared on
IDENTICAL environments (receptor_sampling_seed is fixed too → identical receptor
indices). The only knob is --losses.

  conditions : (3,15), (15,45), (20,20)
  losses     : collision / blocked / annealed / blocked_to_corrected  (default)
               shannon is excluded — its 100*2^R batch is infeasible for R > ~14,
               and every condition here has R >= 15.

  python3 optimizer.py                              # default losses
  python3 optimizer.py --losses blocked annealed    # subset

Run on the cluster:
  ../run_remote.sh optimizer optimizer.py 0
"""
import argparse
import time
import numpy as np
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

CONDITIONS = [(3, 15), (15, 45), (20, 20)]     # (n_genes, n_receptors)
DEFAULT_LOSSES = ["collision", "blocked", "annealed", "blocked_to_corrected"]
N_RUNS = 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--losses", nargs="+", default=DEFAULT_LOSSES,
                   help=f"entropy/loss types to compare (default: {DEFAULT_LOSSES}).")
    return p.parse_args()


def main():
    args = parse_args()

    # One flat config list: for each condition, draw N_RUNS environments and reuse
    # each across every loss (same env + same receptor indices ⇒ fair comparison).
    cols = {k: [] for k in (
        "n_genes", "n_receptors", "entropy", "epochs", "n_families", "n_ligands",
        "latent_dim", "family_spread", "average_family_distance",
        "mu_ligands_per_source", "conc_mean", "conc_std")}

    for ng, nr in CONDITIONS:
        for _ in range(N_RUNS):
            d   = int(np.random.randint(5, 16))
            n   = int(np.random.randint(150, 301))
            nf  = int(np.random.randint(5, 11))
            mu  = int(np.random.randint(20, 30))
            fs  = float(np.random.uniform(0.2, 1.0) / np.sqrt(d))
            afd = float(np.random.uniform(0.5, 1.5))
            cm  = tuple(np.random.uniform(-8.0, -3.0, n))
            cs  = (1.0,) * n
            for loss in args.losses:
                cols["n_genes"].append(ng)
                cols["n_receptors"].append(nr)
                cols["entropy"].append(loss)
                cols["epochs"].append(int(170 * nr + 500))
                cols["n_families"].append(nf)
                cols["n_ligands"].append(n)
                cols["latent_dim"].append(d)
                cols["family_spread"].append(fs)
                cols["average_family_distance"].append(afd)
                cols["mu_ligands_per_source"].append(mu)
                cols["conc_mean"].append(cm)
                cols["conc_std"].append(cs)

    config = RunConfig(
        # --- Environment ---
        n_families              = cols["n_families"],
        n_ligands               = cols["n_ligands"],
        latent_dim              = cols["latent_dim"],
        family_spread           = cols["family_spread"],
        average_family_distance = cols["average_family_distance"],
        environment_geometry    = "asymmetric",
        distribution_type       = "gaussian",
        observation_noise_sigma = 0.01,

        # --- Presence (hierarchical sampler) ---
        n_presence_blocks      = 1,
        mu_sources             = 1,
        mu_ligands_per_source  = cols["mu_ligands_per_source"],
        block_shared_conc_mean = False,

        # --- Interface model ---
        use_interface_model = True,

        # --- Concentration ---
        conc_model_type = "lognormal",
        conc_mean       = cols["conc_mean"],
        conc_std        = cols["conc_std"],

        # --- Physics ---
        k_sub=5, temperature=0.1, affinity_kernel="gaussian", kernel_params=(1.0,),

        # --- Loss (the swept dimension) ---
        entropy = cols["entropy"],

        # --- Training ---
        epochs=cols["epochs"],
        lr=0.05, use_scheduler=False,
        batch_size="auto", test_batch_size="auto",
        measurement_fns=("full_array_entropy",),

        # --- Sweep ---
        n_genes                    = cols["n_genes"],
        n_receptors                = cols["n_receptors"],
        receptor_sampling_strategy = "cascading",
        receptor_sampling_seed     = 0,
        sweep_name                 = "optimizer",
        base_folder                = "/app/data/optimizer",
        warm_start                 = False,
    )

    print(config)
    n_cfg = len(cols["entropy"])
    print(f"\n{len(CONDITIONS)} conditions x {len(args.losses)} losses x {N_RUNS} runs "
          f"= {n_cfg} configs")
    t0 = time.time()
    SweepRunner(config).execute()
    h, rem = divmod(time.time() - t0, 3600)
    m, s = divmod(rem, 60)
    print(f"\nOptimizer comparison complete!  {int(h)}h {int(m)}m {s:.0f}s")


if __name__ == "__main__":
    main()
