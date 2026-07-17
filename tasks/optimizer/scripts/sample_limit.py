#!/usr/bin/env python3
"""Sample-limit study — train with the KT lower bound at several TRAIN sample sizes.

Two questions this study answers (with test_scaling.py + plot_sample_scaling.py):
  1. Am I reaching the sample ceiling?  The converged KT entropy is compared to the
     resolvable-entropy ceiling log2(sample_size). Sitting AT it → sample-limited;
     plateauing BELOW it → optimization/physics-limited.
  2. How much entropy does a bigger TEST set buy?  The optimized env is re-measured
     post-hoc at growing test sizes (test_scaling.py) — no retraining.

This script only does (the training half): entropy = "kt", two conditions
  (n_genes, n_receptors) = (20, 20) and (15, 45),
and sweeps the TRAIN batch over --train_batch (default: auto / 4096 / 1024).
"auto" is the memory-max train batch; the two smaller ones probe train-size effect.

Per condition ONE environment is drawn and reused across all train sizes (same env +
same receptor indices via receptor_sampling_seed=0) so the only varied knob is the
train batch. Separate sweep_name ("sample_limit") keeps it out of plot_optimizer.

  python3 sample_limit.py                              # all 3 train sizes, both conds
  python3 sample_limit.py --train_batch 1024           # one train size (own GPU)

Run on the cluster (split over GPUs to parallelise the 3 train sizes):
  ../run_remote.sh optimizer sample_limit.py 0 -- --train_batch auto
  ../run_remote.sh optimizer sample_limit.py 1 -- --train_batch 4096
  ../run_remote.sh optimizer sample_limit.py 2 -- --train_batch 1024
"""
import argparse
import time
import numpy as np
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

CONDITIONS = [(20, 20), (15, 45)]              # (n_genes, n_receptors)
DEFAULT_TRAIN_BATCH = ["auto", "4096", "1024"]  # "auto" = memory-max train batch
N_RUNS = 1


def _batch(x):
    return x if x == "auto" else int(x)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_batch", nargs="+", default=DEFAULT_TRAIN_BATCH,
                   help=f"train batch sizes ('auto' or int). default: {DEFAULT_TRAIN_BATCH}")
    return p.parse_args()


def main():
    args = parse_args()
    train_batches = [_batch(x) for x in args.train_batch]

    cols = {k: [] for k in (
        "n_genes", "n_receptors", "batch_size", "epochs",
        "n_families", "n_ligands", "latent_dim", "family_spread",
        "average_family_distance", "mu_ligands_per_source", "conc_mean", "conc_std")}

    for ci, (ng, nr) in enumerate(CONDITIONS):
        for r in range(N_RUNS):
            # Fix the env per (condition, run) so every train batch trains the SAME env.
            np.random.seed(1000 * ci + r)
            d   = int(np.random.randint(5, 16))
            n   = int(np.random.randint(150, 301))
            nf  = int(np.random.randint(5, 11))
            mu  = int(np.random.randint(20, 30))
            fs  = float(np.random.uniform(0.2, 1.0) / np.sqrt(d))
            afd = float(np.random.uniform(0.5, 1.5))
            cm  = tuple(np.random.uniform(-8.0, -3.0, n))
            cs  = (1.0,) * n
            for tb in train_batches:
                cols["n_genes"].append(ng)
                cols["n_receptors"].append(nr)
                cols["batch_size"].append(tb)
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

        # --- Loss (fixed: KT lower bound) ---
        entropy = "kt",

        # --- Training ---
        epochs=cols["epochs"],
        lr=0.05, use_scheduler=False,
        batch_size=cols["batch_size"],   # the swept dimension
        test_batch_size="auto",          # per-epoch curve; final = memory-max (persisted)
        measurement_fns=("full_array_entropy", "entropy_kt", "entropy_kt_upper"),

        # --- Sweep ---
        n_genes                    = cols["n_genes"],
        n_receptors                = cols["n_receptors"],
        receptor_sampling_strategy = "cascading",
        receptor_sampling_seed     = 0,
        sweep_name                 = "sample_limit",
        base_folder                = "/app/data/optimizer",
        warm_start                 = False,
    )

    print(config)
    n_cfg = len(cols["batch_size"])
    print(f"\n{len(CONDITIONS)} conditions x {len(train_batches)} train sizes x "
          f"{N_RUNS} runs = {n_cfg} configs")
    t0 = time.time()
    SweepRunner(config).execute()
    h, rem = divmod(time.time() - t0, 3600)
    m, s = divmod(rem, 60)
    print(f"\nSample-limit training complete!  {int(h)}h {int(m)}m {s:.0f}s")


if __name__ == "__main__":
    main()
