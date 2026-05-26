#!/usr/bin/env python3

# First-shot heteromer arm: uniform_random strategy, n_genes = 8 (§6, narrative_and_next_steps.md).
# Receptor fan-out warm-start: each n_receptors > 8 branches from the (n_genes=8, n_receptors=8) baseline.
# Environment axis: n_ligands × average_family_distance.  D = 10 fixed.
#
# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/Fig1_first_shot/first_shot_heteromers_rand_ng8.py

import time
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

config = RunConfig(
    # --- Environment ---
    n_families              = 5,
    n_ligands               = [50, 100, 200],
    latent_dim              = 7,
    family_spread           = 0.15,
    average_family_distance = [0.5, 1.0, 1.5],
    environment_geometry    = "asymmetric",
    distribution_type       = "gaussian",
    observation_noise_sigma = 0.0,

    # --- Presence correlation (Gaussian copula) ---
    n_presence_blocks      = 1,     # independent Bernoulli baseline (rho_block=0 disables copula)
    rho_block              = 0.0,
    block_shared_conc_mean = False,

    # --- Interface model ---
    use_interface_model = False,

    # --- Concentration ---
    conc_model_type  = "lognormal",
    conc_mean_range  = (-7.0, -5.0),
    conc_std_range   = (1.0,  1.0),
    p_presence_range = (0.1,  0.5),

    # --- Physics ---
    k_sub=5, temperature=0.1, affinity_kernel="gaussian", kernel_params=[1.0],

    # --- Loss ---
    entropy="shannon", cov_weight=1.0, penalty_type="repulsion", n_c_bins=10,

    # --- Training ---
    epochs=5000, lr=0.05, use_scheduler=False,
    batch_size="auto", test_batch_size="auto",
    measurement_fns=[
        "full_array_entropy",
        "codeword_entropy",
        "mean_receptor_distance",
        "mean_specialization_index",
        "mutual_information_ligand",
        "mutual_information_concentration",
        "mutual_information_family",
    ],

    # --- Sweep ---
    n_genes                    = 8,
    n_receptors                = list(range(8, 16)),   # [8, 9, …, 15] — warm-start axis
    receptor_sampling_strategy = "uniform_random",
    receptor_sampling_seed     = 0,
    n_samples                  = 1,
    sweep_name                 = "rand_ng8",
    base_folder                = "/app/data/first_shot",
    warm_start_axis            = "n_receptors",  # fan-out from (n_genes=8, n_receptors=8) baseline
    seed                       = 3,
)

print(config)
t0 = time.time()
SweepRunner(config).execute()
h, rem = divmod(time.time() - t0, 3600)
m, s = divmod(rem, 60)
print(f"\nHeteromer uniform_random ng=8 sweep complete!  {int(h)}h {int(m)}m {s:.0f}s")
