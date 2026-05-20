#!/usr/bin/env python3

# First-shot heteromer arm: cascading strategy, n_genes = 5 (§6, narrative_and_next_steps.md).
# Warm-starts over n_receptors from 5 to 15; n_genes fixed.
# Environment axis: n_ligands × average_family_distance.  D = 10 fixed.
#
# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner \
#   python3 /app/run/script/first_shot_heteromers_casc_ng5.py

import time
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

config = RunConfig(
    # --- Environment ---
    n_families              = 5,
    n_ligands               = [50, 100, 200],
    latent_dim              = 10,
    family_spread           = 0.15,
    average_family_distance = [0.5, 1.0, 1.5],
    environment_geometry    = "asymmetric",
    distribution_type       = "gaussian",
    observation_noise_sigma = 0.0,

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
    epochs=500, lr=0.05, use_scheduler=False,
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
    n_genes                    = 5,
    n_receptors                = list(range(5, 16)),   # [5, 6, …, 15] — warm-start axis
    receptor_sampling_strategy = "cascading",
    receptor_sampling_seed     = 0,
    n_samples                  = 1,
    sweep_name                 = "casc_ng5",
    base_folder                = "/app/data/first_shot",
    warm_start_axis            = "n_receptors",
    seed                       = 0,
)

print(config)
t0 = time.time()
SweepRunner(config).execute()
h, rem = divmod(time.time() - t0, 3600)
m, s = divmod(rem, 60)
print(f"\nHeteromer cascading ng=5 sweep complete!  {int(h)}h {int(m)}m {s:.0f}s")
