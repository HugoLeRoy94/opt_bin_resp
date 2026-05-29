#!/usr/bin/env python3

# First-shot homomer arm (§6 of narrative_and_next_steps.md).
# Sweeps n_genes from 3 to 15 with warm-starting; R = n_genes (one homomer per gene).
# Environment axis: n_ligands × average_family_distance.  D = 10 fixed.
#
# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/Fig1_1_0/first_shot_homomers.py

import time
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

config = RunConfig(
    # --- Environment ---
    n_families              = 5,
    n_ligands               = 100,
    latent_dim              = 5,
    family_spread           = 0.15,   # ρ = 0.15·√10 ≈ 0.47 — gradient-rich regime
    average_family_distance = 1.0,
    environment_geometry    = "asymmetric",
    distribution_type       = "gaussian",
    observation_noise_sigma = 0.01,

    # --- Presence correlation (Gaussian copula) ---
    n_presence_blocks      = 5,     # independent Bernoulli baseline (rho_block=0 disables copula)
    rho_block              = 0.3,
    block_shared_conc_mean = True,

    # --- Interface model ---
    use_interface_model = True,

    # --- Concentration ---
    conc_model_type  = "lognormal",
    conc_mean_range  = (-7.0, -4.0),
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
        #"codeword_entropy",
        #"mean_receptor_distance",
        #"mean_specialization_index",
        #"mutual_information_ligand",
        #"mutual_information_concentration",
        #"mutual_information_family",
        #"mutual_information_block"
    ],

    # --- Sweep ---
    n_genes         = list(range(3, 20)),   # [3, 4, …, 15] — warm-start axis
    n_samples       = 1,
    sweep_name      = "homomers",
    base_folder     = "/app/data/first_shot",
    warm_start_axis = "n_genes",
    seed            = 0,
)

print(config)
t0 = time.time()
SweepRunner(config).execute()
h, rem = divmod(time.time() - t0, 3600)
m, s = divmod(rem, 60)
print(f"\nHomomer sweep complete!  {int(h)}h {int(m)}m {s:.0f}s")
