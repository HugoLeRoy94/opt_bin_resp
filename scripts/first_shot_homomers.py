#!/usr/bin/env python3

# First-shot homomer arm: sweeps n_genes from 3 to 19 with warm-start chain on n_genes.
# R = n_genes (one homomer per gene).
#
# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/scripts/first_shot_homomers.py

import time
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

import numpy as np

N_LIG = 200
CONC_MEAN  = tuple(np.random.uniform(-8.0, -3.0, N_LIG))
CONC_STD   = (1.0,) * N_LIG
P_PRESENCE = tuple(np.random.uniform(0.05, 0.3, N_LIG))

config = RunConfig(
    # --- Environment ---
    n_families              = 5,
    n_ligands               = N_LIG,
    latent_dim              = 10,
    family_spread           = 0.15,
    average_family_distance = 1.0,
    environment_geometry    = "asymmetric",
    distribution_type       = "gaussian",
    observation_noise_sigma = 0.01,

    # --- Presence correlation (Gaussian copula) ---
    n_presence_blocks      = 20,
    rho_block              = 0.5,
    block_shared_conc_mean = True,

    # --- Interface model ---
    use_interface_model = True,

    # --- Concentration ---
    conc_model_type = "lognormal",
    conc_mean       = CONC_MEAN,
    conc_std        = CONC_STD,
    p_presence      = P_PRESENCE,

    # --- Physics ---
    k_sub=5, temperature=0.1, affinity_kernel="gaussian", kernel_params=(1.0,),

    # --- Loss ---
    entropy="shannon", cov_weight=1.0, penalty_type="repulsion", n_c_bins=10,

    # --- Training ---
    epochs=5000, lr=0.05, use_scheduler=False,
    batch_size="auto", test_batch_size="auto",
    measurement_fns=(
        "full_array_entropy",
        #"mutual_information_ligand",
        #"mutual_information_concentration",
        #"mutual_information_family",
        #"mutual_information_block",
    ),

    # --- Sweep ---
    n_genes     = list(range(3, 20)),
    sweep_name  = "homomers",
    base_folder = "/app/data/test",
    warm_start  = False,
)

print(config)
t0 = time.time()
SweepRunner(config).execute()
h, rem = divmod(time.time() - t0, 3600)
m, s = divmod(rem, 60)
print(f"\nHomomer sweep complete!  {int(h)}h {int(m)}m {s:.0f}s")
