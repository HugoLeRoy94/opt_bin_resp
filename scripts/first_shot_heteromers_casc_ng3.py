#!/usr/bin/env python3

# First-shot heteromers: cascading strategy, n_genes = 3.
# Sweeps n_receptors from 3 to 29; n_genes fixed — no warm-start.
#
# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/first_shot_heteromers_casc_ng3.py

import time
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

N_LIG = 50
CONC_MEAN  = (-5.5,) * N_LIG
CONC_STD   = (1.0,)  * N_LIG
P_PRESENCE = (0.1,)  * N_LIG

config = RunConfig(
    # --- Environment ---
    n_families              = 5,
    n_ligands               = N_LIG,
    latent_dim              = 5,
    family_spread           = 0.15,
    average_family_distance = 1.0,
    environment_geometry    = "asymmetric",
    distribution_type       = "gaussian",
    observation_noise_sigma = 0.0,

    # --- Presence correlation (Gaussian copula) ---
    n_presence_blocks      = 5,
    rho_block              = 0.3,
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
    entropy="renyi", cov_weight=1.0, penalty_type="repulsion", n_c_bins=10,

    # --- Training ---
    epochs=5000, lr=0.05, use_scheduler=False,
    batch_size="auto", test_batch_size="auto",
    measurement_fns=("full_array_entropy",),

    # --- Sweep ---
    n_genes                    = 3,
    n_receptors                = list(range(3, 30)),
    receptor_sampling_strategy = "cascading",
    receptor_sampling_seed     = 0,
    sweep_name                 = "casc_ng3",
    base_folder                = "/app/data/first_shot",
    warm_start                 = False,
)

print(config)
t0 = time.time()
SweepRunner(config).execute()
h, rem = divmod(time.time() - t0, 3600)
m, s = divmod(rem, 60)
print(f"\nHeteromer cascading ng=3 sweep complete!  {int(h)}h {int(m)}m {s:.0f}s")
