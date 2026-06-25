#!/usr/bin/env python3

# Homomers mixture-size sweep: n_genes = 5.
# Sweeps mu_ligands_per_source from 1 to 29 (step 2); n_genes fixed — no warm-start.
# 10 runs with fixed environment parameters.
#
# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/scripts/mixture/mix_size_hom.py

import time
import numpy as np
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner



config = RunConfig(
    # --- Environment ---
    n_families              = 5,
    n_ligands               = 200,
    latent_dim              = 10,
    family_spread           = 0.2,
    average_family_distance = 1.,
    environment_geometry    = "asymmetric",
    distribution_type       = "gaussian",
    observation_noise_sigma = 0.01,

    # --- Presence (hierarchical sampler) ---
    n_presence_blocks      = 1,
    mu_sources             = 1,
    mu_ligands_per_source  = list(range(1,40,2)),
    block_shared_conc_mean = False,

    # --- Concentration ---
    conc_model_type = "lognormal",
    conc_mean       = -5.,
    conc_std        = 1.,

    # --- Physics ---
    k_sub=5, temperature=0.1, affinity_kernel="gaussian", kernel_params=(1.0,),

    # --- Loss ---
    entropy="blocked",

    # --- Training ---
    epochs=2400,
    lr=0.05, use_scheduler=False,
    batch_size="auto", test_batch_size="auto",
    measurement_fns=("full_array_entropy",),

    # --- Sweep ---
    n_genes     = 15,
    sweep_name  = "hom",
    base_folder = "/app/data/mix_size",
    warm_start  = False,
)

print(config)
t0 = time.time()
SweepRunner(config).execute()
h, rem = divmod(time.time() - t0, 3600)
m, s = divmod(rem, 60)
print(f"\nHomomer mix-size sweep complete!  {int(h)}h {int(m)}m {s:.0f}s")
