#!/usr/bin/env python3

# Heteromers cascading strategy, n_genes = 3.
# Sweeps n_receptors from 3 to 49; n_genes fixed — no warm-start.
# 50 runs with random environment parameters within the high-entropy regime
# (rho in [0.2,1], d_fam/lambda in [0.5,1.5]).
#
# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/scripts/single_ligand/het_casc_ng3.py

import time
import numpy as np
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

N_RUNS = 1
_SWEEP = list(range(3, 50, 1))  # n_receptors values, every 2, length 24
_NS    = len(_SWEEP)


config = RunConfig(
    # --- Environment ---
    n_families              = 5,
    n_ligands               = 200,
    latent_dim              = 5,
    family_spread           = .2,
    average_family_distance = 1.,
    environment_geometry    = "asymmetric",
    distribution_type       = "gaussian",
    observation_noise_sigma = 0.01,

    # --- Presence (hierarchical sampler) ---
    n_presence_blocks      = 1,
    mu_sources             = 1,
    mu_ligands_per_source  = 1,
    block_shared_conc_mean = False,

    # --- Interface model ---
    use_interface_model = True,

    # --- Concentration ---
    conc_model_type = "lognormal",
    conc_mean       = -5.,
    conc_std        = 1.,

    # --- Physics ---
    k_sub=5, temperature=0.1, affinity_kernel="gaussian", kernel_params=(1.0,),

    # --- Loss ---
    entropy="blocked", cov_weight=1.0, penalty_type="repulsion", n_c_bins=10,

    # --- Training ---
    epochs=[int(170 * n + 500) for n in _SWEEP] * N_RUNS,
    lr=0.01, use_scheduler=False,
    batch_size="auto", test_batch_size="auto",
    measurement_fns=("full_array_entropy",),

    # --- Sweep ---
    n_genes                    = 3,
    n_receptors                = _SWEEP * N_RUNS,
    receptor_sampling_strategy = "cascading",
    receptor_sampling_seed     = 0,
    sweep_name                 = "ng3",
    base_folder                = "/app/data/fig1_single_ligand",
    warm_start                 = False,
)

print(config)
t0 = time.time()
SweepRunner(config).execute()
h, rem = divmod(time.time() - t0, 3600)
m, s = divmod(rem, 60)
print(f"\nHeteromer cascading ng=3 sweep complete!  {int(h)}h {int(m)}m {s:.0f}s")
