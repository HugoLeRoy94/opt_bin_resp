#!/usr/bin/env python3

# Homomers sweep: n_genes from 3 to 49 with warm-start chain on n_genes.
# R = n_genes (one homomer per gene).
# 50 runs with random parameters sampled within the high-entropy regime
# (rho in [0.2,1], d_fam/lambda in [0.5,1.5], R_eff=2*mu_lig > R_max=49).
#
# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/scripts/single_ligand/hom.py

import time
import numpy as np
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

N_RUNS = 1
_SWEEP = list(range(3, 15))    # n_genes values, length 47
_NS    = len(_SWEEP)

_D_r = np.random.randint(5, 16, N_RUNS)
_N_r = np.random.randint(150, 301, N_RUNS)
_D   = np.repeat(_D_r, _NS)
_N   = np.repeat(_N_r, _NS)

config = RunConfig(
    # --- Environment ---
    n_families              = 1,
    n_ligands               = 1,
    latent_dim              = 5,
    family_spread           = 1.,
    average_family_distance = 0.,
    environment_geometry    = "asymmetric",
    distribution_type       = "gaussian",
    observation_noise_sigma = 0.0,

    # --- Presence (hierarchical sampler) ---
    n_presence_blocks      = 1,
    mu_sources             = 1,
    mu_ligands_per_source  = 1,
    block_shared_conc_mean = False,

    # --- Interface model ---
    use_interface_model = False,


    # --- Concentration ---
    conc_model_type = "lognormal",
    conc_mean       = -5,
    conc_std        = 1,

    # --- Physics ---
    k_sub=5, temperature=0.1, affinity_kernel="gaussian", kernel_params=(1.0,),

    # --- Loss ---
    entropy="shannon", cov_weight=1.0, penalty_type="repulsion", n_c_bins=10,

    # --- Training ---
    epochs=[int(170 * n + 500) for n in _SWEEP] * N_RUNS, lr=0.01, use_scheduler=False,
    batch_size=500, test_batch_size=500,
    measurement_fns=("full_array_entropy",),

    # --- Sweep ---
    n_genes     = _SWEEP * N_RUNS,
    sweep_name  = "homomers",
    base_folder = "/app/data/fig1_single_ligand",
    warm_start  = False,
)

print(config)
t0 = time.time()
SweepRunner(config).execute()
h, rem = divmod(time.time() - t0, 3600)
m, s = divmod(rem, 60)
print(f"\nHomomer sweep complete!  {int(h)}h {int(m)}m {s:.0f}s")
