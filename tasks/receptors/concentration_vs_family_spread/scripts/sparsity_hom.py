#!/usr/bin/env python3

# Experiment C — sparsity crossover (homomer arm).
#
# Tests the doc §3 prediction: as mixtures become sparse the composition channel
# collapses and the concentration channel takes over.  Concretely we sweep the
# mixture density `mu_ligands_per_source` from dense (~32 ligands/sniff) to sparse
# (1 ligand/sniff) and measure the channel decomposition directly, rather than
# hoping it emerges in the global entropy:
#
#   full_array_entropy            — total H(A)
#   mutual_information_ligand      — composition / identity channel  I(A;presence)
#   mutual_information_concentration — concentration channel          I(A;c)
#
# Simple environment by request: one family, one presence block, one source,
# scalar lognormal concentration.  Geometry held in the gradient-rich window
# (rho = family_spread*sqrt(D)/lambda = 0.2*sqrt(10)/1 ~ 0.63, inside [0.2,1]),
# so nothing here is confounded by the rho>1 saturation decay.
#
# Homomers: R = n_genes = 10.  Pair with sparsity_het.py (R=10 from 3 genes).
#
# docker compose -f /mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp/docker-compose.yaml run --rm gpu-runner python3 /app/scripts/concentration_vs_family_spread/sparsity_hom.py

import time
import numpy as np
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

N_RUNS = 1                                  # independent samples (different seeds)
_SWEEP = [1, 2, 4, 8, 16, 32]                # mu_ligands_per_source: dense -> sparse, length 6
_NS    = len(_SWEEP)

config = RunConfig(
    # --- Environment (simple: one family, healthy geometry) ---
    n_families              = 1,
    n_ligands               = 200,
    latent_dim              = 10,
    family_spread           = 0.2,           # rho ~ 0.63, inside the safe window
    average_family_distance = 1.,
    environment_geometry    = "asymmetric",
    distribution_type       = "gaussian",
    observation_noise_sigma = 0.01,
    initial_temperature     = 4.,

    # --- Presence (single block / single source; density is the swept axis) ---
    n_presence_blocks      = 1,
    mu_sources             = 1,
    mu_ligands_per_source  = _SWEEP * N_RUNS,
    block_shared_conc_mean = False,

    # --- Interface model ---
    use_interface_model = False,

    # --- Concentration (simple scalar lognormal) ---
    conc_model_type = "lognormal",
    conc_mean       = -5.,
    conc_std        = 1.0,

    # --- Physics ---
    k_sub=5, temperature=0.1, affinity_kernel="gaussian", kernel_params=(1.0,),

    # --- Loss ---
    entropy="annealed",

    # --- Training ---
    epochs=2200, lr=0.01, use_scheduler=False,
    batch_size="auto", test_batch_size="auto",
    measurement_fns=("full_array_entropy",          # total H(A)
                     "identity_channel",            # I(A;M)   — total-comparable
                     "concentration_channel",       # H(A|M)   — total-comparable (sum = H(A))
                     "mutual_information_ligand",        # per-ligand identity marginal (dense regime)
                     "mutual_information_concentration"),# per-ligand concentration marginal (dense regime)

    # --- Sweep ---
    n_genes     = 10,
    sweep_name  = "sparsity_hom",
    base_folder = "/app/data/concentration_vs_family_spread",
    warm_start  = False,
)

print(config)
t0 = time.time()
SweepRunner(config).execute()
h, rem = divmod(time.time() - t0, 3600)
m, s = divmod(rem, 60)
print(f"\nHomomer sparsity sweep complete!  {int(h)}h {int(m)}m {s:.0f}s")
