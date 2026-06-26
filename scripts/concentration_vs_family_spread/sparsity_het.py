#!/usr/bin/env python3

# Experiment C — sparsity crossover (heteromer arm).
#
# Same environment and sweep as sparsity_hom.py; the only change is the
# architecture: R = n_receptors = 10 built combinatorially from n_genes = 3.
# The prediction worth testing is not just that the channels re-emerge as the
# mixture gets sparse, but that the HETEROMER concentration channel re-emerges
# more strongly than the homomer one — the geometric-mean EC50 ladder paying off
# exactly where composition coding has collapsed.
#
# Sweep mu_ligands_per_source dense -> sparse; measure the same three channels:
#   full_array_entropy, mutual_information_ligand, mutual_information_concentration.
#
# docker compose -f /mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp/docker-compose.yaml run --rm gpu-runner python3 /app/scripts/concentration_vs_family_spread/sparsity_het.py

import time
import numpy as np
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

N_RUNS = 10                                  # independent samples (different seeds)
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
    use_interface_model = True,

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
    measurement_fns=("full_array_entropy",
                     "mutual_information_ligand",
                     "mutual_information_concentration"),

    # --- Sweep ---
    n_genes                    = 3,
    n_receptors                = 10,
    receptor_sampling_strategy = "cascading",
    receptor_sampling_seed     = np.repeat(np.arange(N_RUNS), _NS).tolist(),
    sweep_name                 = "sparsity_het",
    base_folder                = "/app/data/concentration_vs_family_spread",
    warm_start                 = False,
)

print(config)
t0 = time.time()
SweepRunner(config).execute()
h, rem = divmod(time.time() - t0, 3600)
m, s = divmod(rem, 60)
print(f"\nHeteromer sparsity sweep complete!  {int(h)}h {int(m)}m {s:.0f}s")
