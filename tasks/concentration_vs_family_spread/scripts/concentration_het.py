#!/usr/bin/env python3

# Heteromers, cascading strategy, n_genes = 3, R = n_receptors = 10 (fixed).
# Sweeps conc_std (the log-concentration dispersion — the analog of
# family_spread in concentration space; conc_mean only shifts the operating
# point). family_spread fixed.  Mixture environment
# (mu_ligands_per_source = 25, latent_dim = 10, one family).
# N_RUNS independent samples (distinct receptor_sampling_seed per sample).
#
# docker compose -f /mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp/docker-compose.yaml run --rm gpu-runner python3 /app/scripts/concentration_vs_family_spread/concentration_het.py

import time
import numpy as np
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

N_RUNS = 10                                       # independent samples (different seeds)
_SWEEP = [.01, 0.05, 0.1, 0.2, 0.5, 1., 2.,3.,5.]      # conc_std values, length 7
_NS    = len(_SWEEP)

N_LIG  = 200
# Per-ligand mean log10-concentration ~ U(-7,-3): one draw per sample, held
# fixed across the conc_std sweep (mirrors per-ligand conc_mean in scripts/mixture/hom.py).
_MEANS     = [tuple(np.random.uniform(-7., -3., N_LIG)) for _ in range(N_RUNS)]
_CONC_MEAN = [m for m in _MEANS for _ in range(_NS)]   # len N_RUNS*_NS, aligned to _SWEEP*N_RUNS

config = RunConfig(
    # --- Environment ---
    n_families              = 1,
    n_ligands               = N_LIG,
    latent_dim              = 10,
    family_spread           = 0.1,
    average_family_distance = 1.,
    environment_geometry    = "asymmetric",
    distribution_type       = "gaussian",
    observation_noise_sigma = 0.01,
    initial_temperature     = 4.,

    # --- Presence (hierarchical sampler) — mixture ---
    n_presence_blocks      = 1,
    mu_sources             = 1,
    mu_ligands_per_source  = 25,
    block_shared_conc_mean = False,

    # --- Interface model ---
    use_interface_model = True,

    # --- Concentration (per-ligand mean ~ U(-7,-3); conc_std swept) ---
    conc_model_type = "lognormal",
    conc_mean       = _CONC_MEAN,
    conc_std        = _SWEEP * N_RUNS,

    # --- Physics ---
    k_sub=5, temperature=0.1, affinity_kernel="gaussian", kernel_params=(1.0,),

    # --- Loss ---
    entropy="annealed",

    # --- Training ---
    epochs=2200, lr=0.01, use_scheduler=False,
    batch_size="auto", test_batch_size="auto",
    measurement_fns=("full_array_entropy",),

    # --- Sweep ---
    n_genes                    = 3,
    n_receptors                = 10,
    receptor_sampling_strategy = "cascading",
    receptor_sampling_seed     = np.repeat(np.arange(N_RUNS), _NS).tolist(),
    sweep_name                 = "conc_het",
    base_folder                = "/app/data/concentration_vs_family_spread",
    warm_start                 = False,
)

print(config)
t0 = time.time()
SweepRunner(config).execute()
h, rem = divmod(time.time() - t0, 3600)
m, s = divmod(rem, 60)
print(f"\nHeteromer conc_std sweep complete!  {int(h)}h {int(m)}m {s:.0f}s")
