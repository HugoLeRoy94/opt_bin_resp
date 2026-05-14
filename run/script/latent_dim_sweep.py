#!/usr/bin/env python3

# Investigates Cover bound sharpness (open question 1 from topic3_cover_and_entropy_bounds.md).
# Minimal environment: single ligand family centred at the origin, identical concentration
# distribution for every ligand.  Sweeps latent_dim (D) at increasing pool sizes (n_genes = N),
# building the (D, N) phase diagram needed to test where H(A) saturates relative to
#   Cover bound:      sum_{k=0}^{D+2} C(N, k)
#   Vocabulary bound: M * (N+1)
# With homomers only (default receptor_indices), receptors share no subunits, so this
# establishes the near-general-position baseline before heteromer correlations are added.

# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/latent_dim_sweep.py

import time
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

config = RunConfig(
    # --- Environment: single family at the origin, small intra-family spread ---
    n_families=1,
    n_ligands=100,
    latent_dim=[1, 2, 3, 5, 7, 10, 15, 20],  # independent sweep axis (D)
    family_spread=0.1,
    average_family_distance=0.0,              # family centre fixed at origin
    environment_geometry="asymmetric",
    distribution_type="gaussian",
    observation_noise_sigma=0.,
    affinity_kernel="gaussian",
    kernel_params=[1.0],

    # --- Concentration: identical distribution for every ligand ---
    conc_model_type="lognormal",
    conc_mean_range=(-6.0, -6.0),
    conc_std_range=(1.0, 1.0),
    p_presence_range=(0.2, 0.2),

    # --- Physics ---
    n_genes=[5, 7, 10, 13, 15, 17, 20],  # warm-started trajectory axis (N, homomers only)
    k_sub=5,
    temperature=0.1,

    # --- Mixture ---
    batch_size=2**12,

    # --- Loss ---
    entropy="renyi",
    cov_weight=1.0,
    penalty_type="repulsion",
    n_c_bins=10,

    # --- Training ---
    epochs=500,
    lr=0.05,
    use_scheduler=False,
    test_batch_size=2**12,
    measurement_fns=["full_array_entropy"],

    # --- Sweep control ---
    n_samples=5,
    sweep_name="cover_bound_sharpness",
    base_folder="/app/data/cover_bound_sharpness",
    warm_start_axis="n_genes",
    seed=0,
)

print(config)

start_time = time.time()
SweepRunner(config).execute()
total_time = time.time() - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\nCover bound sharpness sweep complete! Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
