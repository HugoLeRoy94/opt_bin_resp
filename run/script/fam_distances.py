#!/usr/bin/env python3

# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/fam_distances.py

import time
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

config = RunConfig(
    # --- Environment ---
    n_families=[2, 3, 5, 10],                         # independent sweep axis
    n_ligands=100,
    latent_dim=[3],
    family_spread=[0.01, 0.1, 0.5, 1.0],              # independent sweep axis
    average_family_distance=[1.0, 2.0, 5.0, 10.0, 20.0],  # independent sweep axis
    environment_geometry="asymmetric",
    distribution_type="gaussian",
    observation_noise_sigma=0.,
    affinity_length_scale=1.0,

    # --- Concentration ---
    conc_model_type="lognormal",
    conc_mean_range=(-7.0, -5.0),
    conc_std_range=(0.5, 1.5),
    p_presence_range=(0.05, 0.5),

    # --- Physics ---
    n_genes=[5, 10, 20],  # warm-started trajectory axis
    k_sub=5,
    temperature=0.1,

    # --- Mixture ---
    batch_size=2**12,

    # --- Loss ---
    loss_type="exact",
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
    sweep_name="fam_distances",
    base_folder="/app/data/fam_distance_sweep",
    warm_start_axis="n_genes",
    seed=0,
)

print(config)

start_time = time.time()
SweepRunner(config).execute()
total_time = time.time() - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\nFamily distance sweep complete! Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
