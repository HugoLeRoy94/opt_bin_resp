#!/usr/bin/env python3

# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/batch_size_sweep.py

import time
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

config = RunConfig(
    # --- Environment ---
    n_families=5,
    n_ligands=100,
    latent_dim=3,
    family_spread=0.1,
    average_family_distance=5.0,
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
    n_genes=10,
    k_sub=5,
    temperature=0.1,

    # --- Mixture ---
    batch_size=[2**i for i in range(5, 16)],  # independent sweep axis

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
    sweep_name="batch_size_sweep",
    base_folder="/app/data/batch_size_sweep",
    warm_start_axis="n_genes",  # n_genes is scalar here, so warm-starting is a no-op
    seed=0,
)

print(config)

start_time = time.time()
SweepRunner(config).execute()
total_time = time.time() - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\nBatch size sweep complete! Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
