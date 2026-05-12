#!/usr/bin/env python3

# Investigates the batch-size limit on entropy estimation (topic3, open question / sanity check).
#
# Setup rationale (N=10, D=20, M=100):
#   Coding ceiling:    2^N = 2^10 = 1024 codes  →  10 bits
#   Cover bound:       sum_{k=0}^{22} C(10,k) = 2^10 = 1024  →  not limiting (D=20 >> N/2=5)
#   Vocabulary bound:  M*(N+1) = 100*11 = 1100 > 1024  →  not limiting
#
# The only ceiling that can be crossed by varying batch_size is the coding ceiling itself:
# when batch_size < 1024 the empirical entropy estimator never sees all achievable codes
# in a single batch, producing a systematic downward bias. test_batch_size is fixed large
# (2^14) so that the evaluation metric is accurate and we isolate the training-batch effect.

# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/batch_size_sweep.py

import time
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

config = RunConfig(
    # --- Environment: single family at the origin, same as cover_bound_sharpness ---
    n_families=1,
    n_ligands=500,
    latent_dim=20,                            # D >> N/2=5 → Cover bound = 2^N, not limiting
    family_spread=[0.1,1.,10.],
    average_family_distance=0.0,              # family centre fixed at origin
    environment_geometry="asymmetric",
    distribution_type="gaussian",
    observation_noise_sigma=0.,
    affinity_length_scale=1.0,

    # --- Concentration: identical distribution for every ligand ---
    conc_model_type="lognormal",
    conc_mean_range=(-6.0, -6.0),
    conc_std_range=(1.0, 1.0),
    p_presence_range=(0.05, 0.05),

    # --- Physics: N=10 receptors (2^10=1024 achievable codes) ---
    n_genes=10,
    k_sub=5,
    temperature=0.1,

    # --- Mixture: sweep batch_size across the 2^10 = 1024 transition point ---
    batch_size=[2**i for i in range(6, 15)],  # 64 → 16384, independent sweep axis

    # --- Loss ---
    loss_type="exact",
    entropy="renyi",
    cov_weight=0.0,
    penalty_type="repulsion",
    n_c_bins=10,

    # --- Training ---
    epochs=500,
    lr=0.05,
    use_scheduler=False,
    test_batch_size=2**14,                    # large fixed eval batch for accurate measurement
    measurement_fns=["full_array_entropy"],

    # --- Sweep control ---
    n_samples=1,
    sweep_name="batch_size_sweep",
    base_folder="/app/data/batch_size_sweep",
    warm_start_axis=None,                     # no warm-starting; n_genes is fixed
    seed=0,
)

print(config)

start_time = time.time()
SweepRunner(config).execute()
total_time = time.time() - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\nBatch size sweep complete! Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
