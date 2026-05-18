#!/home/hugo/.conda/envs/work/bin/python3

#docker compose -f docker-compose.yaml run --rm gpu-runner python3 /app/run/script/single_run.py

import time
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

config = RunConfig(
    # --- Environment ---
    n_families=1,
    n_ligands=100,
    latent_dim=10,
    family_spread=.1,
    average_family_distance=1.0,
    environment_geometry="asymmetric",
    distribution_type="gaussian",
    

    # --- Concentration ---
    conc_model_type="lognormal",
    conc_mean_range=(-6.0, -6.0),
    conc_std_range=(1.0, 1.0),
    p_presence_range=(.2, .2),
    # --- Physics ---
    n_genes=30,
    k_sub=5,
    temperature=0.1,
    observation_noise_sigma=0.,
    affinity_kernel="gaussian",# "gaussian" for a saturation/ "quadratic"
    kernel_params=[1.],# needs an entry if gaussian is used

    # --- Mixture ---
    batch_size=2**16,

    # --- Loss ---
    entropy="renyi", # shannon | renyi | blocked | proxy
    cov_weight=None,
    penalty_type=None,
    n_c_bins=10,

    # --- Training ---
    epochs=5000,
    lr=0.05,
    use_scheduler=False,
    test_batch_size=2**16,
    measurement_fns=["full_array_entropy"],

    # --- Sweep control ---
    n_samples=1,
    sweep_name="single_run",
    base_folder="/app/data",
    warm_start_axis=None,  # no warm-starting for a single run
    seed=0,
)

print(config)

start_time = time.time()
SweepRunner(config).execute()
total_time = time.time() - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\nSingle run complete! Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
