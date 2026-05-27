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
    observation_noise_sigma=0.0,

    # --- Presence correlation (Gaussian copula) ---
    n_presence_blocks=1,       # independent Bernoulli baseline (rho_block=0 disables copula)
    rho_block=0.0,
    block_shared_conc_mean=False,

    # --- Interface model ---
    use_interface_model=False,

    # --- Concentration ---
    conc_model_type="lognormal",
    conc_mean_range=(-7.0, -5.0),
    conc_std_range=(1.0, 1.0),
    p_presence_range=(0.1, 0.5),

    # --- Physics ---
    n_genes=5,
    k_sub=5,
    temperature=0.1,
    affinity_kernel="gaussian",
    kernel_params=[1.],

    # --- Mixture ---
    batch_size="auto",

    # --- Loss ---
    entropy="shannon",
    cov_weight=1.0,
    penalty_type="repulsion",
    n_c_bins=10,

    # --- Training ---
    epochs=50000,
    lr=0.05,
    use_scheduler=False,
    test_batch_size="auto",
    measurement_fns=[
        "full_array_entropy",
        "codeword_entropy",
        "mean_receptor_distance",
        "mean_specialization_index",
        "mutual_information_ligand",
        "mutual_information_concentration",
        "mutual_information_family",
    ],

    # --- Heteromer sampling ---
    n_receptors=5,
    receptor_sampling_strategy="uniform_random",
    receptor_sampling_seed=0,

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
