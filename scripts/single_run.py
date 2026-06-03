#!/home/hugo/.conda/envs/work/bin/python3

# docker compose -f docker-compose.yaml run --rm gpu-runner python3 /app/run/script/single_run.py

import time
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

N_LIG = 100
CONC_MEAN  = (-5.5,) * N_LIG
CONC_STD   = (1.0,)  * N_LIG
P_PRESENCE = (0.2,)  * N_LIG

config = RunConfig(
    # --- Environment ---
    n_families=5,
    n_ligands=N_LIG,
    latent_dim=10,
    family_spread=.1,
    average_family_distance=1.0,
    environment_geometry="asymmetric",
    distribution_type="gaussian",
    observation_noise_sigma=0.0,

    # --- Presence correlation (Gaussian copula) ---
    n_presence_blocks=1,
    rho_block=0.0,
    block_shared_conc_mean=False,

    # --- Interface model ---
    use_interface_model=False,

    # --- Concentration ---
    conc_model_type="lognormal",
    conc_mean=CONC_MEAN,
    conc_std=CONC_STD,
    p_presence=P_PRESENCE,

    # --- Physics ---
    n_genes=5,
    k_sub=5,
    temperature=0.1,
    affinity_kernel="gaussian",
    kernel_params=(1.,),

    # --- Mixture ---
    batch_size="auto",

    # --- Loss ---
    entropy="renyi",
    cov_weight=1.0,
    penalty_type="repulsion",
    n_c_bins=10,

    # --- Training ---
    epochs=5000,
    lr=0.05,
    use_scheduler=False,
    test_batch_size="auto",
    measurement_fns=("full_array_entropy",),

    # --- Heteromer sampling ---
    n_receptors=50,
    receptor_sampling_strategy="cascading",
    receptor_sampling_seed=0,

    # --- Sweep control ---
    sweep_name="single_run",
    base_folder="/app/data",
    warm_start=False,
)

print(config)

start_time = time.time()
SweepRunner(config).execute()
total_time = time.time() - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\nSingle run complete! Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
