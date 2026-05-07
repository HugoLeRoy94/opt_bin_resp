#!/home/hugo/.conda/envs/work/bin/python3

#docker compose -f docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/single_run.py

# %% Setup and Imports
import time
import sys
import os
import torch

sys.path.append('/app')

# --- Local Imports ---
from src.config import SingleRunConfig
from src.run import SimulationRunner
from src.IO import ExperimentLogger

# %% Configuration
# Adjust your run parameters here
n_families = 5
n_ligands = 100
latent_dim = 3
n_units = 5
run_dir = "/app/data/single_run"
env_type = "asymmetric"
loss_type = "exact"
epochs = 5000
batch_size = 2**12
k_sub = 5
noise_sigma = 0.

# Generate receptor indices for homomers (each receptor is made of k_sub identical units)
receptor_indices = torch.arange(n_units).unsqueeze(1).repeat(1, k_sub)

# Package parameters into the SingleRunConfig
config = SingleRunConfig(
    n_families=n_families,
    n_ligands=n_ligands,
    latent_dim=latent_dim,
    n_units=n_units,
    epochs=epochs,
    batch_size=batch_size,
    k_sub=k_sub,
    temperature=0.1,
    lr=0.05,
    shape_sigma=0.1,
    noise_sigma=noise_sigma,
    average_family_distance=5.0,
    use_sensitivity=False,
    loss_type=loss_type,
    env_type=env_type,
    entropy="renyi",
    receptor_indices=receptor_indices.tolist()
)

print(config)

# %% Execution
logger = ExperimentLogger(run_dir=run_dir)
logger.save_config(config)

start_time = time.time()
runner = SimulationRunner(config, logger)
runner.run(receptor_indices=receptor_indices)

# Wrap up
total_time = time.time() - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\n✅ Single run complete! Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")