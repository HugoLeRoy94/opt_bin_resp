# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/opt_homomers.py Nfamilies
# add -d for silent running

# To run on GPU 2
# MY_GPU=2 docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/opt_homomers.py 20

import sys
sys.path.append('/app')
# unit_test/test_single_receptor.py
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from itertools import cycle

from src import (generate_receptor_indices,
                generate_cascading_receptors,
                generate_targeted_receptors,
                plot_family_summary,                
                plot_latent_radar_chart,
                evaluate_model,
                plot_summary,
                plot_latent_umap,
                receptor_distances,
                full_array_entropy,
                mean_receptor_distance,
                ConditionalEntropyFamily,
                MutualInformationFamily,
                ConditionalEntropyConcentration,
                MutualInformationConcentration)
from run import initialize,train,test
from src.IO import ExperimentLogger


latent_dim_list = [3, 7, 10]
n_units_list = [1,2,3,5,7,8,10,12,15,20,30,50]
n_samples = 5 # Number of independent runs to estimate standard deviation

N_train = 2**17

CONF = {
    # environment
        # energies
    "n_families": 0, # Will be set in the loop
    "latent_dim": 0, # Will be set in the loop
    "avg_family_distance": 1.0, # Target average distance between family centers
        # concentration
    "init_means": [], # Will be set in the loop
    "shape_sigma": 0, # Will be set in the loop
    # receptor 
    "k_sub": 5, # number of sub-units
    "temperature": 0.1, # temperature of the sigmoid that approximate a binary answer
    "n_units" : 0, # number of genes
    "receptor_indices" : torch.tensor([[i for _ in range(5)] for i in range(0)], dtype=torch.long), # actual receptors considered
    
    # training characteristics
    "batch_size": N_train,
    "epochs": 5000,


    "lr": 0.05, # learning rate
    "exact_loss": True # type of loss
    
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Missing n_families argument.")
        print("Usage: python3 opt_homomers.py <n_families>")
        sys.exit(1)
        
    n_families = int(sys.argv[1])

    for latent_dim in latent_dim_list:
        for n_units in n_units_list:
            # Set up the parameter-specific base directory
            base_dir = f"/app/data/families_{n_families}/dim_{latent_dim}/n_units_{n_units}"
            os.makedirs(base_dir, exist_ok=True)

            for sample in range(n_samples):
                # Check if this exact sample has already been computed
                existing_dirs = [d for d in os.listdir(base_dir) if d.startswith(f"sample_{sample}")]
                already_computed = any(os.path.exists(os.path.join(base_dir, d, "test_results.json")) for d in existing_dirs)
                
                if already_computed:
                    print(f"Skipping: Families={n_families}, Dim={latent_dim}, Units={n_units}, Sample={sample+1}/{n_samples} (Already exists)")
                    continue
                    
                print(f"\n--- Training: Families={n_families}, Dim={latent_dim}, Units={n_units}, Sample={sample+1}/{n_samples} ---")
                
                # Update CONF for current parameters
                CONF["n_families"] = n_families
                CONF["latent_dim"] = latent_dim
                CONF["init_means"] = [np.random.randint(1, 8) for _ in range(n_families)]
                CONF["shape_sigma"] = 1. / n_families
                CONF["n_units"] = n_units
                CONF["receptor_indices"] = torch.tensor([[i for _ in range(CONF['k_sub'])] for i in range(n_units)], dtype=torch.long)

                env, rec, loss_fn, optimize = initialize(CONF, SymmetricEnv=False)

                # ExperimentLogger creates a timestamped folder inside base_dir
                logger = ExperimentLogger(base_path=base_dir, experiment_name=f"sample_{sample}")
                logger.save_config(CONF)

                # Initialize our new tracking classes
                cond_h_fam = ConditionalEntropyFamily(env, rec, CONF["receptor_indices"], n_samples=2000)
                mi_fam = MutualInformationFamily(env, rec, CONF["receptor_indices"], n_samples=2000)
                cond_h_conc = ConditionalEntropyConcentration(env, rec, CONF["receptor_indices"], n_samples=2000)
                mi_conc = MutualInformationConcentration(env, rec, CONF["receptor_indices"], n_samples=2000)

                train_out = train(CONF, env, rec, loss_fn, optimize, measurement_fns=[
                    full_array_entropy, 
                    mean_receptor_distance,
                    cond_h_fam,
                    mi_fam,
                    cond_h_conc,
                    mi_conc
                ])

                # Save training statistics 
                epochs_run = len(next(iter(train_out.values())))
                for i in range(epochs_run):
                    stats = {k: v[i] for k, v in train_out.items()}
                    logger.save_stats(i, stats)

                # Save the final checkpoint (best model)
                logger.save_checkpoint(
                    epoch=CONF["epochs"], env=env, physics=rec, 
                    receptor_indices=CONF["receptor_indices"], is_best=True
                )
                
                # Run test measurements for std deviation estimation
                test_epochs = 10
                test_batch_size = 100_000
                test_results = test(
                    CONF, env, rec, loss_fn, optimize, 
                    CONF["receptor_indices"], N_samples=test_batch_size, epoch=test_epochs
                )
                
                # Save test results
                test_out_path = os.path.join(logger.run_dir, "test_results.json")
                with open(test_out_path, "w") as f:
                    clean_results = [float(val) if isinstance(val, torch.Tensor) else val for val in test_results]
                    json.dump({"test_entropies": clean_results}, f, indent=4)