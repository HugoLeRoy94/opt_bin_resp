# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/opt_homomers.py Nfamilies
# add -d for silent running

# To run on GPU 2
# MY_GPU=2 docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/opt_homomers.py Nfamilies

import sys
sys.path.append('/app')
# unit_test/test_single_receptor.py
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from tqdm import tqdm
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
                conditional_entropy_family,
                mutual_information_family,
                conditional_entropy_concentration,
                mutual_information_concentration)
from run import initialize,train,test
from src.IO import ExperimentLogger, ExperimentLoader


latent_dim_list = [3, 7, 10,15,20]
n_units_list = [1,2,3,5,7,8,10,12,15,20,30,50]
n_samples = 10 # Number of independent runs to estimate standard deviation

N_train = 2**12

CONF = {
    # environment
        # energies
    "n_families": 0, # Will be set in the loop
    "latent_dim": 0, # Will be set in the loop
    "average_family_distance" : 5., # Squeeze them tightly together
    "shape_sigma": .1, # Make the clouds fatter so they overlap heavily
        # concentration
    "init_means": [], # Will be set in the loop
    # receptor 
    "k_sub": 5, # number of sub-units
    "temperature": .1, # temperature of the sigmoid that approximate a binary answer
    "n_units" : 0, # number of genes
    "receptor_indices" : None, # actual receptors considered
    
    # training characteristics
    "batch_size": N_train,
    "epochs": 500,


    "lr": 0.05, # learning rate
    "loss": "exact", # type of loss
    'entropy':'renyi',
    "use_scheduler":False,
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Missing n_families argument.")
        print("Usage: python3 opt_homomers.py <n_families>")
        sys.exit(1)
        
    n_families = int(sys.argv[1])

    total_iterations = len(latent_dim_list) * n_samples * len(n_units_list)
    start_time = time.time()

    with tqdm(total=total_iterations, desc="Overall Progress", dynamic_ncols=True) as pbar:
        for latent_dim in latent_dim_list:
            for sample in range(n_samples):
                prev_env = None
                
                # Set default config for this trajectory
                CONF["n_families"] = n_families
                CONF["latent_dim"] = latent_dim
                
                
                # Align the concentration means so they exist in the same intensity range
                CONF["init_means"] = [np.random.uniform(3.0, 5.0) for _ in range(n_families)]
                
                for n_units in n_units_list:
                    # Set up the parameter-specific base directory
                    base_dir = f"/app/data/homomers/families_{n_families}/dim_{latent_dim}/n_units_{n_units}"
                    os.makedirs(base_dir, exist_ok=True)

                    # Check if this exact sample has already been computed
                    existing_dirs = [d for d in os.listdir(base_dir) if d.startswith(f"sample_{sample}")]
                    already_computed_dirs = [d for d in existing_dirs if os.path.exists(os.path.join(base_dir, d, "test_results.json"))]
                    
                    if len(already_computed_dirs) > 0:
                        tqdm.write(f"Skipping: Families={n_families}, Dim={latent_dim}, Units={n_units}, Sample={sample+1}/{n_samples} (Already exists)")
                        # Load the environment to pass it to the next n_units iteration
                        exact_run_folder = os.path.join(base_dir, already_computed_dirs[0])
                        try:
                            device = "cuda" if torch.cuda.is_available() else "cpu"
                            loader = ExperimentLoader(exact_run_folder=exact_run_folder)
                            e, _, _, _, _, c = loader.load_objects(device=device)
                            prev_env = e
                            # Sync the running config to match the loaded environment's initialization
                            if "init_means" in c:
                                CONF["init_means"] = c["init_means"]
                        except Exception as ex:
                            tqdm.write(f"Warning: Could not load previous environment from {exact_run_folder}: {ex}")
                            prev_env = None
                        pbar.update(1)
                        continue
                        
                    tqdm.write(f"\n--- Training: Families={n_families}, Dim={latent_dim}, Units={n_units}, Sample={sample+1}/{n_samples} ---")
                    
                    # Update CONF for current parameters
                    CONF["n_units"] = n_units
                    # Generate heteromers matching the experimental exponential distribution
                    CONF["receptor_indices"] = torch.tensor([[i for _ in range(CONF['k_sub'])] for i in range(n_units)], dtype=torch.long)
                    #generate_exp_distributed_receptors(N_receptors=n_units, n_units=n_units, k_sub=CONF['k_sub'])

                    env, rec, loss_fn, optimize = initialize(CONF, SymmetricEnv=False, prev_env=prev_env)
                    
                    prev_env = env

                    # ExperimentLogger creates a timestamped folder inside base_dir
                    logger = ExperimentLogger(base_path=base_dir, experiment_name=f"sample_{sample}")
                    logger.save_config(CONF)

                    measurement_fns = [
                        full_array_entropy, 
                        mean_receptor_distance,
                        conditional_entropy_family,
                        mutual_information_family,
                        conditional_entropy_concentration,
                        mutual_information_concentration,
                        receptor_distances,
                        rank_ordered_distances,
                        mean_specialization_index,
                        receptor_conditioned_entropy
                    ]

                    train_out = train(CONF, env, rec, loss_fn, optimize, measurement_fns=measurement_fns)

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
                        CONF["receptor_indices"], N_samples=test_batch_size, epoch=test_epochs,
                        measurement_fns=measurement_fns
                    )
                    
                    # Save test results
                    test_out_path = os.path.join(logger.run_dir, "test_results.json")
                    with open(test_out_path, "w") as f:
                        clean_results = {k: [float(val) if isinstance(val, torch.Tensor) else val for val in v] for k, v in test_results.items()}
                        json.dump(clean_results, f, indent=4)
                        
                    pbar.update(1)

    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nOptimization complete! Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")