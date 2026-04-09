import sys
sys.path.append('/app')
# unit_test/test_single_receptor.py
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
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
                mean_receptor_distance)
from run import initialize,train,test
from src.IO import ExperimentLogger

n_units_list = [5]#, 10, 15, 20, 30, 50]
k_sub = 5
n_families = 1
N_train = 2**17
N_test = 2**14

CONF = {
    # environment
        # energies
    "n_families": n_families,
    "latent_dim": 10,
        # concentration
    "init_means": [np.random.randint(1, 8) for _ in range(n_families)], # Fixed 
    "shape_sigma": 10.,
    # receptor 
    "k_sub": k_sub, # number of sub-units
    "temperature": 0.1, # temperature of the sigmoid that approximate a binary answer
    "n_units" : 0, # number of genes
    "receptor_indices" : torch.tensor([[i for _ in range(k_sub)] for i in range(0)], dtype=torch.long), # actual receptors considered
    
    # training characteristics
    "batch_size": N_train,
    "epochs": 500,
    "n_bins": 2, # binary activation


    "lr": 0.05, # learning rate
    "exact_loss": True # type of loss
    
}

for n_units in n_units_list:
    print(f"\n--- Starting training for n_units = {n_units} ---")
    
    CONF["n_units"] = n_units
    CONF["receptor_indices"] = torch.tensor([[i for _ in range(k_sub)] for i in range(n_units)], dtype=torch.long)

    env, rec, loss_fn, optimize = initialize(CONF, SymmetricEnv=False)

    logger = ExperimentLogger(base_path="/app/data/", experiment_name=f"opt_homomers_n_units_{n_units}")
    logger.save_config(CONF)

    train_out = train(CONF, env, rec, loss_fn, optimize,measurement_fns=[full_array_entropy,mean_receptor_distance,])

    print(train_out)

    # Save training statistics 
    epochs_run = len(next(iter(train_out.values())))
    print(next(iter(train_out.values())))
    print(epochs_run)
    for i in range(epochs_run):
        stats = {k: v[i] for k, v in train_out.items()}
        logger.save_stats(i, stats)

    # Save the final checkpoint (best model)
    logger.save_checkpoint(
        epoch=CONF["epochs"], env=env, physics=rec, 
        receptor_indices=CONF["receptor_indices"], is_best=True
    )