#!/usr/bin/env python3

# Investigates how the dimensionality of the latent chemical space (D = latent_dim)
# shapes the mutual information captured by an optimised heteromeric receptor array.
#
# Scientific question:
#   For a fixed gene pool of size n_genes, does the MI plateau earlier (in n_receptors)
#   when D is small, and does the saturation MI itself scale with D?
#   How does this interact with the complexity of the odour world
#   (n_ligands × average_family_distance)?
#
# Design:
#   • Environment axes: n_ligands × average_family_distance  (odour-world complexity)
#   • Sweep axis:       latent_dim  D ∈ {2, 3, 5, 7, 10, 15, 20}
#   • Warm-start axis:  n_receptors ∈ [5 … 15]   (array grows; optimisation warm-starts)
#   • Heteromers:       uniform_random sampling from a pool of n_genes = 10
#
# Produces the (D, n_receptors) MI surface for each (n_ligands, avg_family_distance) slice.
#
# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/latent_dim_mi_heteromers.py

import time
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

config = RunConfig(
    # --- Environment ---
    n_families              = 5,
    n_ligands               = 50,          # odour-world size axis
    latent_dim              = [2, 3, 5, 7, 10, 15, 20],  # SWEEP AXIS (D)
    family_spread           = 0.15,
    average_family_distance = 1.0,         # family separation axis
    environment_geometry    = "asymmetric",
    distribution_type       = "gaussian",
    observation_noise_sigma = 0.0,
    affinity_kernel         = "gaussian",
    kernel_params           = [1.0],
    use_interface_model     = False,

    # --- Concentration ---
    conc_model_type  = "lognormal",
    conc_mean_range  = (-7.0, -5.0),
    conc_std_range   = (1.0,  1.0),
    p_presence_range = (0.1,  0.5),

    # --- Physics ---
    n_genes     = 5,
    k_sub       = 5,
    temperature = 0.1,

    # --- Heteromers ---
    n_receptors                = 20,
    receptor_sampling_strategy = "cascading",
    receptor_sampling_seed     = 0,

    # --- Mixture ---
    batch_size      = "auto",
    test_batch_size = "auto",

    # --- Loss ---
    entropy      = "shannon",
    cov_weight   = 1.0,
    penalty_type = "repulsion",
    n_c_bins     = 10,

    # --- Training ---
    epochs        = 5000,
    lr            = 0.05,
    use_scheduler = False,
    measurement_fns = [
        "full_array_entropy",
        #"codeword_entropy",
        #"mean_receptor_distance",
        #"mean_specialization_index",
        #"mutual_information_ligand",
        #"mutual_information_concentration",
        #"mutual_information_family",
    ],

    # --- Sweep control ---
    n_samples       = 1,
    sweep_name      = "latent_dim_mi_heteromers",
    base_folder     = "/app/data/latent_dim_mi_heteromers",
    warm_start_axis = None,
    seed            = 0,
)

print(config)

t0 = time.time()
SweepRunner(config).execute()
h, rem = divmod(time.time() - t0, 3600)
m, s   = divmod(rem, 60)
print(f"\nLatent-dim × MI (heteromers) sweep complete!  {int(h)}h {int(m)}m {s:.0f}s")
