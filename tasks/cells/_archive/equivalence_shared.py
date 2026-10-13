#!/usr/bin/env python3
"""Everything the two equivalence runs must share, in one place.

Both runs share the receptor list, initial world seed, and KT mutual-information
objective. The cell run uses the mean readout with one receptor per cell, making its
forward map identical to the receptor run when W is the identity.
"""
import sys
sys.path.append('/app')

import torch

# Fixed receptor list, SORTED. Sorting matters: CellArray's pool is the sorted unique
# union, so a sorted input makes the cell array's channel order identical to the
# receptor array's — otherwise W is a permutation and per-channel comparison needs
# undoing it (the entropy would still match, being permutation-invariant).
RECEPTORS = tuple(sorted([
    (0, 0, 0, 0, 0),
    (0, 0, 0, 0, 1),
    (0, 0, 0, 1, 1),
    (0, 0, 1, 1, 1),
    (1, 1, 1, 1, 2),
    (1, 1, 2, 2, 2),
    (2, 2, 2, 3, 3),
    (3, 3, 3, 3, 3),
]))

N_GENES = 5
SEED    = 0        # fixes the initial world and subsequent sniff sequence
N_LIG = 100

MEASUREMENTS = ("entropy_kt", "entropy_kt_upper", "conditional_entropy_response",
                "mutual_information_kt", "mutual_information_kt_upper", "codeword_entropy")

COMMON = dict(
    # --- Environment ---
    n_families              = 1,
    n_ligands               = N_LIG,
    latent_dim              = 6,
    family_spread           = 0.1,
    average_family_distance = 1.0,
    environment_geometry    = "asymmetric",
    distribution_type       = "gaussian",
    observation_noise_sigma = 0.0,
    initial_temperature=3.0,

    # --- Presence ---
    n_presence_blocks      = 1,
    mu_sources             = 1.0,
    mu_ligands_per_source  = 1.0e-6,
    block_shared_conc_mean = False,

    use_interface_model = True,

    # --- Concentration ---
    conc_model_type = "lognormal",
    conc_mean       = (0.0,) * N_LIG,
    conc_std        = (1e-4,) * N_LIG,

    # --- Physics ---
    n_genes=N_GENES, k_sub=5, temperature=0.05,
    affinity_kernel="gaussian", kernel_params=(1.0,),

    # --- Loss / training ---
    entropy="kt_mi",
    epochs=5000, lr=1e-2, use_scheduler=False,
    batch_size=4096, test_batch_size=4096,
    measurement_fns=MEASUREMENTS,
    final_measurement_fns=MEASUREMENTS + ("mutual_information_counting",),

    base_folder = "/app/data/equivalence",
    warm_start  = False,
)


def seed_everything():
    """Give both scripts the same initial world and sniff sequence."""
    torch.manual_seed(SEED)
