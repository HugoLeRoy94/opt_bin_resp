#!/usr/bin/env python3
"""Everything the two equivalence runs must share, in one place.

Both runs share the receptor list, initial world seed, and KT mutual-information
objective. This is a qualitative comparison of optimized endpoints: cell calibration
consumes additional sniffs and threshold cells have a second sharpening phase.
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
SEED    = 0        # fixes the initial world; calibration changes subsequent RNG draws
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

    # --- Cell readout (ignored by the receptor run) ---
    # phase 1 = 80% of epochs, matching the receptor run's own annealing window, so the
    # receptor temperature follows a comparable schedule in both. The remaining 20% is where
    # the receptor run merely holds T at its final value while the cell run additionally
    # hardens its readout — the one structural difference, and an unavoidable one: the
    # receptor array is already binarised by its own temperature and has nothing to
    # harden. See the note in as_cells.py.
    cell_phase_split = 0.5,
    # Receptor MOLECULES per cell. Floors theta at 1/N, which is what makes the
    # threshold readout track the receptor code at all (doc/theory/09 §9.8.1).
    cell_n_molecules = 1e4,

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
    """Share the initial world; later sniff sequences may differ due to calibration."""
    torch.manual_seed(SEED)
