#!/usr/bin/env python3
"""Everything the two equivalence runs must share, in one place.

The check is only meaningful if the two runs differ in EXACTLY one respect — cell
array vs receptor array. Any other difference (a different receptor list, a different
environment draw, a different schedule) invalidates it, so both live here and neither
script is allowed its own copy.
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
SEED    = 0        # torch RNG: fixes the environment draw AND the sniff sequence

COMMON = dict(
    # --- Environment ---
    n_families              = 3,
    n_ligands               = 12,
    latent_dim              = 3,
    family_spread           = 0.3,
    average_family_distance = 2.0,
    environment_geometry    = "asymmetric",
    distribution_type       = "gaussian",
    observation_noise_sigma = 0.05,

    # --- Presence ---
    n_presence_blocks      = 1,
    mu_sources             = 1.0,
    mu_ligands_per_source  = 2.0,
    block_shared_conc_mean = False,

    use_interface_model = False,

    # --- Concentration ---
    conc_model_type = "lognormal",
    conc_mean       = (0.0,) * 12,
    conc_std        = (1.0,) * 12,

    # --- Physics ---
    n_genes=N_GENES, k_sub=5, temperature=0.1,
    affinity_kernel="gaussian", kernel_params=(1.0,),

    # --- Cell readout (ignored by the receptor run) ---
    # phase 1 = 80% of epochs, matching the receptor run's own annealing window, so the
    # receptor temperature follows the SAME schedule in both. The remaining 20% is where
    # the receptor run merely holds T at its final value while the cell run additionally
    # hardens its readout — the one structural difference, and an unavoidable one: the
    # receptor array is already binarised by its own temperature and has nothing to
    # harden. See the note in as_cells.py.
    cell_phase_split = 0.8,
    # Receptor MOLECULES per cell. Floors theta at 1/N, which is what makes the
    # threshold readout track the receptor code at all (doc/theory/09 §9.8.1).
    cell_n_molecules = 1e4,

    # --- Loss / training ---
    entropy="kt",
    epochs=300, lr=1e-2, use_scheduler=False,
    batch_size=2048, test_batch_size=2048,
    measurement_fns=("entropy_kt", "entropy_kt_upper", "codeword_entropy"),

    base_folder = "/app/data/equivalence",
    warm_start  = False,
)


def seed_everything():
    """Both runs must draw the SAME environment and the SAME sniffs.

    The environment is initialised from the torch RNG and every batch is sampled from
    it, so seeding here makes the two trajectories comparable step for step. Cell-set
    construction uses `random.Random` instead, and does not disturb this stream.
    """
    torch.manual_seed(SEED)
