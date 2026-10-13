#!/usr/bin/env python3
"""Equivalence check, CELL side: C cells, each holding exactly ONE receptor.

Run this and as_receptors.py, then compare with analysis/equivalence.py. A cell holding
exactly one receptor IS that receptor, so cell mode must reproduce the receptor model
numerically. Any discrepancy diagnoses the cell plumbing, not a modelling difference.

This is the ONE task where both scripts deliberately share WORLD_SEED. It is a
numerical identity check, not a scientific comparison: the two runs must see the same
world or the check means nothing. Every other task under tasks/cells/ gives each arm
its own world on purpose.

Every parameter is a constant below, duplicated in as_receptors.py. The two files must stay
identical apart from the cell/receptor block and the sweep name — that is the whole
point, so the duplication is deliberate, not an oversight.

    python3 tasks/cells/equivalence/scripts/as_cells.py
    bash tasks/run_remote.sh cells/equivalence as_cells.py 0
"""
import sys
import time

sys.path.append('/app')

import torch

from src.config import RunConfig
from src.run import SweepRunner

# ─── Environment: one ligand per sniff at a nearly FIXED concentration ──────
N_LIGANDS     = 100
LATENT_DIM    = 6
FAMILY_SPREAD = 0.1
FAMILY_DIST   = 1.0
GEOMETRY      = "asymmetric"
DISTRIBUTION  = "gaussian"
NOISE_SIGMA   = 0.0
MU_LIGANDS    = 1e-6     # ZTP rate -> P(exactly one ligand) ~ 1
CONC_MEAN     = 0.0      # log10 concentration mean, same for every ligand
CONC_STD      = 1e-4     # ~fixed: the concentration channel contributes ~0 bits

# ─── The receptor list, SORTED ──────────────────────────────────────────────
# Sorting matters: CellArray's pool is the sorted unique union, so a sorted input
# makes the cell run's channel order identical to the receptor run's. Unsorted, W
# would be a permutation and a per-channel comparison would need undoing it (the
# entropy would still match, being permutation-invariant).
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
K_SUB   = 5

# ─── Objective ──────────────────────────────────────────────────────────────
ENTROPY = "kt_mi"
MEASUREMENTS = ("entropy_kt", "entropy_kt_upper", "conditional_entropy_response",
                "mutual_information_kt", "mutual_information_kt_upper",
                "codeword_entropy")

# ─── Training ───────────────────────────────────────────────────────────────
EPOCHS           = 5000
LR               = 1e-2
BATCH_SIZE       = 4096
TEST_BATCH_SIZE  = 4096
TEMPERATURE      = 0.05
INIT_TEMPERATURE = 3.0
WORLD_SEED       = 0        # SHARED with as_receptors.py, on purpose

BASE_FOLDER = "/app/data/equivalence"
SWEEP_NAME  = "as_cells"


def main():
    print(f"{len(RECEPTORS)} cells, one receptor each -> W is the identity, drive = p")

    torch.manual_seed(WORLD_SEED)
    config = RunConfig(
        # --- Environment ---
        n_families              = 1,
        n_ligands               = N_LIGANDS,
        latent_dim              = LATENT_DIM,
        family_spread           = FAMILY_SPREAD,
        average_family_distance = FAMILY_DIST,
        environment_geometry    = GEOMETRY,
        distribution_type       = DISTRIBUTION,
        observation_noise_sigma = NOISE_SIGMA,

        n_presence_blocks      = 1,
        mu_sources             = 1.0,
        mu_ligands_per_source  = MU_LIGANDS,
        block_shared_conc_mean = False,

        conc_model_type = "lognormal",
        conc_mean       = (CONC_MEAN,) * N_LIGANDS,
        conc_std        = (CONC_STD,) * N_LIGANDS,

        # --- Physics ---
        n_genes             = N_GENES,
        k_sub               = K_SUB,
        temperature         = TEMPERATURE,
        initial_temperature = INIT_TEMPERATURE,
        affinity_kernel     = "gaussian",
        kernel_params       = (1.0,),
        use_interface_model = True,

        # --- Cells: one cell per receptor, cell j holds exactly RECEPTORS[j] ---
        # Gene sets cannot express this for a heteromer: expressing the genes of
        # [0,0,0,1,1] also builds every other combination of 0 and 1. Only the
        # explicit path pins an arbitrary receptor into a cell. With RECEPTORS
        # sorted, W is exactly the identity and the drive is p itself.
        cell_receptors = tuple((r,) for r in RECEPTORS),
        cell_readout   = "mean",

        # --- Objective ---
        entropy               = ENTROPY,
        measurement_fns       = MEASUREMENTS,
        final_measurement_fns = MEASUREMENTS + ("mutual_information_counting",),

        # --- Training ---
        epochs          = EPOCHS,
        lr              = LR,
        use_scheduler   = False,
        batch_size      = BATCH_SIZE,
        test_batch_size = TEST_BATCH_SIZE,

        # --- Sweep ---
        sweep_name  = SWEEP_NAME,
        base_folder = BASE_FOLDER,
        warm_start  = False,
    )
    t0 = time.time()
    SweepRunner(config).execute()
    print(f"\nas_cells finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
