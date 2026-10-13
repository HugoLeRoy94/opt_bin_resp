#!/usr/bin/env python3
"""Does a cell array reach the single-ligand identity limit of log2(N_LIGANDS) bits?

A sanity target, not a comparison. The world holds almost always exactly one of 100
ligands at a nearly fixed concentration, so the input is a categorical variable
carrying log2(100) = 6.64 bits and nothing else. If the array cannot approach that,
the problem is the array or the optimizer, not the task.

Every parameter is a constant below. No command line, nothing imported from another
task script.

Readout: mean. With one receptor per cell a cell's firing probability IS that
receptor's open probability.

    python3 tasks/cells/convergence/scripts/convergence.py
    bash tasks/run_remote.sh cells/convergence convergence.py 0
"""
import math
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

# ─── Cells: eight cells, one homomer each, declared outright ────────────────
# Listing the repertoires directly guarantees eight DISTINCT cells. Sampling gene
# sets independently would allow duplicates, and duplicate cells are one group to
# the grouped estimator, which would quietly change the alphabet.
N_GENES = 10
K_SUB   = 5
N_CELLS = 8
CELL_RECEPTORS = tuple((((gene,) * K_SUB),) for gene in range(N_CELLS))

# ─── Objective and measurements ─────────────────────────────────────────────
# kt_mi brackets full-sniff information between a lower and an upper bound.
# identity_channel / concentration_channel split H(A) by presence pattern; that is
# only estimable because MU_LIGANDS keeps sniffs single-ligand, so patterns repeat.
# Raise MU_LIGANDS and those two columns stop meaning anything.
ENTROPY = "kt_mi"
MEASUREMENTS = ("entropy_kt", "entropy_kt_upper", "conditional_entropy_response",
                "mutual_information_kt", "mutual_information_kt_upper",
                "identity_channel", "concentration_channel", "codeword_entropy")

# ─── Training ───────────────────────────────────────────────────────────────
EPOCHS           = 5000
LR               = 1e-2
BATCH_SIZE       = 4096
TEST_BATCH_SIZE  = 4096
TEMPERATURE      = 0.05
INIT_TEMPERATURE = 3.0
WORLD_SEED       = 20260107

BASE_FOLDER = "/app/data/convergence"
SWEEP_NAME  = "cell_convergence"


def main():
    print(f"{N_CELLS} cells, one homomer each; target = ligand identity = "
          f"{math.log2(N_LIGANDS):.4f} bits")
    print(f"{N_LIGANDS} ligands, mu={MU_LIGANDS:g} per sniff, concentration ~fixed "
          f"(std {CONC_STD:g} decades)")

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

        # --- Cells ---
        cell_receptors = CELL_RECEPTORS,
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
    print(f"\nconvergence finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
