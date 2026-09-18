#!/usr/bin/env python3
"""Sweep genes expressed per cell while allowing homomeric receptors only."""
import sys
import time
import torch

sys.path.append('/app')

from src.cells import sample_gene_sets_by_size
from src.config import RunConfig
from src.run import SweepRunner

N_LIG = 100
N_CELLS = 10
N_GENES = 3
K_SUB = 5
MEAN_GENES = [1, 2, 3, 4, 5]
CELL_SEED = 0
WORLD_SEED = 0
MEASUREMENTS = (
    "entropy_kt", "entropy_kt_upper", "conditional_entropy_response",
    "mutual_information_kt", "mutual_information_kt_upper",
    "identity_channel", "concentration_channel", "codeword_entropy",
)


def homomer_repertoires(genes_per_cell):
    """Sample expressed genes, then give each cell only their homomers."""
    size_pmf = (0.0,) * (genes_per_cell - 1) + (1.0,)
    gene_sets = sample_gene_sets_by_size(
        N_CELLS, N_GENES, size_pmf, seed=CELL_SEED
    )
    return tuple(
        tuple((gene,) * K_SUB for gene in genes)
        for genes in gene_sets
    )


# A list is a SweepRunner axis; each tuple is one complete cell array.
CELL_RECEPTOR_SWEEP = [homomer_repertoires(g) for g in MEAN_GENES]


def main():
    torch.manual_seed(WORLD_SEED)

    config = RunConfig(
        # --- Environment ---
        n_families              = 1,
        n_ligands               = N_LIG,
        latent_dim              = 6,
        family_spread           = 0.1,
        average_family_distance = 1.0,
        environment_geometry    = "asymmetric",
        distribution_type       = "gaussian",
        observation_noise_sigma = 0.0,

        # --- Presence (hierarchical sampler) ---
        n_presence_blocks      = 1,
        mu_sources             = 1.0,
        mu_ligands_per_source  = 1e-6,
        block_shared_conc_mean = False,

        # --- Interface model ---
        use_interface_model = True,

        # --- Concentration ---
        conc_model_type = "lognormal",
        conc_mean       = (0.0,) * N_LIG,
        conc_std        = (1.,) * N_LIG,

        # --- Physics ---
        n_genes=N_GENES, k_sub=K_SUB, temperature=0.05,
        initial_temperature=3.0,
        affinity_kernel="gaussian", kernel_params=(1.0,),

        # --- Cells: explicit homomer-only repertoires; the list is the sweep ---
        cell_receptors = CELL_RECEPTOR_SWEEP,
        cell_readout   = "mean",

        # --- Loss ---
        entropy="kt_mi",

        # --- Training ---
        epochs=5000, lr=1e-2, use_scheduler=False,
        batch_size=4096, test_batch_size=4096,
        measurement_fns=MEASUREMENTS,
        final_measurement_fns=MEASUREMENTS + ("mutual_information_counting",),

        # --- Sweep ---
        sweep_name  = "homomer_gene_expression",
        base_folder = "/app/data/gene_expression",
        warm_start  = False,
    )

    print(config)
    print(f"Genes/cell: {MEAN_GENES}; homomers only; "
          f"{N_CELLS} cells, {N_GENES} available genes")
    t0 = time.time()
    SweepRunner(config).execute()
    h, rem = divmod(time.time() - t0, 3600)
    m, s = divmod(rem, 60)
    print(f"\nHomomer-only gene-expression sweep complete! "
          f"{int(h)}h {int(m)}m {s:.0f}s")


if __name__ == "__main__":
    main()
