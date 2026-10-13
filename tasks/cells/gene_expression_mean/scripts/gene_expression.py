#!/usr/bin/env python3
"""Full-repertoire cells swept over the number of genes expressed per cell.

Each cell expresses exactly the swept number of genes and assembles EVERY
receptor those genes allow (homomers AND heteromers), with multinomial
abundance. Pair with gene_expression_homomers.py, which pins the same cells to
their homomers only: the gap between the two curves is the heteromer
contribution. Gene identities are sampled with a fixed seed; different cells may
share a repertoire.
"""
import sys
import time
import torch

sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

N_LIG = 100
N_CELLS = 10
N_GENES = 5
MEAN_GENES = list(range(1, N_GENES + 1))
CELL_SEED = 0
WORLD_SEED = 0

# Presence: ZTP rate for the number of ligands drawn per sniff (one source block,
# so this is the only presence knob in play). > 1 puts the task in the mixture
# regime, where identity_channel / concentration_channel stop being estimable.
MU_LIGANDS = 0.0001

# Concentration: per-ligand log10 mean, drawn uniformly in [-SPAN, +SPAN], so
# ligands differ in typical abundance by up to 2*SPAN decades. One independent
# draw per sweep point, matching the convention that worlds are not matched
# across expression levels. CONC_SEED differs from the homomer script's, so the
# two conditions do not share a concentration world either.
CONC_LOG10_SPAN = 2.0
CONC_SEED = 1

MEASUREMENTS = (
    "full_array_entropy", "grouped_information", "conditional_entropy_response",
    "identity_channel", "concentration_channel", "codeword_entropy",
)


def conc_means(n_points):
    """One independent per-ligand log10 concentration-mean vector per sweep point."""
    gen = torch.Generator().manual_seed(CONC_SEED)
    return [tuple(((torch.rand(N_LIG, generator=gen) * 2.0 - 1.0)
                   * CONC_LOG10_SPAN).tolist())
            for _ in range(n_points)]


def main():
    # Seed the sequence, not each run: worlds are independent across sweep points.
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
        mu_ligands_per_source  = MU_LIGANDS,
        block_shared_conc_mean = False,

        # --- Interface model ---
        use_interface_model = True,

        # --- Concentration: the list of mean vectors is zipped with the cell axis ---
        conc_model_type = "lognormal",
        conc_mean       = conc_means(len(MEAN_GENES)),
        conc_std        = (1.,) * N_LIG,

        # --- Physics ---
        n_genes=N_GENES, k_sub=5, temperature=0.05, initial_temperature=3.0,
        affinity_kernel="gaussian", kernel_params=(1.0,),

        # --- Cells: the list of PMF tuples is the runner's sweep axis ---
        n_cells                = N_CELLS,
        cell_sampling_strategy = "size_pmf",
        cell_size_pmf          = [(0.0,) * (g - 1) + (1.0,) for g in MEAN_GENES],
        cell_sampling_seed     = CELL_SEED,
        cell_stoichiometry     = "multinomial",
        cell_readout           = "mean",

        # --- Loss ---
        entropy="kt_mi",

        # --- Training ---
        epochs=5000, lr=1e-2, use_scheduler=False,
        batch_size=4096, test_batch_size=4096,
        measurement_fns=MEASUREMENTS,
        final_measurement_fns=MEASUREMENTS + ("mutual_information_counting",),

        # --- Sweep ---
        sweep_name  = "cell_gene_expression",
        base_folder = "/app/data/gene_expression",
        warm_start  = False,
    )

    print(config)
    print(f"Mean genes/cell: {MEAN_GENES}; {N_CELLS} cells, {N_GENES} available genes")
    print(f"~{MU_LIGANDS} ligands/sniff; conc means spread over "
          f"{2 * CONC_LOG10_SPAN:g} decades")
    t0 = time.time()
    SweepRunner(config).execute()
    h, rem = divmod(time.time() - t0, 3600)
    m, s = divmod(rem, 60)
    print(f"\nGene-expression sweep complete!  {int(h)}h {int(m)}m {s:.0f}s")


if __name__ == "__main__":
    main()
