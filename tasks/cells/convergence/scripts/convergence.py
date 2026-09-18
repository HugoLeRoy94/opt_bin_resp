#!/usr/bin/env python3
"""Qualitative cell convergence toward the approximately three-bit singleton target.

The world has almost always one of eight ligands at nearly fixed concentration.
The structured repertoire can limit attainability. Small concentration variation
and rare multi-ligand sniffs remain part of the input.

Training maximizes the KT mutual-information lower bound (entropy='kt_mi').
mutual_information_kt / _upper bracket full-sniff information; the separate
conditional_entropy_response records Bernoulli uncertainty, not signal. Final
mutual_information_counting_mm samples stochastic outputs and subtracts this term.
identity_channel conditions on the full ligand mask, while concentration_channel
also includes readout uncertainty. Their exact entropy calculation is affordable
here because there are only six cells; it is not required for larger-array MI.

  python3 convergence.py
  ../../run_remote.sh cells/convergence convergence.py 0
"""
import time
import sys
import math
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

N_LIG      = 100      # approximate singleton identity reference: log2(N_LIG)
N_CELLS    = 8
N_GENES    = 10
GENES_CELL = 1      # every cell expresses exactly this many genes
MEASUREMENTS = ("entropy_kt", "entropy_kt_upper", "conditional_entropy_response",
                "mutual_information_kt", "mutual_information_kt_upper",
                "identity_channel", "concentration_channel", "codeword_entropy")


def main():
    config = RunConfig(
        # --- Environment: approximately one ligand, nearly fixed concentration ---
        n_families              = 1,
        n_ligands               = N_LIG,
        latent_dim              = 6,
        family_spread           = 0.1,
        average_family_distance = 1.0,
        environment_geometry    = "asymmetric",
        distribution_type       = "gaussian",
        observation_noise_sigma = 0.0,
        initial_temperature=3.0,

        # --- Presence (hierarchical sampler) ---
        n_presence_blocks      = 1,
        mu_sources             = 1.0,
        mu_ligands_per_source  = 1e-6,      # ZTP -> P(exactly 1 ligand) ~ 1
        block_shared_conc_mean = False,

        # --- Interface model ---
        use_interface_model = True,

        # --- Concentration ---
        conc_model_type = "lognormal",
        conc_mean       = (0.0,) * N_LIG,
        conc_std        = (1e-4,) * N_LIG,  # ~fixed: contributes ~0 bits

        # --- Physics ---
        n_genes=N_GENES, k_sub=5, temperature=0.05,
        affinity_kernel="gaussian", kernel_params=(1.0,),

        # --- Cells ---
        n_cells                = N_CELLS,
        cell_sampling_strategy = "size_pmf",
        cell_size_pmf          = (0.0,) * (GENES_CELL - 1) + (1.0,),
        cell_sampling_seed     = 0,
        cell_stoichiometry     = "multinomial",
        cell_readout           = "threshold",

        # --- Loss ---
        entropy="kt_mi",

        # --- Training ---
        epochs=5000, lr=1e-2, use_scheduler=False,
        batch_size=4096, test_batch_size=4096,
        measurement_fns=MEASUREMENTS,
        final_measurement_fns=MEASUREMENTS + ("mutual_information_counting",),

        # --- Sweep ---
        sweep_name  = "cell_convergence",
        base_folder = "/app/data/convergence",
        warm_start  = False,
    )

    print(config)
    print(f"APPROXIMATE TARGET: mutual_information_kt -> {math.log2(N_LIG):.4f} bits, "
          f"hard K_hat -> {N_LIG}; the structured repertoire may limit attainment")

    t0 = time.time()
    SweepRunner(config).execute()
    h, rem = divmod(time.time() - t0, 3600)
    m, s = divmod(rem, 60)
    print(f"\nCell convergence check complete!  {int(h)}h {int(m)}m {s:.0f}s")


if __name__ == "__main__":
    main()
