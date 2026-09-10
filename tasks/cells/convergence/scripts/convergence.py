#!/usr/bin/env python3
"""Cell-mode sanity check: does the optimizer converge to a KNOWN answer?

The world is crippled so its entropy is a number we can write down: exactly ONE
ligand per sniff (mu_ligands_per_source -> 0), drawn uniformly from N_LIG, at a
fixed concentration (conc_std ~ 0; not exactly 0, torch's Normal needs scale > 0).
A sniff therefore carries one fact — which ligand arrived — so

    H(world) = log2(N_LIG) = 3 bits,  and the target is 8 distinct codewords.

N_CELLS = 6 > 3 bits, so the ceiling is set by the WORLD, not by the array. The
target is a specific number: overshooting it is as much a failure as falling short.

What to read in test_results.json (all standard measurement_fns, no custom code):

  identity_channel        I(A ; which ligand)  -> 3.00    the real information
  concentration_channel   H(A | which ligand)  -> 0       must stay ~0, see below
  full_array_entropy_kt   KT bracket on H(A)   -> 3.00    = the two above summed
  codeword_entropy_K_hat  distinct codewords   -> 8

concentration_channel is the check that matters. The concentration is fixed, so
conditioning on the ligand fixes the input completely and anything left is readout
softness, not signal. A cell parked at activity 0.5 is a coin: it inflates
full_array_entropy_kt while telling you nothing, and that degenerate solution reports
the MAXIMUM entropy (doc/theory/07 §3b.6-3b.7). It shows up here as
concentration_channel > 0 with identity_channel short of 3.

Convergence is slow — measured on CPU at 8 ligands / 6 cells / batch 2048:
epoch 400 gave 1.06 bits and 3 codewords, epoch 3000 gave 2.75 bits and 7 codewords,
still climbing, coin term 1e-4 throughout. 7/8 codewords is under-trained, not broken.

  python3 convergence.py
  ../../run_remote.sh cells/convergence convergence.py 0
"""
import time
import sys
import math
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner

N_LIG      = 8      # world entropy is log2(N_LIG)
N_CELLS    = 6      # > log2(N_LIG), so the world is the binding constraint
N_GENES    = 10
GENES_CELL = 3      # every cell expresses exactly this many genes


def main():
    config = RunConfig(
        # --- Environment: one ligand per sniff, fixed concentration ---
        n_families              = 3,
        n_ligands               = N_LIG,
        latent_dim              = 10,
        family_spread           = 0.3,
        average_family_distance = 1.0,      # families far apart -> ligands separable
        environment_geometry    = "asymmetric",
        distribution_type       = "gaussian",
        observation_noise_sigma = 0.0,

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
        entropy="kt",

        # --- Training ---
        epochs=5000, lr=1e-2, use_scheduler=False,
        batch_size=4096, test_batch_size=4096,
        # identity_channel + concentration_channel = H(A) exactly, so the split into
        # signal and readout softness comes straight out of the standard registry.
        measurement_fns=("entropy_kt", "entropy_kt_upper",
                         "identity_channel", "concentration_channel",
                         "codeword_entropy"),

        # --- Sweep ---
        sweep_name  = "cell_convergence",
        base_folder = "/app/data/convergence",
        warm_start  = False,
    )

    print(config)
    print(f"TARGET: identity_channel -> {math.log2(N_LIG):.4f} bits, "
          f"concentration_channel -> 0, K_hat -> {N_LIG}")

    t0 = time.time()
    SweepRunner(config).execute()
    h, rem = divmod(time.time() - t0, 3600)
    m, s = divmod(rem, 60)
    print(f"\nCell convergence check complete!  {int(h)}h {int(m)}m {s:.0f}s")


if __name__ == "__main__":
    main()
