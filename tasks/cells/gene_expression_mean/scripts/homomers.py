#!/usr/bin/env python3
"""Genes expressed per cell, MEAN readout, HOMOMERS ONLY.

Same cells as heteromers.py, pinned to the homomers of their genes. No heteromer
is ever assembled. The gap against heteromers.py is the heteromer contribution.

Every parameter of this experiment is a constant below. Nothing is inherited from
another script, there is no command line, and the only import from tasks/ is the
gene-set generator. To change the experiment, edit a constant.

Readout: mean. A cell's firing probability IS its drive S_bc, the
abundance-weighted fraction of its receptors that are open. No threshold.

Sweep axis: GENES_PER_CELL. One independent run per entry, fresh world and fresh
optimizer at every point (warm_start=False). Worlds are NOT matched across points.

    python3 tasks/cells/gene_expression_mean/scripts/homomers.py
    bash tasks/run_remote.sh cells/gene_expression_mean homomers.py 0
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.append('/app')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.config import RunConfig
from src.run import SweepRunner
from _lib.repertoires import (complete_coverage_gene_sets, count_alphabet,
                              heterogeneous_conc_means, homomers_only)

# ─── Environment ────────────────────────────────────────────────────────────
N_LIGANDS     = 100
LATENT_DIM    = 6
FAMILY_SPREAD = 0.1
FAMILY_DIST   = 1.0
GEOMETRY      = "asymmetric"
DISTRIBUTION  = "gaussian"
NOISE_SIGMA   = 0.0

# Ligands present per sniff: zero-truncated Poisson rate, with one presence block
# this is the only presence knob. 1e-6 means effectively exactly one ligand per
# sniff. Anything above 1 is a MIXTURE, which is a different problem, not a tweak:
# it also breaks identity_channel / concentration_channel, which need presence
# patterns to repeat.
MU_LIGANDS = 1e-6

# Per-ligand log10 concentration mean, drawn uniformly in +/- CONC_SPAN, one
# independent vector per sweep point. 0.0 is the flat channel: every ligand has the
# same typical abundance. 2.0 spreads abundances over 4 decades.
CONC_SPAN = 0.0
CONC_STD  = 1.0
CONC_SEED = 12

# ─── Cells ──────────────────────────────────────────────────────────────────
N_GENES        = 5
N_CELLS        = 30
K_SUB          = 5
GENES_PER_CELL = (1, 2, 3, 4, 5)
CELL_SEED      = 0          # one seed for all levels -> gene sets are NESTED
READOUT        = "mean"

# ─── Objective and estimator ────────────────────────────────────────────────
ENTROPY      = "grouped_mi"      # exact enumeration of joint count states
MAX_STATES   = 1_048_576         # guard on prod_j (n_j + 1); see the table printed below
MEASUREMENTS = ("grouped_information", "full_array_entropy",
                "conditional_entropy_response", "codeword_entropy")

# ─── Training ───────────────────────────────────────────────────────────────
EPOCHS           = 5000
LR               = 1e-2
BATCH_SIZE       = 4096
TEST_BATCH_SIZE  = 4096
FINAL_BATCH_SIZE = 16384
EVAL_CHUNK_SIZE  = 512
TEMPERATURE      = 0.05
INIT_TEMPERATURE = 3.0
WORLD_SEED       = 20260102

BASE_FOLDER = "/app/data/gene_expression"
SWEEP_NAME  = "mean_homomers"

# Complete coverage: every gene is expressed by at least one cell. That requires
# coordination between cells which a developing tissue has no mechanism for, so this
# is a controlled baseline, not a biological model.
GENE_SETS = [complete_coverage_gene_sets(N_GENES, N_CELLS, g, CELL_SEED)
             for g in GENES_PER_CELL]

# Homomers only: a cell expressing {u, v} gets [(u,)*K_SUB, (v,)*K_SUB] and nothing
# else. Explicit repertoires bypass expansion and are weighted uniformly, which the
# gene-set path cannot express — expressing u and v necessarily builds every mixed
# receptor too.
CELL_RECEPTORS = [homomers_only(sets, K_SUB) for sets in GENE_SETS]


# ─── Provenance ─────────────────────────────────────────────────────────────
# Written next to the sweep before training. The analysis joins each run back to
# its planned design through (cell_sampling_seed, genes_per_cell), which catches a
# folder re-run under a different design but kept under its old name.
MANIFEST = dict(
    experiment = "gene_expression",
    condition  = "homomers",
    coverage   = "complete",
    arguments  = dict(cell_readout=READOUT, cell_threshold="-",
                      entropy=ENTROPY, evaluation="exact",
                      n_genes=N_GENES, n_cells=N_CELLS, mu_ligands=MU_LIGANDS,
                      conc_span=CONC_SPAN, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      world_seed=WORLD_SEED, cell_seed=CELL_SEED),
    rows       = [dict(n_genes=N_GENES, n_cells=N_CELLS, genes_per_cell=g,
                       cell_sampling_seed=CELL_SEED, count_states=count_alphabet(sets),
                       distinct_gene_sets=len(set(sets)), cell_gene_sets=sets)
                  for g, sets in zip(GENES_PER_CELL, GENE_SETS)],
)


def main():
    # grouped MI cannot exceed log2 of the count alphabet, and that alphabet is set
    # by how many DISTINCT repertoires the sampler drew — which is NOT monotonic in
    # genes per cell. Read this table before believing a bump in the MI curve.
    print(f"{'g':>2} {'cells':>6} {'distinct':>9} {'count states':>13} "
          f"{'grouped MI cap':>15} {'train table':>12}")
    for g, sets in zip(GENES_PER_CELL, GENE_SETS):
        states = count_alphabet(sets)
        print(f"{g:>2} {len(sets):>6} {len(set(sets)):>9} {states:>13,} "
              f"{math.log2(states):>12.2f} b {BATCH_SIZE * states * 4 / 2 ** 30:>9.2f} GiB")
        if states > MAX_STATES:
            raise ValueError(f"g={g}: {states:,} count states exceed MAX_STATES={MAX_STATES:,}.")
    print(f"\n{N_LIGANDS} ligands, mu={MU_LIGANDS:g} per sniff, "
          f"conc means over {2 * CONC_SPAN:g} decades, readout={READOUT}")

    torch.manual_seed(WORLD_SEED)
    config = RunConfig(
        # --- Environment: lists are zipped with the cell axis, never crossed ---
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
        conc_mean       = heterogeneous_conc_means(
            len(GENES_PER_CELL), N_LIGANDS, CONC_SPAN, CONC_SEED),
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
        cell_receptors     = CELL_RECEPTORS,
        cell_sampling_seed = CELL_SEED,
        cell_stoichiometry = "multinomial",
        cell_readout       = READOUT,
        cell_pool_chunk    = 128,
        recompute_backward = True,

        # --- Objective ---
        entropy                 = ENTROPY,
        cell_grouped_max_states = MAX_STATES,
        measurement_fns         = MEASUREMENTS,
        final_measurement_fns   = MEASUREMENTS,

        # --- Training ---
        epochs                = EPOCHS,
        lr                    = LR,
        use_scheduler         = False,
        batch_size            = BATCH_SIZE,
        test_batch_size       = TEST_BATCH_SIZE,
        final_test_batch_size = FINAL_BATCH_SIZE,
        eval_chunk_size       = EVAL_CHUNK_SIZE,
        per_epoch_measure     = False,

        # --- Sweep ---
        sweep_name  = SWEEP_NAME,
        base_folder = BASE_FOLDER,
        warm_start  = False,
    )
    t0 = time.time()
    runner = SweepRunner(config)
    (Path(runner.master_logger.sweep_root) / "experiment.json").write_text(
        json.dumps(MANIFEST, indent=2))
    runner.execute()
    print(f"\nmean_homomers finished in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
