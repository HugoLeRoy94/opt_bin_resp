#!/usr/bin/env python3
"""fig3 arm: k_sub = 5. Does the MI maximum move when receptors have 5 subunits?

In fig2 the maximum sits at 2 genes per cell for every G and every C >= 20, and it
will not move. The reason is that under the mean readout the receptor POOL and the
averaging WIDTH are the same function of g: a cell expressing g genes assembles
(g^k + ...)/k receptors, and that same number is what its drive averages over. The
benefit and the cost rise together.

k_sub changes how fast both rise. Pool size per genes/cell, G=5, C=50:

    k=2   5  15  15  15  15        per cell   1   3   6  10  15
    k=3   5  25  45  45  45        per cell   1   4  11  24  45
    k=5   5  65 365 605 629        per cell   1   8  51 208 629

Ratio of pool gain to dilution gain, step by step:

    k_sub 1->2  2->3  3->4  4->5
        2  1.00  0.50  0.60  0.67
        3  1.25  0.65  0.46  0.53
        5  1.62  0.88  0.41  0.34

A step pays off only above 1. So the prediction is that the peak moves LEFT as k_sub
falls: g=1 at k=2, g=2 at k=3 and k=5. At k=2 the pool is already exhausted at g=2,
which makes that arm a clean pure-dilution control: everything past g=2 is dilution
with zero pool gain.

Heteromers only. The peak does not exist in the homomer arm of fig2, so there is
nothing there to move.

Every parameter is a constant below. No command line. The other two arms are
k2.py, k3.py; the three files are identical except for K_SUB, the seeds and the
sweep name.

    python3 tasks/cells/fig3/scripts/k5.py
    bash tasks/run_remote.sh cells/fig3 k5.py 0
"""
import json
import sys
import time
from pathlib import Path

sys.path.append('/app')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.config import RunConfig
from src.run import SweepRunner
from _lib.budget import counting_budget_report
from _lib.repertoires import (complete_coverage_gene_sets, count_alphabet,
                              heterogeneous_conc_means)

# ─── Environment: identical to tasks/cells/fig2 ─────────────────────────────
N_LIGANDS     = 100
LATENT_DIM    = 6
FAMILY_SPREAD = 0.1
FAMILY_DIST   = 1.0
GEOMETRY      = "asymmetric"
DISTRIBUTION  = "gaussian"
NOISE_SIGMA   = 0.0
MU_LIGANDS    = 3.0
CONC_SPAN     = 1.0
CONC_STD      = 1.0
CONC_SEED     = 73

# ─── The design ─────────────────────────────────────────────────────────────
K_SUB          = 5      # THE variable this task isolates
N_GENES        = 5
N_CELLS        = 50
GENES_PER_CELL = (1, 2, 3, 4, 5)
REPLICATES     = 3        # independent worlds, for the error band
CELL_SEED      = 0        # SHARED across the three arms -> identical gene sets

# ─── Objective and measurements ─────────────────────────────────────────────
ENTROPY      = "kt_mi"
MEASUREMENTS = ("grouped_counting",)

# ─── Training ───────────────────────────────────────────────────────────────
EPOCHS           = 5000
LR               = 1e-2
BATCH_SIZE       = 4096
TEST_BATCH_SIZE  = 4096
FINAL_BATCH_SIZE = 2 ** 26
EVAL_CHUNK_SIZE  = 512
TEMPERATURE      = 0.05
INIT_TEMPERATURE = 3.0
MAX_BIAS         = 0.2
WORLD_SEED       = 20260503

BASE_FOLDER = "/app/data/fig3"
SWEEP_NAME  = "fig3_k5"


def design():
    """One row per (replicate, genes per cell).

    CELL_SEED is the same in all three scripts, so the gene sets are identical across
    k_sub and only the receptors those genes assemble differ.
    """
    rows = []
    for replicate in range(REPLICATES):
        for genes_per_cell in GENES_PER_CELL:
            sets = complete_coverage_gene_sets(N_GENES, N_CELLS, genes_per_cell,
                                               CELL_SEED)
            rows.append(dict(
                k_sub=K_SUB, replicate=replicate, n_genes=N_GENES,
                n_cells=N_CELLS, genes_per_cell=genes_per_cell,
                cell_sampling_seed=replicate * 1000 + genes_per_cell,
                gene_sets=sets, distinct_gene_sets=len(set(sets)),
                count_states=count_alphabet(sets)))
    return rows


ROWS = design()


def main():
    print(f"k_sub = {K_SUB}, G = {N_GENES}, C = {N_CELLS}\n")
    print(f"{'g':>2} {'types':>6} {'count states':>13}   (replicate 0 only)")
    for row in ROWS:
        if row["replicate"]:
            continue
        print(f"{row['genes_per_cell']:>2} {row['distinct_gene_sets']:>6} "
              f"{row['count_states']:>13,}")
    print(f"\n{len(ROWS)} runs ({REPLICATES} replicates); {N_LIGANDS} ligands, "
          f"mu={MU_LIGANDS:g} per sniff, conc means over {2 * CONC_SPAN:g} decades")

    counting_budget_report([r for r in ROWS if r["replicate"] == 0],
                           FINAL_BATCH_SIZE, limit=MAX_BIAS)

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
            len(ROWS), N_LIGANDS, CONC_SPAN, CONC_SEED),
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
        cell_gene_sets     = [r["gene_sets"] for r in ROWS],
        cell_sampling_seed = [r["cell_sampling_seed"] for r in ROWS],
        cell_stoichiometry = "multinomial",
        cell_readout       = "mean",
        cell_pool_chunk    = 128,
        recompute_backward = True,

        # --- Objective ---
        entropy               = ENTROPY,
        measurement_fns       = MEASUREMENTS,
        final_measurement_fns = MEASUREMENTS,

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
    manifest = dict(
        experiment="fig3", k_sub=K_SUB,
        arguments=dict(entropy=ENTROPY, n_genes=N_GENES, n_cells=N_CELLS,
                       mu_ligands=MU_LIGANDS, conc_span=CONC_SPAN, conc_std=CONC_STD,
                       epochs=EPOCHS, final_batch_size=FINAL_BATCH_SIZE,
                       replicates=REPLICATES, world_seed=WORLD_SEED,
                       cell_seed=CELL_SEED),
        rows=[dict(r, gene_sets=[list(g) for g in r["gene_sets"]]) for r in ROWS],
    )
    t0 = time.time()
    runner = SweepRunner(config)
    (Path(runner.master_logger.sweep_root) / "experiment.json").write_text(
        json.dumps(manifest, indent=2))
    runner.execute()
    print(f"\nfig3_k5 finished in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
