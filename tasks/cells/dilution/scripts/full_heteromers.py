#!/usr/bin/env python3
"""dilution arm: every receptor built from FIVE genes, one per subunit (five-gene heteromers).

The fig2 sweep moves genes-per-cell, which moves FOUR things at once: the receptor
pool, the granularity of each cell's weighted drive, the dilution of that drive, and
the diversity of cell types. This task pins three of them and moves one.

Every run here holds:
  - the same POOL SIZE (8 receptors), so pool diversity is constant;
  - the same CELL SUBSETS (shared_subsets with one seed, applied to every arm), so
    the number of distinct cell types and their multiplicities are IDENTICAL across
    the three arms at every point;
  - the same C = 50 cells and the same environment.

The sweep axis is RECEPTORS PER CELL, which is the dilution knob on its own: with
the mean readout a cell's firing probability is the uniform average of its
receptors' open probabilities, so averaging more of them pushes the drive toward
the middle and turns the cell into a coin.

The three arms differ ONLY in how many distinct genes form each receptor. Under the
interface model a pocket sits between adjacent subunits, so a homomer presents
k_sub copies of ONE pocket while a fully mixed word presents k_sub different ones.
That is heteromerisation per se, with dilution held fixed.

    python3 tasks/cells/dilution/scripts/full_heteromers.py
    bash tasks/run_remote.sh cells/dilution full_heteromers.py 0
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
from _lib.repertoires import heterogeneous_conc_means, receptor_pool, shared_subsets

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
CONC_SEED     = 53

# ─── The design ─────────────────────────────────────────────────────────────
N_GENES    = 8      # >= N_POOL, because the homomer arm needs N_POOL distinct homomers
K_SUB      = 5
N_POOL     = 8      # pool diversity, HELD FIXED across arms and across the sweep
N_CELLS    = 50
N_DISTINCT = 5      # genes per receptor: THE variable this task isolates
RECEPTORS_PER_CELL = (1, 2, 3, 4, 6, 8)   # the dilution axis
POOL_SEED  = 0
CELL_SEED  = 0      # SHARED by all three arms -> identical cell-type structure

# ─── Objective and measurements ─────────────────────────────────────────────
ENTROPY      = "kt_mi"
MEASUREMENTS = ("grouped_counting",)

# ─── Training ───────────────────────────────────────────────────────────────
EPOCHS           = 5000
LR               = 1e-2
BATCH_SIZE       = 4096
TEST_BATCH_SIZE  = 4096
FINAL_BATCH_SIZE = 2 ** 20
EVAL_CHUNK_SIZE  = 512
TEMPERATURE      = 0.05
INIT_TEMPERATURE = 3.0
WORLD_SEED       = 20260303

BASE_FOLDER = "/app/data/dilution"
SWEEP_NAME  = "dilution_full_heteromers"

# The pool: N_POOL receptors, each formed from exactly N_DISTINCT genes.
POOL = receptor_pool(N_GENES, K_SUB, N_DISTINCT, N_POOL, POOL_SEED)


def design():
    """One row per point on the dilution axis.

    shared_subsets returns POOL INDICES, and CELL_SEED is the same in all three
    scripts, so cell c holds the same slot of the pool in every arm. The arms are
    therefore matched cell-type for cell-type and differ only in pool content.
    """
    rows = []
    for per_cell in RECEPTORS_PER_CELL:
        subsets = shared_subsets(N_CELLS, N_POOL, per_cell, CELL_SEED)
        rows.append(dict(
            n_distinct=N_DISTINCT, receptors_per_cell=per_cell,
            n_pool=N_POOL, n_cells=N_CELLS,
            cell_sampling_seed=CELL_SEED * 1000 + per_cell,
            cell_receptors=tuple(tuple(POOL[i] for i in subset) for subset in subsets),
            distinct_repertoires=len(set(subsets)),
            genes_seen=len({g for subset in subsets for i in subset for g in POOL[i]})))
    return rows


ROWS = design()


def main():
    print(f"pool of {N_POOL} receptors, {N_DISTINCT} distinct gene(s) each "
          f"(five-gene heteromers):")
    for receptor in POOL:
        print(f"   {receptor}")
    print(f"\n{'per cell':>9} {'types':>6} {'genes seen':>11}")
    for row in ROWS:
        print(f"{row['receptors_per_cell']:>9} {row['distinct_repertoires']:>6} "
              f"{row['genes_seen']:>8}/{N_GENES:<2}")
    print(f"\n{len(ROWS)} runs; C={N_CELLS} cells, {N_LIGANDS} ligands, "
          f"mu={MU_LIGANDS:g} per sniff, conc means over {2 * CONC_SPAN:g} decades")

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

        # --- Cells: explicit repertoires, so weights are UNIFORM over the listed
        # receptors. That is what makes "receptors per cell" a clean dilution knob:
        # the drive is the plain average, with no multinomial weighting on top.
        cell_receptors     = [r["cell_receptors"] for r in ROWS],
        cell_sampling_seed = [r["cell_sampling_seed"] for r in ROWS],
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
        experiment="dilution", n_distinct=N_DISTINCT, label="five-gene heteromers",
        arguments=dict(entropy=ENTROPY, n_genes=N_GENES, n_pool=N_POOL,
                       n_cells=N_CELLS, mu_ligands=MU_LIGANDS, conc_span=CONC_SPAN,
                       conc_std=CONC_STD, epochs=EPOCHS, world_seed=WORLD_SEED,
                       pool_seed=POOL_SEED, cell_seed=CELL_SEED),
        pool=[list(r) for r in POOL],
        rows=[dict(r, cell_receptors=[[list(x) for x in c] for c in r["cell_receptors"]])
              for r in ROWS],
    )
    t0 = time.time()
    runner = SweepRunner(config)
    (Path(runner.master_logger.sweep_root) / "experiment.json").write_text(
        json.dumps(manifest, indent=2))
    runner.execute()
    print(f"\ndilution_full_heteromers finished in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
