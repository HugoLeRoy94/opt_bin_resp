#!/usr/bin/env python3
"""fig2 arm: INDEPENDENT (bernoulli) sampling, HOMOMERS ONLY.

Gene sets drawn independently per gene at a uniform probability, with no size
law; each cell is pinned to the homomers of its genes.

One of four arms. The other three are complete_heteromers.py, complete_homomers.py, bernoulli_heteromers.py. All four hold the same
environment, the same G x C grid and the same objective, so the only thing that
differs across the figure is how gene sets are drawn and what each cell may
assemble.

Every parameter is a constant below. No command line.

Grid: G in {2,3,4,5} x C in {10,20,30,50} x genes-per-cell 1..G, which is
56 independent optimizations.

    python3 tasks/cells/fig2/scripts/bernoulli_homomers.py
    bash tasks/run_remote.sh cells/fig2 bernoulli_homomers.py 0
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
from _lib.repertoires import (bernoulli_gene_sets, count_alphabet, heterogeneous_conc_means, homomers_only)

# ─── Environment ────────────────────────────────────────────────────────────
N_LIGANDS     = 100
LATENT_DIM    = 6
FAMILY_SPREAD = 0.1
FAMILY_DIST   = 1.0
GEOMETRY      = "asymmetric"
DISTRIBUTION  = "gaussian"
NOISE_SIGMA   = 0.0
MU_LIGANDS    = 3.0    # ZTP rate: ~3 of the 100 ligands present per sniff
CONC_SPAN     = 1.0    # log10 mean drawn in [-1, +1] -> 2 decades of variation
CONC_STD      = 1.0    # 1 decade of log10 spread per ligand, per sniff
CONC_SEED     = 44

# ─── The design ─────────────────────────────────────────────────────────────
N_GENES_LIST = (2, 3, 4, 5)
N_CELLS_LIST = (10, 20, 30, 50)
K_SUB        = 5
CELL_SEED    = 0

# ─── Objective and measurements ─────────────────────────────────────────────
# kt_mi trains; grouped_counting is the ONLY measurement, and it already carries
# everything the analyses need:
#   mutual_information_grouped_counting_mm        MI
#   response_entropy_grouped_counting_mm          H(Y)
#   conditional_entropy_response_grouped_counting H(Y|X)   (analytic, not sampled)
#   grouped_counting_K_hat / _missing_mass        alphabet coverage
# They are one estimator family, so MI = H(Y) - H(Y|X) holds exactly in the figure.
# The runner builds its own cell grouping, so a grouped loss is not required.
ENTROPY      = "kt_mi"
MEASUREMENTS = ("grouped_counting",)

# ─── Training ───────────────────────────────────────────────────────────────
EPOCHS           = 5000
LR               = 1e-2
BATCH_SIZE       = 4096
TEST_BATCH_SIZE  = 4096
FINAL_BATCH_SIZE = 2 ** 20   # scripts/evaluation_budget.py re-measures at other budgets
EVAL_CHUNK_SIZE  = 512
TEMPERATURE      = 0.05
INIT_TEMPERATURE = 3.0
WORLD_SEED       = 20260204

BASE_FOLDER = "/app/data/fig2"
SWEEP_NAME  = "fig2_bernoulli_homomers"


def design():
    """One row per planned run: every (G, C, genes per cell) in the grid.

    The seed is unique per row, so the analysis can join a finished run back to the
    design point it was supposed to be.
    """
    rows = []
    for n_genes in N_GENES_LIST:
        for n_cells in N_CELLS_LIST:
            for target in range(1, n_genes + 1):
                seed = CELL_SEED * 100000 + n_genes * 10000 + n_cells * 10 + target
                # Independent expression: each gene on at probability p, the SAME p for
                # every gene, so the number per cell is induced (Binomial) rather than
                # chosen. p is solved for the target mean; empty cells are redrawn, so
                # the realised mean sits slightly above the target at target=1.
                sets, probability, realised = bernoulli_gene_sets(
                    n_genes, n_cells, float(target), seed)
                rows.append(dict(
                    strategy="bernoulli", condition="homomers",
                    n_genes=n_genes, n_cells=n_cells, cell_sampling_seed=seed,
                    target_genes=float(target), realised_genes=realised,
                    gene_probability=probability, gene_sets=sets,
                    distinct_gene_sets=len(set(sets)),
                    genes_seen=len({g for gs in sets for g in gs}),
                    count_states=count_alphabet(sets)))
    return rows


ROWS = design()


def main():
    print(f"{'G':>2} {'C':>4} {'target':>7} {'realised':>9} {'p':>6} "
          f"{'types':>6} {'seen':>5} {'count states':>13}")
    for row in ROWS:
        print(f"{row['n_genes']:>2} {row['n_cells']:>4} {row['target_genes']:>7.1f} "
              f"{row['realised_genes']:>9.2f} {row['gene_probability']:>6.3f} "
              f"{row['distinct_gene_sets']:>6} {row['genes_seen']:>3}/{row['n_genes']:<2} "
              f"{row['count_states']:>13,}")
    print(f"\n{len(ROWS)} runs; {N_LIGANDS} ligands, mu={MU_LIGANDS:g} per sniff, "
          f"conc means over {2 * CONC_SPAN:g} decades, std {CONC_STD:g} decades")

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
        n_genes             = [r["n_genes"] for r in ROWS],
        k_sub               = K_SUB,
        temperature         = TEMPERATURE,
        initial_temperature = INIT_TEMPERATURE,
        affinity_kernel     = "gaussian",
        kernel_params       = (1.0,),
        use_interface_model = True,

        # --- Cells ---
        cell_receptors     = [homomers_only(r["gene_sets"], K_SUB) for r in ROWS],
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
        # No per-epoch evaluation: stats.csv still gets `loss` and
        # `train_mutual_information` every 1% of epochs, reusing the objective already
        # computed for the gradient step, so the convergence trace costs nothing.
        per_epoch_measure     = False,

        # --- Sweep ---
        sweep_name  = SWEEP_NAME,
        base_folder = BASE_FOLDER,
        warm_start  = False,
    )
    manifest = dict(
        experiment="fig2", strategy="bernoulli", condition="homomers",
        arguments=dict(entropy=ENTROPY, mu_ligands=MU_LIGANDS, conc_span=CONC_SPAN,
                       conc_std=CONC_STD, n_ligands=N_LIGANDS, epochs=EPOCHS,
                       batch_size=BATCH_SIZE, final_batch_size=FINAL_BATCH_SIZE,
                       world_seed=WORLD_SEED, cell_seed=CELL_SEED),
        rows=[dict(r, gene_sets=[list(g) for g in r["gene_sets"]]) for r in ROWS],
    )
    t0 = time.time()
    runner = SweepRunner(config)
    (Path(runner.master_logger.sweep_root) / "experiment.json").write_text(
        json.dumps(manifest, indent=2))
    runner.execute()
    print(f"\nfig2_bernoulli_homomers finished in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
