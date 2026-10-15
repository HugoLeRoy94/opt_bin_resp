#!/usr/bin/env python3
"""fig2 arm: INDEPENDENT (bernoulli) sampling, HETEROMERS.

Gene sets drawn independently per gene at a uniform probability, with no size
law; each cell assembles EVERY receptor its genes allow.

One of four arms. The other three are complete_heteromers.py, complete_homomers.py, bernoulli_homomers.py. All four hold the same
environment, the same grid and the same objective, so the only thing that differs
across the figure is how gene sets are drawn and what each cell may assemble.

Every parameter is a constant below. No command line.

Grid: G in {2,3,4,5} x C in {20,30} x mean genes/cell from 1.0 to G in steps of
0.25, x 3 replicates = 264 independent optimizations.

Replicates are INDEPENDENT WORLDS, not repeated measurements: each row draws its own
latent space from the torch stream and its own concentration means, so the spread
across them is what belongs on an error bar. A gap between arms smaller than that
spread is not a result.

    python3 tasks/cells/fig2/scripts/bernoulli_heteromers.py
    bash tasks/run_remote.sh cells/fig2 bernoulli_heteromers.py 0
"""
import json
import sys
import time
from pathlib import Path

sys.path.append('/app')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.config import RunConfig
from src.run import SweepRunner
from _lib.budget import counting_budget_report
from _lib.repertoires import (bernoulli_gene_sets, count_alphabet, heterogeneous_conc_means)

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
CONC_SEED     = 63

# ─── The design ─────────────────────────────────────────────────────────────
# C = 50 is deliberately ABSENT. H(K|X) is a sum over cells, and at C=50 it reaches
# ~15 bits, which needs a counting budget of 2^30 (345 h) to stay under 0.2 bits of
# bias. At C=30 the same grid projects 0.04 bits at 2^26. The pre-flight below will
# refuse to launch if you put C=50 back without raising FINAL_BATCH_SIZE.
N_GENES_LIST = (2, 3, 4, 5)
N_CELLS_LIST = (20, 30)
GENE_STEP    = 0.25
K_SUB        = 5
REPLICATES   = 3
CELL_SEED    = 0

# ─── Objective and measurements ─────────────────────────────────────────────
# kt_mi trains; grouped_counting is the ONLY measurement and already carries
# everything the analyses need: MI, H(Y), H(Y|X) and alphabet coverage, as one
# estimator family so MI = H(Y) - H(Y|X) holds exactly.
ENTROPY      = "kt_mi"
MEASUREMENTS = ("grouped_counting",)

# ─── Training ───────────────────────────────────────────────────────────────
EPOCHS           = 5000
LR               = 1e-2
BATCH_SIZE       = 4096
TEST_BATCH_SIZE  = 4096
FINAL_BATCH_SIZE = 2 ** 26   # see the pre-flight: 2^20 made C=50 MI go NEGATIVE
EVAL_CHUNK_SIZE  = 512
TEMPERATURE      = 0.05
INIT_TEMPERATURE = 3.0
MAX_BIAS         = 0.2       # bits; above this the bias distorts the curve SHAPE
WORLD_SEED       = 20260403

BASE_FOLDER = "/app/data/fig2"
SWEEP_NAME  = "fig2_bernoulli_heteromers"


def design():
    """One row per planned run: replicate x G x C x mean genes per cell."""
    rows = []
    for replicate in range(REPLICATES):
        for n_genes in N_GENES_LIST:
            for n_cells in N_CELLS_LIST:
                for target in np.arange(1.0, n_genes + 1e-9, GENE_STEP):
                    target = round(float(target), 4)
                    seed = (replicate * 1000000 + n_genes * 10000
                            + n_cells * 100 + int(round(target * 4)))
                    # Independent expression: each gene on at probability p, the
                    # SAME p for every gene, so the number per cell is induced
                    # (Binomial) rather than chosen. p is solved for the target mean.
                    sets, probability, realised = bernoulli_gene_sets(
                        n_genes, n_cells, target, seed)
                    rows.append(dict(
                        strategy="bernoulli", condition="heteromers",
                        replicate=replicate, n_genes=n_genes, n_cells=n_cells,
                        cell_sampling_seed=seed, target_genes=target,
                        realised_genes=realised, gene_probability=probability,
                        gene_sets=sets, distinct_gene_sets=len(set(sets)),
                        genes_seen=len({g for gs in sets for g in gs}),
                        count_states=count_alphabet(sets)))
    return rows


ROWS = design()


def main():
    print(f"{'G':>2} {'C':>4} {'target':>7} {'realised':>9} {'p':>6} "
          f"{'types':>6} {'seen':>5}   (replicate 0 only)")
    for row in ROWS:
        if row["replicate"]:
            continue
        print(f"{row['n_genes']:>2} {row['n_cells']:>4} {row['target_genes']:>7.2f} "
              f"{row['realised_genes']:>9.2f} {row['gene_probability']:>6.3f} "
              f"{row['distinct_gene_sets']:>6} {row['genes_seen']:>3}/{row['n_genes']:<2}")
    print(f"\n{len(ROWS)} runs ({REPLICATES} replicates); {N_LIGANDS} ligands, "
          f"mu={MU_LIGANDS:g} per sniff, conc means over {2 * CONC_SPAN:g} decades")

    # Refuses to launch if sampled counting cannot resolve this design. Checked on
    # replicate 0, which has the same group structure as the others.
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
        n_genes             = [r["n_genes"] for r in ROWS],
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
        experiment="fig2", strategy="bernoulli", condition="heteromers",
        arguments=dict(entropy=ENTROPY, mu_ligands=MU_LIGANDS, conc_span=CONC_SPAN,
                       conc_std=CONC_STD, n_ligands=N_LIGANDS, epochs=EPOCHS,
                       batch_size=BATCH_SIZE, final_batch_size=FINAL_BATCH_SIZE,
                       gene_step=GENE_STEP, replicates=REPLICATES,
                       world_seed=WORLD_SEED, cell_seed=CELL_SEED),
        rows=[dict(r, gene_sets=[list(g) for g in r["gene_sets"]]) for r in ROWS],
    )
    t0 = time.time()
    runner = SweepRunner(config)
    (Path(runner.master_logger.sweep_root) / "experiment.json").write_text(
        json.dumps(manifest, indent=2))
    runner.execute()
    print(f"\nfig2_bernoulli_heteromers finished in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
