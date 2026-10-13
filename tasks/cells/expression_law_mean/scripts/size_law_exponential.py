#!/usr/bin/env python3
"""Expression law: genes drawn UNIFORMLY, NUMBER of genes drawn EXPONENTIALLY.

A cell draws how many genes it expresses from p_i proportional to exp(lambda*i) over
{1..G}, with lambda solved for the swept mean, then draws which genes uniformly
without replacement. The support is the WHOLE range at any mean above 1, so matched
MEAN is not matched POOL and not matched COST: at G=5 this arm sits at the full
629-receptor pool for every mean above 1, against 53 -> 629 for the flat law.

Every parameter is a constant below. No command line, no shared design machinery.
The paired script is size_law_uniform.py; the two files are identical except for
SIZE_FAMILY, the seeds and the sweep name, so the comparison is single-variable.

The two arms do NOT share a world: WORLD_SEED and CONC_SEED differ, so latents and
concentration means differ between them. A gap between the curves has to survive a
change of environment to mean anything.

    python3 tasks/cells/expression_law_mean/scripts/size_law_exponential.py
    bash tasks/run_remote.sh cells/expression_law_mean size_law_exponential.py 0
"""
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.append('/app')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.cells import expand_gene_set, sample_gene_sets_by_size, size_pmf_for_mean
from src.config import RunConfig
from src.run import SweepRunner
from _lib.repertoires import heterogeneous_conc_means

# ─── Environment ────────────────────────────────────────────────────────────
# Identical to tasks/cells/gene_expression_mean/scripts/heteromers.py. Keep them in
# step or the two tasks are measuring different problems.
N_LIGANDS     = 100
LATENT_DIM    = 6
FAMILY_SPREAD = 0.1
FAMILY_DIST   = 1.0
GEOMETRY      = "asymmetric"
DISTRIBUTION  = "gaussian"
NOISE_SIGMA   = 0.0
MU_LIGANDS    = 1e-6    # ligands per sniff (ZTP rate). > 1 is a MIXTURE, a different problem.
CONC_SPAN     = 0.0     # per-ligand log10 concentration mean in +/- this. 0 = flat channel.
CONC_STD      = 1.0
CONC_SEED     = 32

# ─── The expression law ─────────────────────────────────────────────────────
# SIZE law: how many genes a cell expresses, over {1..G}.
#   "uniform"     flat over {1..M}, so the mean is (M+1)/2 and only 1, 1.5, 2, ...
#                 are reachable. Max genes per cell is M.
#   "exponential" p_i proportional to exp(lambda*i), lambda solved for the mean.
#                 Support is the WHOLE range at any mean above 1, so a few cells
#                 express every gene and the receptor pool saturates immediately.
# GENE law is uniform in both arms: which genes, sampled without replacement, equally
# weighted. Only the SIZE law differs.
SIZE_FAMILY = "exponential"

N_GENES_LIST = (3, 5)
MEANS        = (1.0, 1.5, 2.0, 2.5, 3.0)
N_CELLS      = 30
K_SUB        = 5
REPLICATES   = 3
REPLICATE_START = 0
CELL_SEED_BASE  = 0
MAX_POOL     = 8192      # refuse a point whose receptor pool exceeds this

# ─── Objective and estimator ────────────────────────────────────────────────
# Independent sampling leaves nearly every cell its own type, so the joint count
# alphabet reaches ~10^7 and exact enumeration is out of reach. Training uses the KT
# bound with grouping; evaluation uses sampled group counts. Both are estimates.
ENTROPY      = "grouped_kt_mi"
MEASUREMENTS = ("grouped_counting", "conditional_entropy_response", "codeword_entropy")

# ─── Training ───────────────────────────────────────────────────────────────
EPOCHS           = 5000
LR               = 1e-2
BATCH_SIZE       = 4096
TEST_BATCH_SIZE  = 4096
FINAL_BATCH_SIZE = 16_777_216   # sampled counting is biased DOWN; only this reduces it
EVAL_CHUNK_SIZE  = 512
TEMPERATURE      = 0.05
INIT_TEMPERATURE = 3.0
WORLD_SEED       = 20260106

BASE_FOLDER = "/app/data/expression_law"
SWEEP_NAME  = "expression_law_size_exponential"

# Measured on the G=5, C=30 arrays, used only by the printed bias projection.
CELL_FIRING_P = 0.0905
TYPICAL_MI    = 6.0


def design():
    """One row per planned run, with its gene sets already drawn.

    Sampling happens HERE so the heteromer and homomer arms can receive identical
    arrays, and so the design is visible before anything trains.

    MATCHED GRID: a point is skipped when a FLAT law over {1..M} could not reach
    that mean, i.e. when 2*mean - 1 > G. The exponential law could reach it, but then
    the two arms would have different support and the curves would not be comparable.
    This is the only reason the skip is here and not keyed on SIZE_FAMILY.
    """
    rows = []
    for replicate in range(REPLICATE_START, REPLICATE_START + REPLICATES):
        for n_genes in N_GENES_LIST:
            for mean in MEANS:
                if 2 * mean - 1 > n_genes:
                    continue
                seed = (CELL_SEED_BASE * 1_000_000 + replicate * 1000
                        + n_genes * 10 + int(mean * 2))
                pmf = size_pmf_for_mean(n_genes, mean, SIZE_FAMILY)
                sets = sample_gene_sets_by_size(N_CELLS, n_genes, pmf, seed=seed)
                pool = {r for gs in sets for r in expand_gene_set(gs, K_SUB, True)}
                if len(pool) > MAX_POOL:
                    raise ValueError(f"G={n_genes}, mean={mean}: receptor pool "
                                     f"{len(pool)} exceeds MAX_POOL={MAX_POOL}.")
                rows.append(dict(
                    replicate=replicate, seed=seed, n_genes=n_genes,
                    target_mean=mean,
                    realised_mean=sum(map(len, sets)) / len(sets),
                    gene_sets=sets, pool=len(pool), distinct=len(set(sets)),
                    genes_seen=len({g for gs in sets for g in gs}),
                ))
    if not rows:
        raise ValueError("No design points: every requested mean exceeds its gene pool.")
    return rows


def report(rows):
    """Design table, then the projected counting bias at FINAL_BATCH_SIZE.

    bias [bits] ~= 3.34 * (B / 2^H(K)) ** -0.92, fitted against exact enumeration on
    12 points of the G=5, C=30 arrays. What matters is the ratio of budget to
    EFFECTIVE ALPHABET 2^H(K), not the budget alone. H(K) = MI + H(K|X), and H(K|X)
    is a sum of binomial entropies over the groups actually drawn. The per-cell
    firing rate and MI come from measured arrays, so this is an extrapolation for
    sizing the budget, not a guarantee.
    """
    print(f"{'G':>3} {'target':>7} {'realised':>9} {'pool':>6} {'types':>6} "
          f"{'seen':>6} {'rep':>4} {'bias':>7}")
    worst = 0.0
    for row in rows:
        sizes = Counter(row["gene_sets"]).values()
        h_kx = sum(-sum(q * math.log2(q) for q in
                        (math.comb(n, k) * CELL_FIRING_P ** k
                         * (1 - CELL_FIRING_P) ** (n - k) for k in range(n + 1)) if q > 0)
                   for n in sizes)
        bias = 3.34 * (FINAL_BATCH_SIZE / 2 ** (TYPICAL_MI + h_kx)) ** -0.92
        worst = max(worst, bias)
        print(f"{row['n_genes']:>3} {row['target_mean']:>7.2f} {row['realised_mean']:>9.2f} "
              f"{row['pool']:>6} {row['distinct']:>6} "
              f"{row['genes_seen']:>3}/{row['n_genes']:<2} {row['replicate']:>4} "
              f"{bias:>6.2f}b")
    print(f"\n{len(rows)} independent optimizations; largest pool "
          f"{max(r['pool'] for r in rows)}; worst projected bias {worst:.2f} bits")
    if worst > 0.2:
        print("Above ~0.2 bits the curve SHAPE is distorted, because the bias differs\n"
              "between design points. Raise FINAL_BATCH_SIZE or lower N_CELLS: H(K|X) is\n"
              "a sum over cells, so 2^H(K) grows close to exponentially in C while an\n"
              "affordable budget grows linearly.")
    print(f"Training ceiling: the KT objective cannot exceed log2(BATCH_SIZE) = "
          f"{math.log2(BATCH_SIZE):.0f} bits.")
    print(f"{N_LIGANDS} ligands, mu={MU_LIGANDS:g} per sniff, "
          f"conc means over {2 * CONC_SPAN:g} decades, size law {SIZE_FAMILY}")


def main():
    rows = design()
    report(rows)
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
            len(rows), N_LIGANDS, CONC_SPAN, CONC_SEED),
        conc_std        = (CONC_STD,) * N_LIGANDS,

        # --- Physics ---
        n_genes             = [r["n_genes"] for r in rows],
        k_sub               = K_SUB,
        temperature         = TEMPERATURE,
        initial_temperature = INIT_TEMPERATURE,
        affinity_kernel     = "gaussian",
        kernel_params       = (1.0,),
        use_interface_model = True,

        # --- Cells ---
        cell_gene_sets     = [r["gene_sets"] for r in rows],
        cell_sampling_seed = [r["seed"] for r in rows],
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
    # Provenance, written next to the sweep before training. The analysis joins
    # each run back to its planned design through cell_sampling_seed.
    manifest = dict(
        experiment="expression_law", condition="heteromers",
        size_family=SIZE_FAMILY, gene_family="uniform", gene_ratio=1.0,
        arguments=dict(entropy=ENTROPY, n_cells=N_CELLS, replicates=REPLICATES,
                       mu_ligands=MU_LIGANDS, conc_span=CONC_SPAN,
                       world_seed=WORLD_SEED, final_batch_size=FINAL_BATCH_SIZE),
        rows=[dict(r, gene_sets=[list(g) for g in r["gene_sets"]],
                   cell_sampling_seed=r["seed"]) for r in rows],
    )
    t0 = time.time()
    runner = SweepRunner(config)
    (Path(runner.master_logger.sweep_root) / "experiment.json").write_text(
        json.dumps(manifest, indent=2))
    runner.execute()
    print(f"\nexpression_law_size_exponential finished in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
