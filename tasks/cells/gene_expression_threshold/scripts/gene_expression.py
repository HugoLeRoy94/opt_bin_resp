#!/usr/bin/env python3
"""Genes expressed per cell, with a THRESHOLDED cell instead of a mean cell.

Single-variable change against the mean baseline. Everything else — environment,
physics, expression design, objective, estimator — is taken from
`tasks/cells/gene_expression_mean/_experiments.py`, so any difference in the
curves is the readout and nothing else.

  mean      A_bc = S_bc                           (cell fires at the weighted
                                                   fraction of open receptors)
  threshold A_bc = sigmoid((S_bc - theta)/T)      (cell fires above a current)

  S_bc    drive: abundance-weighted fraction of cell c's receptors open in sniff b
  theta   ONE scalar shared by every cell, pinned to the median drive and re-pinned
          every `cell_recalibrate_every` epochs; not fitted
  T       cell sharpness, annealed in phase 2 down to `cell_temperature` times the
          live spread of the drive

Design (defaults): G = 5 genes, C = 30 cells, complete coverage (every gene is
expressed by at least one cell), genes/cell g = 1..5, latent dimension 6, one
ligand per sniff, training on the exact grouped objective, final evaluation by
exact enumeration of joint count states.

    python3 tasks/cells/gene_expression_threshold/scripts/gene_expression.py --dry_run
    python3 tasks/cells/gene_expression_threshold/scripts/gene_expression.py --condition heteromers
    python3 tasks/cells/gene_expression_threshold/scripts/gene_expression.py --condition homomers
"""
import sys
from pathlib import Path

sys.path.append('/app')
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.config import RunConfig
from tasks.cells.gene_expression_mean._experiments import (
    BASE_ENVIRONMENT, measurements, argument_parser, cell_axes, launch, make_rows,
)


def main(argv=None):
    parser = argument_parser(__doc__, default_coverage="complete")
    parser.add_argument("--n_genes", type=int, default=5)
    parser.add_argument("--n_cells", type=int, default=30)
    # --- the readout, which is the whole point of this task ---
    parser.add_argument("--cell_readout", choices=("threshold", "mean", "noisy_or"),
                        default="threshold")
    parser.add_argument("--cell_threshold", default="auto",
                        help='"auto" pins theta to the median drive; a float fixes it '
                             "and disables recalibration.")
    parser.add_argument("--cell_temperature", type=float, default=0.01,
                        help="Final cell sharpness as a FRACTION of the live drive "
                             "spread. 0.01 leaves cells effectively deterministic.")
    parser.add_argument("--cell_threshold_learnable", action="store_true",
                        help="Fit theta instead of pinning it. Collapses to the "
                             "fair-coin degeneracy while cells are soft; ablation only.")
    parser.add_argument("--cell_recalibrate_every", type=int, default=25,
                        help="Re-pin theta to the median drive every N epochs; 0 disables.")
    parser.add_argument("--cell_n_molecules", type=float, default=1e4,
                        help="Receptor copies per cell. 1/N floors theta, which is what "
                             "keeps a rank statistic off a point mass at zero drive.")
    # G=5, C=30 complete coverage peaks at 460,800 joint count states (g=2).
    parser.set_defaults(replicates=1, max_states=1048576,
                        base_folder="/app/data/gene_expression_threshold")
    args = parser.parse_args(argv)

    threshold = args.cell_threshold
    if threshold != "auto":
        threshold = float(threshold)
    rows = make_rows(args, [args.n_genes], [BASE_ENVIRONMENT], n_cells=args.n_cells)

    config = RunConfig(
        # --- Environment: explicit lists are zipped, not crossed ---
        n_families=1,
        n_ligands=[r["n_ligands"] for r in rows],
        latent_dim=[r["latent_dim"] for r in rows],
        family_spread=[r["family_spread"] for r in rows],
        average_family_distance=1.0,
        environment_geometry="asymmetric", distribution_type="gaussian",
        observation_noise_sigma=0.0,
        n_presence_blocks=1, mu_sources=1.0,
        mu_ligands_per_source=[r["mu_ligands_per_source"] for r in rows],
        block_shared_conc_mean=False,
        conc_model_type="lognormal",
        conc_mean=[(0.0,) * r["n_ligands"] for r in rows],
        conc_std=[(1.0,) * r["n_ligands"] for r in rows],

        # --- Physics: identical to the mean baseline ---
        n_genes=[r["n_genes"] for r in rows], k_sub=5,
        temperature=0.05, initial_temperature=3.0,
        affinity_kernel="gaussian", kernel_params=(1.0,), use_interface_model=True,

        # --- Cells: complete repertoires or explicit uniform homomer pools ---
        **cell_axes(rows, args.condition),
        n_cells=[r["n_cells"] for r in rows],
        cell_sampling_seed=[r["cell_sampling_seed"] for r in rows],
        cell_max_genes=[r["genes_per_cell"] for r in rows],
        cell_stoichiometry="multinomial",
        cell_readout=args.cell_readout,
        cell_threshold=threshold,
        cell_threshold_learnable=args.cell_threshold_learnable,
        cell_temperature=args.cell_temperature,
        cell_recalibrate_every=args.cell_recalibrate_every,
        cell_n_molecules=args.cell_n_molecules,
        cell_pool_chunk=128, recompute_backward=True,

        # --- Loss and independent training runs ---
        entropy=args.entropy, cell_grouped_max_states=args.max_states,
        epochs=args.epochs, lr=1e-2, use_scheduler=False,
        batch_size=args.batch_size, test_batch_size=args.test_batch_size,
        final_test_batch_size=args.final_batch_size, eval_chunk_size=args.eval_chunk_size,
        per_epoch_measure=False, measurement_fns=measurements(args),
        final_measurement_fns=measurements(args),

        # --- Sweep: one runner, fresh environment and optimizer at every point ---
        sweep_name=f"threshold_{args.condition}_{args.coverage}",
        base_folder=args.base_folder, warm_start=False,
    )
    print(f"Readout: {args.cell_readout}; theta={args.cell_threshold} "
          f"(learnable={args.cell_threshold_learnable}); T_cell -> {args.cell_temperature} "
          f"x spread; recalibrate every {args.cell_recalibrate_every} epochs")
    return launch(config, args, rows, "threshold")


if __name__ == "__main__":
    main()
