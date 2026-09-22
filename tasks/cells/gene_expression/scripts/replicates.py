#!/usr/bin/env python3
"""Independent repetitions of the original three-gene comparison.

Run each condition separately; use --dry_run to inspect the experiment first.
"""
import sys
from pathlib import Path

sys.path.append('/app')
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.config import RunConfig
from tasks.cells.gene_expression._experiments import (
    BASE_ENVIRONMENT, MEASUREMENTS, argument_parser, cell_axes, launch, make_rows,
)


def main(argv=None):
    parser = argument_parser(__doc__, default_coverage="random")
    parser.add_argument("--n_genes", type=int, default=3)
    parser.add_argument("--n_cells", type=int, default=10)
    args = parser.parse_args(argv)
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

        # --- Physics: the same mean readout as the receptor-equivalent baseline ---
        n_genes=[r["n_genes"] for r in rows], k_sub=5,
        temperature=0.05, initial_temperature=3.0,
        affinity_kernel="gaussian", kernel_params=(1.0,), use_interface_model=True,

        # --- Cells: complete repertoires or explicit uniform homomer pools ---
        **cell_axes(rows, args.condition),
        n_cells=[r["n_cells"] for r in rows],
        cell_sampling_seed=[r["cell_sampling_seed"] for r in rows],
        cell_max_genes=[r["genes_per_cell"] for r in rows],
        cell_stoichiometry="multinomial", cell_readout="mean",
        cell_pool_chunk=128, recompute_backward=True,

        # --- Loss and independent training runs ---
        entropy=args.entropy, cell_grouped_max_states=args.max_states,
        epochs=args.epochs, lr=1e-2, use_scheduler=False,
        batch_size=args.batch_size, test_batch_size=args.test_batch_size,
        final_test_batch_size=args.final_batch_size, eval_chunk_size=args.eval_chunk_size,
        per_epoch_measure=False, measurement_fns=MEASUREMENTS,
        final_measurement_fns=MEASUREMENTS,

        # --- Sweep: one runner, fresh environment and optimizer at every point ---
        sweep_name=f"replicates_{args.condition}_{args.coverage}",
        base_folder=args.base_folder, warm_start=False,
    )
    return launch(config, args, rows, "replicates")


if __name__ == "__main__":
    main()
