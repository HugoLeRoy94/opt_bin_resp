#!/usr/bin/env python3

# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/opt_homomers.py --n_families <N_families>

import argparse
import time
import sys
sys.path.append('/app')

from src.config import RunConfig
from src.run import SweepRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize Homomer Receptors via Parameter Sweep")
    parser.add_argument("--n_families", type=int, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--base_folder", type=str, default="/app/data/homomers_w")
    parser.add_argument("--environment_geometry", type=str,
                        choices=["asymmetric", "symmetric"], default="asymmetric")
    parser.add_argument("--entropy", type=str,
                        choices=["shannon", "renyi", "blocked", "proxy", "mi_ligand", "mi_conc"],
                        default="renyi")
    return parser.parse_args()


def main():
    args = parse_args()

    config = RunConfig(
        # --- Environment ---
        n_families=args.n_families,
        n_ligands=100,
        latent_dim=[3],
        family_spread=0.1,
        average_family_distance=5.0,
        environment_geometry=args.environment_geometry,
        distribution_type="gaussian",
        observation_noise_sigma=0.,
        affinity_kernel="gaussian",
        kernel_params=[1.0],

        # --- Concentration ---
        conc_model_type="lognormal",
        conc_mean_range=(-7.0, -5.0),
        conc_std_range=(0.5, 1.5),
        p_presence_range=(0.05, 0.5),

        # --- Physics ---
        n_genes=[1, 2, 3, 5],  # warm-started trajectory axis
        k_sub=5,
        temperature=0.1,

        # --- Mixture ---
        batch_size=2**12,

        # --- Loss ---
        entropy=args.entropy,
        cov_weight=1.0,
        penalty_type="repulsion",
        n_c_bins=10,

        # --- Training ---
        epochs=500,
        lr=0.05,
        use_scheduler=False,
        test_batch_size=2**12,
        measurement_fns=["full_array_entropy"],

        # --- Sweep control ---
        n_samples=args.samples,
        sweep_name="homomer_sweep",
        base_folder=args.base_folder,
        warm_start_axis="n_genes",
        seed=0,
    )

    print(config)
    start_time = time.time()
    SweepRunner(config).execute()
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nSweep complete! Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")


if __name__ == "__main__":
    main()
