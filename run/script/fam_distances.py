#!/usr/bin/env python3

# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/fam_distances.py
# add -d for silent running
# To run on GPU 2
# MY_GPU=2 docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/fam_distances.py

import argparse
import time
import sys

sys.path.append('/app')

# --- Local Imports ---
from src.config import SweepConfig
from src.run import SweepRunner

def parse_args():
    parser = argparse.ArgumentParser(description="Investigate family distance influence on optimization")
    parser.add_argument("--samples", type=int, default=5, help="Number of independent runs per configuration")
    parser.add_argument("--base_folder", type=str, default="/app/data/fam_distance_sweep", help="Output directory root")
    parser.add_argument("--env_type", type=str, choices=["asymmetric", "symmetric"], default="asymmetric", help="Type of environment geometry")
    parser.add_argument("--loss_type", type=str, choices=["exact", "proxy", "family", "conc"], default="exact", help="Information loss objective")
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()

    # === Sweep Parameters ===
    n_families_list = [2, 3, 5, 10]
    n_units_list = [5, 10, 20]  # Must be sorted ascending for prev_env passing
    latent_dim_list = [3]  # Fixed at 3
    
    # Shape sigma values to test (controls spread of energy distributions)
    shape_sigma_list = [0.01, 0.1, 0.5, 1.0]
    
    # Average family distance values (controls separation between family centers)
    # These are the distances that may cause optimization to fail when too large
    family_distance_list = [1.0, 2.0, 5.0, 10.0, 20.0]

    # Fixed parameters
    fixed_params = {
        "epochs": 500,
        "batch_size": 2**12,
        "k_sub": 5,
        "temperature": 0.1,
        "lr": 0.05,
        "use_sensitivity": False,
        "loss_type": args.loss_type,
        "env_type": args.env_type,
        "entropy": "renyi"
    }

    total_configs = (len(n_families_list) * len(n_units_list) * 
                     len(shape_sigma_list) * len(family_distance_list))
    print(f"\n🔬 Starting family distance investigation")
    print(f"   Configurations to test: {total_configs}")
    print(f"   n_families: {n_families_list}")
    print(f"   n_units: {n_units_list}")
    print(f"   shape_sigma: {shape_sigma_list}")
    print(f"   avg_family_distance: {family_distance_list}")
    print()

    # === Nested Sweep Loop ===
    for n_families in n_families_list:
        for shape_sigma in shape_sigma_list:
            for avg_family_distance in family_distance_list:
                # Create sweep-specific name for organization
                sweep_name = (f"fam_dist_nf{n_families}_ss{shape_sigma}_"
                             f"dist{avg_family_distance}")
                
                # Build base_run_params with current sweep parameters
                base_run_params = {
                    **fixed_params,
                    "shape_sigma": shape_sigma,
                    "average_family_distance": avg_family_distance
                }

                # Create sweep config
                sweep_config = SweepConfig(
                    n_families=n_families,
                    latent_dim_list=latent_dim_list,
                    n_units_list=n_units_list,
                    n_samples=args.samples,
                    base_folder=args.base_folder,
                    sweep_name=sweep_name,
                    base_run_params=base_run_params
                )

                print(f"→ Running sweep: {sweep_config.sweep_name}")
                print(f"   n_families={n_families}, shape_sigma={shape_sigma}, "
                      f"avg_family_distance={avg_family_distance}")

                # Execute sweep
                runner = SweepRunner(sweep_config)
                runner.execute()

                print()

    # Wrap up
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n✅ Family distance investigation complete!")
    print(f"   Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"\n📊 Analysis suggestion:")
    print(f"   Load results with SweepLoader and plot:")
    print(f"   - Entropy vs average_family_distance (should be flat if optimization succeeds)")
    print(f"   - Entropy vs n_families")
    print(f"   - Entropy vs shape_sigma")
    print(f"   - Drops in entropy indicate optimization failure")

if __name__ == "__main__":
    main()
