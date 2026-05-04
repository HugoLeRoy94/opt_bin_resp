#!/usr/bin/env python3

# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/latent_dim_sweep.py
# add -d for silent running
# To run on GPU 2
# MY_GPU=2 docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/latent_dim_sweep.py

import argparse
import time
import sys

sys.path.append('/app')

# --- Local Imports ---
from src.config import SweepConfig
from src.run import SweepRunner

def parse_args():
    parser = argparse.ArgumentParser(description="Investigate latent dimension influence on optimization")
    parser.add_argument("--samples", type=int, default=5, help="Number of independent runs per configuration")
    parser.add_argument("--base_folder", type=str, default="/app/data/latent_dim_sweep", help="Output directory root")
    parser.add_argument("--env_type", type=str, choices=["asymmetric", "symmetric"], default="asymmetric", help="Type of environment geometry")
    parser.add_argument("--loss_type", type=str, choices=["exact", "proxy", "family", "conc"], default="exact", help="Information loss objective")
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()

    # === Sweep Parameters ===
    n_families = 1  # Single family
    n_units_list = [1, 2, 3, 5, 7, 10, 20, 30]  # Must be sorted ascending for prev_env passing
    latent_dim_list = [1, 2, 3, 5, 7, 10, 15, 20]
    
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
        "entropy": "renyi",
        "shape_sigma": 0.1,
        "average_family_distance": 5.0,
        "measurement_fns": ["full_array_entropy"]
    }

    total_configs = len(latent_dim_list) * len(n_units_list) * args.samples
    print(f"\n🔬 Starting latent dimension investigation")
    print(f"   Configurations to test: {total_configs}")
    print(f"   n_families: {n_families}")
    print(f"   n_units: {n_units_list}")
    print(f"   latent_dim: {latent_dim_list}")
    print()

    # Create sweep config
    sweep_name = "latent_dim_sweep"
    
    sweep_config = SweepConfig(
        n_families=n_families,
        latent_dim_list=latent_dim_list,
        n_units_list=n_units_list,
        n_samples=args.samples,
        base_folder=args.base_folder,
        sweep_name=sweep_name,
        base_run_params=fixed_params
    )

    print(f"→ Running sweep: {sweep_config.sweep_name}")

    # Execute sweep
    runner = SweepRunner(sweep_config)
    runner.execute()

    # Wrap up
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n✅ Latent dimension investigation complete!")
    print(f"   Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"\n📊 Analysis suggestion:")
    print(f"   Load results with SweepLoader and plot:")
    print(f"   - Entropy vs latent_dim")
    print(f"   - Entropy vs n_units")
    print(f"   - 2D heatmap of entropy with latent_dim on x-axis and n_units on y-axis")
    print(f"   - Drops in entropy indicate optimization difficulty in high dimensions")

if __name__ == "__main__":
    main()
