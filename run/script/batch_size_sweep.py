#!/usr/bin/env python3

# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/batch_size_sweep.py
# add -d for silent running
# To run on GPU 2
# MY_GPU=2 docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/batch_size_sweep.py

import argparse
import time
import sys

sys.path.append('/app')

# --- Local Imports ---
from src.config import SweepConfig
from src.run import SweepRunner

def parse_args():
    parser = argparse.ArgumentParser(description="Investigate batch size influence on optimization")
    parser.add_argument("--samples", type=int, default=5, help="Number of independent runs per configuration")
    parser.add_argument("--base_folder", type=str, default="/app/data/batch_size_sweep", help="Output directory root")
    parser.add_argument("--env_type", type=str, choices=["asymmetric", "symmetric"], default="asymmetric", help="Type of environment geometry")
    parser.add_argument("--loss_type", type=str, choices=["exact", "proxy", "family", "conc"], default="exact", help="Information loss objective")
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()

    # === Sweep Parameters ===
    # Average complexity configuration
    n_families = 5
    n_units = 10
    latent_dim = 3
    
    # Batch sizes to sweep: 2^5 to 2^15
    batch_size_list = [2**i for i in range(5, 16)]  # [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    
    # Fixed parameters
    fixed_params = {
        "epochs": 500,
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

    total_configs = len(batch_size_list) * args.samples
    print(f"\n🔬 Starting batch size investigation")
    print(f"   Configurations to test: {total_configs}")
    print(f"   n_families: {n_families}")
    print(f"   n_units: {n_units}")
    print(f"   latent_dim: {latent_dim}")
    print(f"   batch_size: {batch_size_list}")
    print()

    # === Sweep over batch sizes ===
    for batch_size in batch_size_list:
        # Create sweep-specific name
        sweep_name = f"batch_size_{batch_size}"
        
        # Build base_run_params with current batch size
        base_run_params = {
            **fixed_params,
            "batch_size": batch_size
        }

        # Create sweep config (latent_dim_list and n_units_list need to be lists for SweepConfig)
        sweep_config = SweepConfig(
            n_families=n_families,
            latent_dim_list=[latent_dim],
            n_units_list=[n_units],
            n_samples=args.samples,
            base_folder=args.base_folder,
            sweep_name=sweep_name,
            base_run_params=base_run_params
        )

        print(f"→ Running sweep: {sweep_config.sweep_name}")
        print(f"   batch_size={batch_size}")

        # Execute sweep
        runner = SweepRunner(sweep_config)
        runner.execute()

        print()

    # Wrap up
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n✅ Batch size investigation complete!")
    print(f"   Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"\n📊 Analysis suggestion:")
    print(f"   Load results with SweepLoader and plot:")
    print(f"   - Entropy vs batch_size (log scale)")
    print(f"   - Training stability (variance across samples) vs batch_size")
    print(f"   - Convergence speed vs batch_size")
    print(f"   - Look for batch sizes that are too small (high noise) or too large (memory issues)")

if __name__ == "__main__":
    main()
