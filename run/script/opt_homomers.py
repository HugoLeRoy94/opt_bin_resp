#!/usr/bin/env python3

# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/opt_homomers.py <N_families>
# add -d for silent running
# To run on GPU 2
# MY_GPU=2 docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/opt_homomers.py <N_families>

import argparse
import time
import sys

sys.path.append('/app')

# --- Local Imports ---
from src.config import SweepConfig
from src.run import SweepRunner

def parse_args():
    parser = argparse.ArgumentParser(description="Optimize Homomer Receptors via Parameter Sweep")
    parser.add_argument("--n_families", type=int, help="Number of ligand families to simulate")
    parser.add_argument("--samples", type=int, default=5, help="Number of independent runs per configuration")
    parser.add_argument("--base_folder", type=str, default="/app/data/homomers_w", help="Output directory root")
    parser.add_argument("--env_type", type=str, choices=["asymmetric", "symmetric"], default="asymmetric", help="Type of environment geometry")
    parser.add_argument("--loss_type", type=str, choices=["exact", "proxy", "family", "conc"], default="exact", help="Information loss objective")
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()

    # 1. Define the limits of your grid
    latent_dim_list = [3]#[3, 7, 10]
    
    # NOTE: n_units_list must be sorted ascending so the SweepRunner can 
    # properly pass the 'prev_env' state forward through the trajectory.
    n_units_list = [1, 2, 3, 5]#, 7, 8, 10, 12, 15, 20, 30] 

    # 2. Package everything into the SweepConfig
    sweep_config = SweepConfig(
        n_families=args.n_families,
        latent_dim_list=latent_dim_list,
        n_units_list=n_units_list,
        n_samples=args.samples,
        base_folder=args.base_folder,
        sweep_name="homomer_sweep",
        
        # Base parameters applied to every single run in the grid
        base_run_params={
            "epochs": 500,
            "batch_size": 2**12,
            "k_sub": 5,
            "temperature": 0.1,
            "lr": 0.05,
            "shape_sigma": 0.1,
            "average_family_distance": 5.0,
            "use_sensitivity": False,
            "loss_type": args.loss_type,
            "env_type": args.env_type,
            "entropy": "renyi"
        }
    )

    # 3. Let the SweepRunner handle the nested loops and logging orchestration
    runner = SweepRunner(sweep_config)
    runner.execute()

    # Wrap up
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n✅ Sweep complete! Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

if __name__ == "__main__":
    main()