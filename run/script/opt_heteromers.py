# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/opt_heteromers.py Nfamilies
# add -d for silent running

# To run on GPU 2
# MY_GPU=2 docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/opt_heteromers.py Nfamilies

import sys
sys.path.append('/app')
import numpy as np
import os
import time
from tqdm import tqdm
import math

from src.geometry import generate_bernoulli_receptors
from src.config import SingleRunConfig
from src.run import SimulationRunner
from src.IO import ExperimentLogger

base_folder = "/app/data/heteromers"
latent_dim_list = [3,7, 10]
n_units_list = [1,2,3,5,7,8,10]
n_receptors_list = [1, 2, 3, 5, 7, 8, 10, 12, 15, 20, 30]
n_samples = 10 
N_train = 2**12

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Missing n_families argument.")
        print("Usage: python3 opt_heteromers.py <n_families>")
        sys.exit(1)
        
    n_families = int(sys.argv[1])

    total_iterations = 0
    for _ in latent_dim_list:
        for _ in range(n_samples):
            for n_units in n_units_list:
                for n_receptors in n_receptors_list:
                    max_combinations = math.comb(n_units + 5 - 1, 5)
                    if n_receptors > max_combinations:
                        break
                    total_iterations += 1
                    
    start_time = time.time()

    with tqdm(total=total_iterations, desc="Overall Progress", dynamic_ncols=True) as pbar:
        for latent_dim in latent_dim_list:
            for sample in range(n_samples):
                prev_env = None
                init_means = [float(np.random.randint(1, 8)) for _ in range(n_families)]
                
                for n_units in n_units_list:
                    for n_receptors in n_receptors_list:
                        max_combinations = math.comb(n_units + 5 - 1, 5)
                        if n_receptors > max_combinations:
                            break
                            
                        base_dir = base_folder+f"/families_{n_families}/dim_{latent_dim}/n_units_{n_units}/n_receptors_{n_receptors}"

                        gene_probs = [min(1.0, 2.69 / n_units)] * n_units
                        receptor_indices = generate_bernoulli_receptors(N_receptors=n_receptors, n_units=n_units, k_sub=5, gene_probs=gene_probs)
                        
                        config = SingleRunConfig(
                            n_families=n_families,
                            latent_dim=latent_dim,
                            n_units=n_units,
                            init_means=init_means,
                            k_sub=5,
                            batch_size=N_train,
                            epochs=500,
                            lr=0.05,
                            loss_type="exact",
                            entropy="renyi"
                        )
                        
                        logger = ExperimentLogger(os.path.join(base_dir, f"sample_{sample}"))
                        logger.save_config(config)

                        runner = SimulationRunner(config, logger)
                        prev_env = runner.run(prev_env=prev_env, receptor_indices=receptor_indices)
                            
                        pbar.update(1)

    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nOptimization complete! Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")