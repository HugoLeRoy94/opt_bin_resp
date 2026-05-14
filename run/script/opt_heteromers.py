# docker compose -f /home/leroy/opt_bin_resp/docker-compose.server.yaml run --rm gpu-runner python3 /app/run/script/opt_heteromers.py Nfamilies

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
latent_dim_list  = [3, 7, 10]
n_genes_list     = [1, 2, 3, 5, 7, 8, 10]
n_receptors_list = [1, 2, 3, 5, 7, 8, 10, 12, 15, 20, 30]
n_samples = 10
N_train   = 2**12
rng = np.random.default_rng(0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 opt_heteromers.py <n_families>")
        sys.exit(1)

    n_families = int(sys.argv[1])

    total_iterations = sum(
        1
        for _ in latent_dim_list
        for _ in range(n_samples)
        for n_genes in n_genes_list
        for n_receptors in n_receptors_list
        if n_receptors <= math.comb(n_genes + 5 - 1, 5)
    )

    start_time = time.time()

    with tqdm(total=total_iterations, desc="Overall Progress", dynamic_ncols=True) as pbar:
        for latent_dim in latent_dim_list:
            for sample in range(n_samples):
                prev_env = None

                # Draw concentration params once per (latent_dim, sample) trajectory
                conc_mean  = rng.uniform(-7.0, -5.0, size=n_families).tolist()
                conc_std   = rng.uniform(0.5, 1.5,   size=n_families).tolist()
                p_presence = rng.uniform(0.05, 0.5,  size=n_families).tolist()

                for n_genes in n_genes_list:
                    for n_receptors in n_receptors_list:
                        if n_receptors > math.comb(n_genes + 5 - 1, 5):
                            break

                        run_dir = os.path.join(
                            base_folder,
                            f"families_{n_families}",
                            f"dim_{latent_dim}",
                            f"n_genes_{n_genes}",
                            f"n_receptors_{n_receptors}",
                            f"sample_{sample}",
                        )

                        gene_probs       = [min(1.0, 2.69 / n_genes)] * n_genes
                        receptor_indices = generate_bernoulli_receptors(
                            N_receptors=n_receptors, n_genes=n_genes, k_sub=5, gene_probs=gene_probs
                        )

                        config = SingleRunConfig(
                            n_families=n_families,
                            n_ligands=n_families,
                            latent_dim=latent_dim,
                            n_genes=n_genes,
                            k_sub=5,
                            temperature=0.1,
                            affinity_kernel="gaussian",
                            kernel_params=[1.0],
                            observation_noise_sigma=0.,
                            family_spread=0.1,
                            average_family_distance=5.0,
                            environment_geometry="asymmetric",
                            distribution_type="gaussian",
                            conc_model_type="lognormal",
                            conc_mean=conc_mean,
                            conc_std=conc_std,
                            p_presence=p_presence,
                            batch_size=N_train,
                            epochs=500,
                            lr=0.05,
                            entropy="renyi",
                            cov_weight=1.0,
                            penalty_type="repulsion",
                            n_c_bins=10,
                            use_scheduler=False,
                            test_batch_size=2**12,
                            measurement_fns=["full_array_entropy"],
                            receptor_indices=receptor_indices.tolist(),
                        )

                        logger = ExperimentLogger(run_dir)
                        logger.save_config(config)

                        runner   = SimulationRunner(config, logger)
                        prev_env = runner.run(prev_env=prev_env)

                        pbar.update(1)

    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nOptimization complete! Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
