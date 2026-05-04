from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Generator, Tuple, Optional
import itertools
import numpy as np

@dataclass
class SingleRunConfig:
    """The absolute fingerprint of a single simulation run."""
    
    # Physics
    n_units: int # number of genes
    k_sub: int = 5
    temperature: float = 0.1
    use_sensitivity: bool = False

    # environment
    #  Energy
    n_families: int
    latent_dim: int
    shape_sigma: float = 0.1
    average_family_distance: float = 5.0
    env_type: str = "asymmetric"
    #  Concentration
    init_means: List[float]
    init_std: None # not implemented yet
    
    # Training
    batch_size: int = 4096
    epochs: int = 500
    lr: float = 0.05
    loss_type: str = "exact"
    entropy: str = "renyi"
    cov_weight: float = 1.0
    
    # Array Architecture
    receptor_indices: Optional[List[List[int]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SweepConfig:
    """Wraps lists of parameters to sweep over, handling the nested grid logic."""
    n_families: int
    latent_dim_list: List[int]
    n_units_list: List[int] # Will be sorted ascending internally for prev_env logic
    n_samples: int
    base_folder: str
    sweep_name: str = "homomer_sweep"
    base_run_params: Dict[str, Any] = field(default_factory=dict)

    def generate_trajectories(self) -> Generator[Tuple[Dict[str, int], List[SingleRunConfig]], None, None]:
        """
        Generates the simulation grid.
        Yields a tuple containing:
        1. meta: A dictionary with 'latent_dim' and 'sample_id'
        2. trajectory: A list of SingleRunConfigs sequentially ordered by n_units.
        """
        independent_runs = list(itertools.product(self.latent_dim_list, range(self.n_samples)))
        sorted_units = sorted(self.n_units_list)
        
        for latent_dim, sample_id in independent_runs:
            # Generate shared means for this specific trajectory to maintain environment consistency
            trajectory_means = [float(np.random.uniform(3.0, 5.0)) for _ in range(self.n_families)]
            
            trajectory_configs = []
            for n_units in sorted_units:
                params = {
                    "n_families": self.n_families,
                    "latent_dim": latent_dim,
                    "n_units": n_units,
                    "init_means": trajectory_means,
                    **self.base_run_params
                }
                trajectory_configs.append(SingleRunConfig(**params))
                
            meta = {"latent_dim": latent_dim, "sample_id": sample_id}
            yield meta, trajectory_configs