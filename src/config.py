from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Generator, Tuple, Optional
import itertools
import numpy as np

@dataclass
class SingleRunConfig:
    """The absolute fingerprint of a single simulation run."""
    
    # Physics
    n_units: int = 1
    k_sub: int = 5
    temperature: float = 0.1
    use_sensitivity: bool = False

    # environment
    #  Energy
    n_families: int = 1
    latent_dim: int = 3
    shape_sigma: float = 0.1
    average_family_distance: float = 5.0
    env_type: str = "asymmetric"
    #  Concentration
    conc_model_type: str = "lognormal"
    conc_mean: Optional[List[float]] = None
    conc_std: Optional[List[float]] = None
    
    # Training
    batch_size: int = 4096
    epochs: int = 500
    lr: float = 0.05
    loss_type: str = "exact"
    entropy: str = "renyi"
    cov_weight: float = 1.0
    
    # Array Architecture
    receptor_indices: Optional[List[List[int]]] = None
    
    # Measurement functions to compute during training/evaluation
    measurement_fns: List[str] = field(default_factory=lambda: [
        "full_array_entropy",
        "mean_receptor_distance",
        "conditional_entropy_family",
        "mutual_information_family",
        "conditional_entropy_concentration",
        "mutual_information_concentration",
        "receptor_distances",
        "rank_ordered_distances",
        "mean_specialization_index",
        "receptor_conditioned_entropy"
    ])

    def __post_init__(self):
        """Initialize conc_mean and conc_std with random values per family if not provided."""
        if self.conc_mean is None:
            self.conc_mean = [float(np.random.uniform(-7.0, -5.0)) for _ in range(self.n_families)]
        if self.conc_std is None:
            self.conc_std = [float(np.random.uniform(0.5, 1.5)) for _ in range(self.n_families)]

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
            # Generate concentration mean and std for this specific trajectory
            # Each family gets a different random value
            # The values are in log10 space for lognormal, or linear space for normal
            trajectory_conc_mean = [float(np.random.uniform(-7.0, -5.0)) for _ in range(self.n_families)]
            trajectory_conc_std = [float(np.random.uniform(0.5, 1.5)) for _ in range(self.n_families)]
            
            trajectory_configs = []
            for n_units in sorted_units:
                params = {
                    "n_families": self.n_families,
                    "latent_dim": latent_dim,
                    "n_units": n_units,
                    "conc_mean": trajectory_conc_mean,
                    "conc_std": trajectory_conc_std,
                    **self.base_run_params
                }
                # Only include conc_model_type if it's in base_run_params, otherwise use default
                if "conc_model_type" not in self.base_run_params:
                    params["conc_model_type"] = "lognormal"
                trajectory_configs.append(SingleRunConfig(**params))
                
            meta = {"latent_dim": latent_dim, "sample_id": sample_id}
            yield meta, trajectory_configs