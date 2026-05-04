import torch
import torch.optim as optim
import inspect
from tqdm import tqdm

# --- Local Imports ---
from src.config import SingleRunConfig, SweepConfig
from src.IO import ExperimentLogger, SweepLogger

from src import (LigandEnvironment, 
                 SymmetricLigandEnvironment, 
                 BinaryReceptor, 
                 LogNormalConcentration, 
                 NormalConcentration,
                 full_array_entropy, 
                 mean_receptor_distance,
                 conditional_entropy_family, 
                 mutual_information_family,
                 conditional_entropy_concentration, 
                 mutual_information_concentration,
                 receptor_distances, 
                 rank_ordered_distances,
                 mean_specialization_index, 
                 receptor_conditioned_entropy)

from src.bin_loss import DiscreteProxyLoss, DiscreteExactLoss
from src.family_mi_loss import MaximizeMutualInformationFamilyLoss
from src.concentration_mi_loss import MaximizeMutualInformationConcentrationLoss

# ==========================================
# 1. GLOBALS & REGISTRIES
# ==========================================

MEASUREMENT_FNS = [
    full_array_entropy, mean_receptor_distance,
    conditional_entropy_family, mutual_information_family,
    conditional_entropy_concentration, mutual_information_concentration,
    receptor_distances, rank_ordered_distances,
    mean_specialization_index, receptor_conditioned_entropy
]

# The Factory Pattern: Kills the if/else logic
ENV_REGISTRY = {
    "asymmetric": LigandEnvironment,
    "symmetric": SymmetricLigandEnvironment
}

LOSS_REGISTRY = {
    "exact": lambda cfg: DiscreteExactLoss(entropy_type=cfg.entropy),
    "proxy": lambda cfg: DiscreteProxyLoss(cov_weight=cfg.cov_weight),
    "family": lambda cfg: MaximizeMutualInformationFamilyLoss(entropy_type=cfg.entropy),
    "conc": lambda cfg: MaximizeMutualInformationConcentrationLoss(entropy_type=cfg.entropy)
}

# Concentration Model Registry
CONC_REGISTRY = {
    "lognormal": lambda cfg: LogNormalConcentration(
        n_families=cfg.n_families,
        init_mean=cfg.conc_mean,
        init_scale=cfg.conc_std
    ),
    "normal": lambda cfg: NormalConcentration(
        n_families=cfg.n_families,
        init_mean=cfg.conc_mean,
        init_scale=cfg.conc_std
    )
}

# ==========================================
# 2. SINGLE RUN MANAGER
# ==========================================

class SimulationRunner:
    """Handles initialization, training, testing, and logging for ONE specific configuration."""
    def __init__(self, config: SingleRunConfig, logger: ExperimentLogger):
        self.config = config
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _initialize(self, prev_env=None):
        """Builds components using base classes and registries."""
        if prev_env is not None:
            extra_units = max(0, self.config.n_units - prev_env.n_units)
            env = prev_env.clone_with_extra_units(extra_units).to(self.device)
        else:
            # Use concentration model registry
            conc_model_type = getattr(self.config, 'conc_model_type', 'lognormal')
            conc_strategy = CONC_REGISTRY[conc_model_type](self.config)
            
            env_class = ENV_REGISTRY[self.config.env_type]
            env = env_class(
                self.config.n_units, 
                self.config.n_families, 
                conc_model=conc_strategy,
                latent_dim=self.config.latent_dim, 
                shape_sigma=self.config.shape_sigma,
                avg_family_distance=self.config.average_family_distance,
                use_sensitivity=self.config.use_sensitivity
            ).to(self.device)

        physics = BinaryReceptor(self.config.n_units, self.config.k_sub, temperature=self.config.temperature).to(self.device)
        loss_fn = LOSS_REGISTRY[self.config.loss_type](self.config).to(self.device)
        
        # Dampen LR if picking up from a previous environment to preserve learned representations
        lr = self.config.lr if prev_env is None else self.config.lr * 0.1
        optimizer = optim.Adam(list(env.parameters()) + list(physics.parameters()), lr=lr)
        
        return env, physics, loss_fn, optimizer

    def _train(self, env, physics, loss_fn, optimizer, receptor_indices):
        start_temp = 1.0
        end_temp = self.config.temperature
        
        # Fallback to False if not present in older configs
        use_scheduler = getattr(self.config, 'use_scheduler', False) 
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs, eta_min=1e-5) if use_scheduler else None

        stats = []
        for epoch in range(self.config.epochs):
            optimizer.zero_grad()
            
            # Temperature Annealing
            current_temp = end_temp + (start_temp - end_temp) * (1.0 - (epoch / self.config.epochs)) if end_temp < start_temp else end_temp
            if hasattr(physics, 'temperature'): physics.temperature = current_temp

            energies, concs, family_ids = env.sample_batch(self.config.batch_size)
            activity = physics(energies, concs, receptor_indices)
            
            # Handle specialized loss requirements
            if isinstance(loss_fn, MaximizeMutualInformationFamilyLoss): 
                loss = loss_fn(activity, family_ids=family_ids)
            elif isinstance(loss_fn, MaximizeMutualInformationConcentrationLoss): 
                loss = loss_fn(activity, concs=concs)
            else: 
                loss = loss_fn(activity)
            
            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()
                
            # Evaluation Step
            if epoch % max(1, self.config.epochs // 100) == 0:
                with torch.no_grad():
                    if hasattr(physics, 'temperature'): physics.temperature = end_temp
                    
                    eval_batch = getattr(self.config, 'eval_batch_size', 2**12)
                    E_open_stats, concs_stats, family_ids_stats = env.sample_batch(batch_size=eval_batch)
                    activity_stats = physics(E_open_stats, concs_stats, receptor_indices)
                    
                    stat = {}
                    for fn in MEASUREMENT_FNS:
                        sig = inspect.signature(fn)
                        kwargs = {}
                        if 'env' in sig.parameters: kwargs['env'] = env
                        if 'physics' in sig.parameters: kwargs['physics'] = physics
                        if 'receptor_indices' in sig.parameters: kwargs['receptor_indices'] = receptor_indices
                        if 'loss_fn' in sig.parameters: kwargs['loss_fn'] = loss_fn
                        if 'activity' in sig.parameters: kwargs['activity'] = activity_stats
                        if 'epoch' in sig.parameters: kwargs['epoch'] = epoch
                        if 'concs' in sig.parameters: kwargs['concs'] = concs_stats
                        if 'family_ids' in sig.parameters: kwargs['family_ids'] = family_ids_stats
                        
                        result = fn(**kwargs)
                        if isinstance(result, dict): stat.update(result)
                        else: stat[getattr(fn, '__name__', str(fn))] = result
                                
                    stat['lr'] = optimizer.param_groups[0]['lr']
                    if hasattr(physics, 'temperature'): physics.temperature = current_temp
                    stats.append(stat)
                    
        return {key: [s[key] for s in stats] for key in stats[0].keys()} if stats else {}

    def _test(self, env, physics, loss_fn, indices, N_samples: int, test_epochs: int = 10):
        stats = []
        with torch.no_grad():
            for i in range(test_epochs):
                E_open_stats, concs_stats, family_ids_stats = env.sample_batch(batch_size=N_samples)
                activity_stats = physics(E_open_stats, concs_stats, indices)
                
                stat = {}
                for fn in MEASUREMENT_FNS:
                    sig = inspect.signature(fn)
                    kwargs = {}
                    if 'env' in sig.parameters: kwargs['env'] = env
                    if 'physics' in sig.parameters: kwargs['physics'] = physics
                    if 'receptor_indices' in sig.parameters: kwargs['receptor_indices'] = indices
                    if 'loss_fn' in sig.parameters: kwargs['loss_fn'] = loss_fn
                    if 'activity' in sig.parameters: kwargs['activity'] = activity_stats
                    if 'epoch' in sig.parameters: kwargs['epoch'] = i
                    if 'concs' in sig.parameters: kwargs['concs'] = concs_stats
                    if 'family_ids' in sig.parameters: kwargs['family_ids'] = family_ids_stats
                    
                    result = fn(**kwargs)
                    if isinstance(result, dict): stat.update(result)
                    else: stat[getattr(fn, '__name__', str(fn))] = result
                stats.append(stat)
                
        return {key: [s[key] for s in stats] for key in stats[0].keys()} if stats else {}

    def run(self, prev_env=None, receptor_indices=None):
        """Executes the pipeline for this configuration."""
        # 1. Initialize
        env, physics, loss_fn, optimizer = self._initialize(prev_env)

        # 2. Train
        train_stats = self._train(env, physics, loss_fn, optimizer, receptor_indices)

        # 3. Save Stats & Checkpoints
        if train_stats:
            epochs_run = len(next(iter(train_stats.values())))
            for i in range(epochs_run):
                self.logger.save_stats(i, {k: train_stats[k][i] for k in train_stats.keys()})

        self.logger.save_checkpoint(self.config.epochs, env, physics, receptor_indices, is_best=True)

        # 4. Test
        test_batch_size = max(100_000, self.config.n_families * 2000)
        test_results = self._test(env, physics, loss_fn, receptor_indices, N_samples=test_batch_size)

        # We can dump the test results JSON using the logger's path 
        import json
        from src.IO import CustomJSONEncoder
        import os
        with open(os.path.join(self.logger.run_dir, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=4, cls=CustomJSONEncoder)

        # Return environment to pass state to the next step in the trajectory
        return env


# ==========================================
# 3. MULTIPLE RUN (SWEEP) MANAGER
# ==========================================

class SweepRunner:
    """Consumes a SweepConfig and executes the generated trajectories."""
    def __init__(self, sweep_config: SweepConfig):
        self.sweep = sweep_config
        self.master_logger = SweepLogger(self.sweep)

    def execute(self):
        # Calculate total runs for the progress bar
        total_steps = len(self.sweep.latent_dim_list) * self.sweep.n_samples * len(self.sweep.n_units_list)
        
        print(f"\n🚀 Initiating Sweep: {self.master_logger.sweep_root}")
        print(f"Total Trajectory Nodes to Process: {total_steps}\n")
        
        with tqdm(total=total_steps, desc="Sweep Progress", dynamic_ncols=True) as pbar:
            
            # The config's generator handles all the grid logic and shared trajectory variables
            for meta, trajectory in self.sweep.generate_trajectories():
                
                prev_env = None
                
                # Iterate sequentially through the trajectory (ascending n_units)
                for run_cfg in trajectory:
                    
                    tqdm.write(f"--- F: {run_cfg.n_families} | D: {run_cfg.latent_dim} | U: {run_cfg.n_units} | Sample: {meta['sample_id']} ---")
                    
                    # 1. Generate standard explicit receptor indices
                    indices = torch.tensor(
                        [[i for _ in range(run_cfg.k_sub)] for i in range(run_cfg.n_units)], 
                        dtype=torch.long, 
                        device="cuda" if torch.cuda.is_available() else "cpu"
                    )
                    
                    # Inject standard indices into the config so they appear in config.json
                    run_cfg.receptor_indices = indices.tolist()
                    
                    # 2. Get a dedicated logger for this exact node in the grid
                    node_logger = self.master_logger.get_run_logger(meta, run_cfg)
                    
                    # 3. Instantiate Runner and pass state
                    single_runner = SimulationRunner(config=run_cfg, logger=node_logger)
                    prev_env = single_runner.run(prev_env=prev_env, receptor_indices=indices)
                    
                    pbar.update(1)