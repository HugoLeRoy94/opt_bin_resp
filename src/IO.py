import os
import json
import csv
import pandas as pd
import torch
import numpy as np
from datetime import datetime
from src.config import SingleRunConfig, SweepConfig

# --- Custom Encoder ---
class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to seamlessly handle PyTorch tensors and NumPy types."""
    def default(self, obj):
        if isinstance(obj, torch.Tensor): return obj.cpu().tolist()
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)): return obj.item()        
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)): return 0.0 
        return super().default(obj)

# ==========================================
# LOGGERS (Writing Data)
# ==========================================

class ExperimentLogger:
    """Core logger for saving configs, stats, and checkpoints to a specific directory."""
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.stats_path = os.path.join(self.run_dir, "stats.csv")
        self.config_path = os.path.join(self.run_dir, "config.json")
        
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def save_config(self, config: SingleRunConfig):
        with open(self.config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=4, cls=CustomJSONEncoder)

    def save_stats(self, epoch: int, stats: dict):
        stats['epoch'] = epoch
        file_exists = os.path.isfile(self.stats_path)
        with open(self.stats_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(stats)

    def save_checkpoint(self, epoch: int, env, physics, receptor_indices, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "env_state": env.state_dict(),
            "physics_state": physics.state_dict(),
            "receptor_indices": receptor_indices.cpu() if isinstance(receptor_indices, torch.Tensor) else receptor_indices,
        }
        torch.save(checkpoint, os.path.join(self.ckpt_dir, f"checkpoint_epoch_{epoch:04d}.pt"))
        if is_best:
            torch.save(checkpoint, os.path.join(self.run_dir, "best_model.pt"))

class SingleRunLogger(ExperimentLogger):
    """A specialized logger for a node within a Sweep grid."""
    def __init__(self, sweep_root: str, meta: dict, config: SingleRunConfig):
        self.config = config
        
        # Enforce strict hierarchy: root / dim_X / sample_Y / units_Z
        rel_path = f"dim_{config.latent_dim}/sample_{meta['sample_id']}/units_{config.n_units}"
        run_dir = os.path.join(sweep_root, rel_path)
        super().__init__(run_dir)
        self.save_config(config)

class SweepLogger:
    """Initializes the master sweep folder and generates SingleRunLoggers."""
    def __init__(self, config: SweepConfig):
        self.config = config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sweep_root = os.path.join(config.base_folder, f"{config.sweep_name}_{timestamp}")
        
        os.makedirs(self.sweep_root, exist_ok=True)
        self._save_sweep_config()

    def _save_sweep_config(self):
        path = os.path.join(self.sweep_root, "sweep_config.json")
        with open(path, "w") as f:
            # Save the raw dictionary of the dataclass
            json.dump(self.config.__dict__, f, indent=4, cls=CustomJSONEncoder)

    def get_run_logger(self, meta: dict, run_config: SingleRunConfig) -> SingleRunLogger:
        return SingleRunLogger(self.sweep_root, meta, run_config)


# ==========================================
# LOADERS (Reading Data)
# ==========================================

class SingleRunLoader:
    """Loads data from a targeted single run directory."""
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        self.stats_path = os.path.join(run_dir, "stats.csv")
        self.config_path = os.path.join(run_dir, "config.json")
        
        if not os.path.exists(self.run_dir):
            raise FileNotFoundError(f"Directory {self.run_dir} does not exist.")

    def load_config(self) -> SingleRunConfig:
        with open(self.config_path, "r") as f:
            data = json.load(f)
            # Filter keys safely
            valid_keys = {k for k in SingleRunConfig.__dataclass_fields__.keys()}
            filtered = {k: v for k, v in data.items() if k in valid_keys}
            return SingleRunConfig(**filtered)

    def load_history(self) -> pd.DataFrame:
        return pd.read_csv(self.stats_path)

    def load_checkpoint(self, filename="best_model.pt", map_location="cpu"):
        return torch.load(os.path.join(self.run_dir, filename), map_location=map_location)


class SweepLoader:
    """Aggregates an entire Sweep directory into analysis-ready data structures."""
    def __init__(self, sweep_root: str):
        self.sweep_root = sweep_root
        self.config_path = os.path.join(sweep_root, "sweep_config.json")
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Sweep config missing at {self.config_path}")
            
        with open(self.config_path, "r") as f:
            data = json.load(f)
            self.config = SweepConfig(**data)

    def load_all_histories(self) -> pd.DataFrame:
        """
        Crawls the sweep grid, loads all stats.csv files, and injects metadata 
        (latent_dim, sample_id, n_units) to return one massive DataFrame.
        """
        all_dfs = []
        
        # Traverse the expected grid generated by the config
        for meta, trajectories in self.config.generate_trajectories():
            for run_config in trajectories:
                rel_path = f"dim_{run_config.latent_dim}/sample_{meta['sample_id']}/units_{run_config.n_units}"
                run_dir = os.path.join(self.sweep_root, rel_path)
                stats_file = os.path.join(run_dir, "stats.csv")
                
                if os.path.exists(stats_file):
                    df = pd.read_csv(stats_file)
                    # Inject identifying metadata for downstream analysis (e.g. Seaborn plotting)
                    df['latent_dim'] = run_config.latent_dim
                    df['sample_id'] = meta['sample_id']
                    df['n_units'] = run_config.n_units
                    all_dfs.append(df)
                    
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()