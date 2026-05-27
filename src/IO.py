import os
import json
import csv
from dataclasses import asdict as _dc_asdict
import pandas as pd
import torch
import numpy as np
from datetime import datetime
from src.config import SingleRunConfig, RunConfig


# ==========================================
# SERIALISATION HELPERS
# ==========================================

class CustomJSONEncoder(json.JSONEncoder):
    """Handles PyTorch tensors and NumPy types transparently."""
    def default(self, obj):
        if isinstance(obj, torch.Tensor):            return obj.cpu().tolist()
        if isinstance(obj, np.ndarray):              return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)): return obj.item()
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)): return 0.0
        return super().default(obj)


# ==========================================
# PATH HELPERS
# ==========================================

def _run_rel_path(run_config: RunConfig, meta: dict, single_cfg: SingleRunConfig) -> str:
    """
    Builds the relative path for one run within a sweep.

    Structure:
        {ind_axis_1}_{val}/ ... {ind_axis_N}_{val}/   (sorted, only if swept)
        sample_{id}/
        {warm_axis}_{val}/                             (only if warm axis is swept)

    Only axes that actually vary appear in the path, keeping single-value
    sweeps from creating redundant nesting.
    """
    parts = []
    for k in sorted(k for k in meta if k != "sample_id"):
        parts.append(f"{k}_{meta[k]}")
    parts.append(f"sample_{meta['sample_id']}")

    warm_axis = run_config.warm_start_axis
    if warm_axis and isinstance(getattr(run_config, warm_axis, None), list):
        parts.append(f"{warm_axis}_{getattr(single_cfg, warm_axis)}")

    return os.path.join(*parts)


# ==========================================
# LOGGERS  (Writing Data)
# ==========================================

class ExperimentLogger:
    """Core logger: saves configs, training stats, and checkpoints to a directory."""

    def __init__(self, run_dir: str):
        self.run_dir  = run_dir
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        self.stats_path  = os.path.join(run_dir, "stats.csv")
        self.config_path = os.path.join(run_dir, "config.json")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def save_config(self, config: SingleRunConfig):
        with open(self.config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=4, cls=CustomJSONEncoder)

    def save_stats(self, epoch: int, stats: dict):
        stats["epoch"] = epoch
        file_exists = os.path.isfile(self.stats_path)
        with open(self.stats_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(stats)

    def save_checkpoint(self, epoch: int, env, physics, receptor_indices, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "env_state":    env.state_dict(),
            "physics_state": physics.state_dict(),
            "receptor_indices": (
                receptor_indices.cpu()
                if isinstance(receptor_indices, torch.Tensor)
                else receptor_indices
            ),
        }
        path = os.path.join(self.ckpt_dir, f"checkpoint_epoch_{epoch:04d}.pt")
        torch.save(checkpoint, path)
        if is_best:
            torch.save(checkpoint, os.path.join(self.run_dir, "best_model.pt"))


class SweepLogger:
    """Initialises the master sweep directory and vends per-run loggers."""

    def __init__(self, config: RunConfig):
        self.config = config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sweep_root = os.path.join(
            config.base_folder, f"{config.sweep_name}_{timestamp}"
        )
        os.makedirs(self.sweep_root, exist_ok=True)
        self._save_sweep_config()

    def _save_sweep_config(self):
        path = os.path.join(self.sweep_root, "sweep_config.json")
        with open(path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=4, cls=CustomJSONEncoder)

    def get_run_logger(self, meta: dict, single_cfg: SingleRunConfig) -> ExperimentLogger:
        rel_path = _run_rel_path(self.config, meta, single_cfg)
        run_dir  = os.path.join(self.sweep_root, rel_path)
        logger   = ExperimentLogger(run_dir)
        logger.save_config(single_cfg)
        return logger


# ==========================================
# LOADERS  (Reading Data)
# ==========================================

class SingleRunLoader:
    """Loads data from a single-run directory."""

    def __init__(self, run_dir: str):
        self.run_dir     = run_dir
        self.ckpt_dir    = os.path.join(run_dir, "checkpoints")
        self.stats_path  = os.path.join(run_dir, "stats.csv")
        self.config_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(run_dir):
            raise FileNotFoundError(f"Directory {run_dir} does not exist.")

    def load_config(self) -> SingleRunConfig:
        with open(self.config_path) as f:
            data = json.load(f)
        valid_keys = set(SingleRunConfig.__dataclass_fields__)
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        # Backward compat: Gaussian-copula fields added after initial release.
        # Old per-run config.json files won't have these; default to independent
        # Bernoulli (rho_block=0) so existing results remain loadable.
        filtered.setdefault("n_presence_blocks", 1)
        filtered.setdefault("rho_block", 0.0)
        filtered.setdefault("block_shared_conc_mean", False)
        return SingleRunConfig(**filtered)

    def load_history(self) -> pd.DataFrame:
        return pd.read_csv(self.stats_path)

    def load_checkpoint(self, filename: str = "best_model.pt", map_location: str = "cpu"):
        return torch.load(os.path.join(self.run_dir, filename), map_location=map_location)


class SweepLoader:
    """Aggregates a full sweep directory into analysis-ready data structures."""

    def __init__(self, sweep_root: str):
        self.sweep_root  = sweep_root
        config_path = os.path.join(sweep_root, "sweep_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Sweep config missing at {config_path}")
        with open(config_path) as f:
            self.config = RunConfig.from_dict(json.load(f))

    def iter_run_dirs(self):
        """Yields (meta, single_cfg, run_dir) for every run in the sweep."""
        for meta, trajectory in self.config.generate_trajectories():
            for single_cfg in trajectory:
                rel = _run_rel_path(self.config, meta, single_cfg)
                yield meta, single_cfg, os.path.join(self.sweep_root, rel)

    def load_all_test_results(self) -> pd.DataFrame:
        """
        Crawls the sweep grid, loads all test_results.json files, and returns a
        DataFrame with one row per run.  Each metric column holds the mean over
        the test-sample array stored in the JSON.

        Column priority (highest wins):
          4. warm-axis value  (from single_cfg via warm_start_axis)
          3. swept-axis values (from meta dict)
          2. metric values    (from test_results.json)
          1. scalar config fields (from single_cfg — lowest priority, provides
             context for axes that happen to be fixed in a given sweep, e.g.
             n_ligands when not swept).  List-valued fields (conc_mean, p_presence,
             receptor_indices, …) are skipped.
        """
        rows = []
        for meta, single_cfg, run_dir in self.iter_run_dirs():
            json_path = os.path.join(run_dir, "test_results.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path) as f:
                data = json.load(f)

            # Priority 1 — scalar config fields (background context)
            row = {k: v for k, v in _dc_asdict(single_cfg).items()
                   if not isinstance(v, list)}
            # Priority 2 — metric values
            row.update({k: float(np.mean(v)) for k, v in data.items()
                        if isinstance(v, list)})
            # Priority 3 — swept axes + sample_id
            row.update(meta)
            # Priority 4 — warm axis
            warm_axis = self.config.warm_start_axis
            if warm_axis and isinstance(getattr(self.config, warm_axis, None), list):
                row[warm_axis] = getattr(single_cfg, warm_axis)

            rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def load_all_histories(self) -> pd.DataFrame:
        """
        Crawls the sweep grid, loads all stats.csv files, and injects metadata
        columns for downstream analysis.

        All scalar (non-list) fields from SingleRunConfig are added to every row
        so that fixed axes (e.g. n_ligands when not swept) are always queryable.
        Swept axes from meta and the warm axis take precedence.
        """
        all_dfs = []
        for meta, single_cfg, run_dir in self.iter_run_dirs():
            stats_file = os.path.join(run_dir, "stats.csv")
            if not os.path.exists(stats_file):
                continue
            df = pd.read_csv(stats_file)
            # Scalar config fields as background context
            for k, v in _dc_asdict(single_cfg).items():
                if not isinstance(v, list):
                    df[k] = v
            # Swept axes + sample_id override
            for k, v in meta.items():
                df[k] = v
            # Warm axis override
            warm_axis = self.config.warm_start_axis
            if warm_axis and isinstance(getattr(self.config, warm_axis, None), list):
                df[warm_axis] = getattr(single_cfg, warm_axis)
            all_dfs.append(df)
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# ==========================================
# MODULE-LEVEL UTILITIES
# ==========================================

def find_latest_sweep(base_dir: str, prefix: str = "") -> list[str]:
    """
    Returns sweep directories under base_dir ordered from most to least recently modified.
    Optionally filters by a name prefix (e.g. "latent_dim_sweep").
    Index [0] is the latest, [1] the second latest, etc.
    Raises FileNotFoundError if nothing matches.
    """
    from pathlib import Path
    dirs = [d for d in Path(base_dir).iterdir()
            if d.is_dir() and d.name.startswith(prefix)]
    if not dirs:
        raise FileNotFoundError(
            f"No directories matching '{prefix}*' found in {base_dir}"
        )
    return [str(d) for d in sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)]
