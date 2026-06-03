import os
import json
import csv
import warnings
from dataclasses import asdict as _dc_asdict
import pandas as pd
import torch
import numpy as np
from datetime import datetime
from src.config import SingleRunConfig, RunConfig, _TUPLE_FIELDS


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

def _run_rel_path(run_config: RunConfig, single_cfg: SingleRunConfig, run_timestamp: str) -> str:
    """
    Builds the relative path for one run within a sweep.

    Structure:
        {scalar_axis_1}_{val}/ ... {scalar_axis_N}_{val}/   (sorted alphabetically)
        run_{timestamp}/

    Only scalar-valued axes appear in directory names; tuple-typed axes
    (conc_mean, conc_std, p_presence, kernel_params, measurement_fns) are
    too large for path components and are identified from the saved config.json.
    The timestamp leaf guarantees uniqueness when identical parameters are run
    multiple times.
    """
    axes = run_config._axes()
    parts = []
    for k in sorted(axes.keys()):
        # Skip array-valued axes — they don't fit in directory names
        if k not in _TUPLE_FIELDS:
            parts.append(f"{k}_{getattr(single_cfg, k)}")
    parts.append(f"run_{run_timestamp}")
    return os.path.join(*parts) if parts else f"run_{run_timestamp}"


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

    def save_config(self, config: SingleRunConfig, run_timestamp: str = ""):
        d = config.to_dict()
        if run_timestamp:
            d["run_timestamp"] = run_timestamp
        with open(self.config_path, "w") as f:
            json.dump(d, f, indent=4, cls=CustomJSONEncoder)

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

    def get_run_logger(self, single_cfg: SingleRunConfig, run_timestamp: str) -> ExperimentLogger:
        rel_path = _run_rel_path(self.config, single_cfg, run_timestamp)
        run_dir  = os.path.join(self.sweep_root, rel_path)
        logger   = ExperimentLogger(run_dir)
        logger.save_config(single_cfg, run_timestamp)
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
        self.sweep_root = sweep_root
        config_path = os.path.join(sweep_root, "sweep_config.json")
        self.config = None
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    self.config = RunConfig.from_dict(json.load(f))
            except Exception as e:
                warnings.warn(f"Could not parse sweep_config.json in {sweep_root}: {e}")

    def iter_run_dirs(self):
        """Yields (single_cfg, run_dir) for every run directory found on disk.

        Crawls the sweep root recursively for config.json files.  Partial or
        missing runs (no config.json) are silently skipped.  This approach is
        independent of the path-generation logic and tolerates interrupted sweeps.
        """
        for root, _dirs, files in os.walk(self.sweep_root):
            if root == self.sweep_root:
                continue
            if "config.json" in files:
                try:
                    yield SingleRunLoader(root).load_config(), root
                except Exception as e:
                    warnings.warn(f"Skipping {root}: {e}")

    def load_all_test_results(self) -> pd.DataFrame:
        """
        Crawls the sweep, loads all test_results.json files, and returns a
        DataFrame with one row per run.  Each metric column holds the mean over
        the test-sample array stored in the JSON.

        Column priority (highest wins):
          2. metric values    (from test_results.json)
          1. scalar config fields (from single_cfg — skips list-valued fields
             such as conc_mean, p_presence, receptor_indices).
        """
        rows = []
        for single_cfg, run_dir in self.iter_run_dirs():
            json_path = os.path.join(run_dir, "test_results.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path) as f:
                data = json.load(f)

            # Priority 1 — scalar config fields (background context)
            row = {k: v for k, v in _dc_asdict(single_cfg).items()
                   if not isinstance(v, list)}
            # Priority 2 — metric means
            row.update({k: float(np.mean(v)) for k, v in data.items()
                        if isinstance(v, list)})
            rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def load_all_histories(self) -> pd.DataFrame:
        """
        Crawls the sweep, loads all stats.csv files, and injects scalar config
        columns from SingleRunConfig for downstream analysis.
        """
        all_dfs = []
        for single_cfg, run_dir in self.iter_run_dirs():
            stats_file = os.path.join(run_dir, "stats.csv")
            if not os.path.exists(stats_file):
                continue
            df = pd.read_csv(stats_file)
            for k, v in _dc_asdict(single_cfg).items():
                if not isinstance(v, list):
                    df[k] = v
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
    import re
    _ts_re = re.compile(r'(\d{8}_\d{6})')

    def _sort_key(d: Path):
        m = _ts_re.search(d.name)
        return m.group(1) if m else d.stat().st_mtime

    return [str(d) for d in sorted(dirs, key=_sort_key, reverse=True)]
