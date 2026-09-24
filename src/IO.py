import os
import json
import csv
import re
import warnings
from dataclasses import asdict as _dc_asdict
import pandas as pd
import torch
import numpy as np
from datetime import datetime
from src.config import SingleRunConfig, RunConfig, _TUPLE_FIELDS, _NESTED_TUPLE_FIELDS
from src.curation import sweep_curation

_TS_RE = re.compile(r'(\d{8}_\d{6})')


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

    Explicit cell repertoires are labelled by receptors per cell (a single
    count when uniform, otherwise hyphen-separated counts in cell order).
    Other array-valued axes are identified from the saved config.json.
    The timestamp leaf guarantees uniqueness when identical parameters are run
    multiple times.
    """
    axes = run_config._axes()
    parts = []
    path_fields = set(axes)
    if single_cfg.cell_receptors is not None:
        path_fields.add("cell_receptors")
    for k in sorted(path_fields):
        if k == "cell_receptors" and single_cfg.cell_receptors is not None:
            counts = [len(cell) for cell in single_cfg.cell_receptors]
            label = (str(counts[0]) if counts and len(set(counts)) == 1
                     else "-".join(map(str, counts)))
            parts.append(f"receptors_per_cell_{label}")
        # Skip array-valued axes — they don't fit in directory names
        elif k not in _TUPLE_FIELDS | _NESTED_TUPLE_FIELDS:
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

    def save_checkpoint(self, epoch: int, env, physics, receptor_indices, is_best: bool = False,
                        readout=None):
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
        if readout is not None:
            # Cell mode: W, theta (buffers) plus the calibrated T_cell, which is a
            # plain attribute and so absent from state_dict.
            checkpoint["readout_state"] = {k: v.cpu() for k, v in readout.state_dict().items()}
            checkpoint["readout_mode"]  = readout.mode
            checkpoint["readout_temperature"] = readout.temperature
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
        self.set_execution_state("running")

    def _save_sweep_config(self):
        path = os.path.join(self.sweep_root, "sweep_config.json")
        with open(path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=4, cls=CustomJSONEncoder)

    def set_execution_state(self, state: str):
        """Atomically record the small, machine-owned sweep execution state."""
        if state not in {"running", "complete", "failed", "interrupted"}:
            raise ValueError(f"Unknown execution state: {state}")
        path = os.path.join(self.sweep_root, ".state")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(state + "\n")
        os.replace(tmp, path)

    def get_run_logger(self, single_cfg: SingleRunConfig, run_timestamp: str) -> ExperimentLogger:
        rel_path = _run_rel_path(self.config, single_cfg, run_timestamp)
        run_dir  = os.path.join(self.sweep_root, rel_path)
        logger   = ExperimentLogger(run_dir)
        logger.save_config(single_cfg, run_timestamp)
        return logger


# ==========================================
# LOADERS  (Reading Data)
# ==========================================

def _normalise_config(data: dict) -> dict:
    """Drop keys no longer in SingleRunConfig and fill fields added after saving.

    The single definition of "how to read an old config.json". Both
    SingleRunLoader.load_config and the goal indexer go through it, so a run that
    loads must also index.
    """
    filtered = {k: v for k, v in data.items()
                if k in SingleRunConfig.__dataclass_fields__}
    # Backward compat: presence-model fields added after early sweeps were run.
    filtered.setdefault("n_presence_blocks", 1)
    filtered.setdefault("mu_sources", 1.0)
    filtered.setdefault("mu_ligands_per_source", 1.0)
    filtered.setdefault("block_shared_conc_mean", False)
    if filtered.get("entropy") == "renyi":
        filtered["entropy"] = "collision"
    return filtered


def _scalar_config_defaults() -> dict:
    """Declared defaults of the scalar SingleRunConfig fields.

    Indexing never instantiates the dataclass, so these are applied by hand; a run
    saved before a field existed then shows that field's default instead of a hole.
    """
    from dataclasses import MISSING
    return {name: f.default for name, f in SingleRunConfig.__dataclass_fields__.items()
            if f.default is not MISSING and not isinstance(f.default, (list, dict, tuple))}


class SingleRunLoader:
    """Loads data from a single-run directory."""

    def __init__(self, run_dir: str):
        self.run_dir     = run_dir
        self.ckpt_dir    = os.path.join(run_dir, "checkpoints")
        self.stats_path  = os.path.join(run_dir, "stats.csv")
        self.config_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(run_dir):
            raise FileNotFoundError(f"Directory {run_dir} does not exist.")

    def load_config_dict(self) -> dict:
        """Saved config as a plain dict, with the backward-compatibility fixes applied.

        Use this when you only need field VALUES.  load_config() additionally runs
        SingleRunConfig.__post_init__, which for a cell-mode run expands every gene
        set into its full receptor repertoire — correct, but far too slow to do
        once per run while indexing a whole goal folder.
        """
        with open(self.config_path) as f:
            return _normalise_config(json.load(f))

    def load_config(self) -> SingleRunConfig:
        return SingleRunConfig(**self.load_config_dict())

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
        """One row per completed run: scalar config fields plus metric means.

        Metric columns are named WITHOUT the `_mean` suffix here, unlike
        src.IO.index_goal, which keeps it. Use index_goal for a whole goal folder.
        """
        rows = []
        for single_cfg, run_dir in self.iter_run_dirs():
            json_path = os.path.join(run_dir, "test_results.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path) as f:
                data = json.load(f)
            row = {k: v for k, v in _dc_asdict(single_cfg).items()
                   if not isinstance(v, list)}
            row.update({k: float(np.mean(v)) for k, v in data.items()
                        if isinstance(v, list)})
            rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def find_run_dir(self, **filters) -> "str | None":
        """Return the absolute path of the first run matching all filter kwargs.

        Filter keys must be scalar SingleRunConfig field names
        (e.g. n_genes=3, average_family_distance=1.0).
        """
        for cfg, run_dir in self.iter_run_dirs():
            if all(getattr(cfg, k, None) == v for k, v in filters.items()):
                return run_dir
        return None

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
# RUN FILE HELPERS
# ==========================================

def run_files(rel_path: str, data_root: str) -> dict:
    """Return absolute paths to all standard files for a run.

    Parameters
    ----------
    rel_path  : run path relative to the goal folder, as in the `path` column
                of src.IO.index_goal
    data_root : the goal folder, parent of all its sweep directories

    Returns a dict with keys: run_dir, config, stats, test_results,
    best_model, checkpoints.  Values are absolute path strings regardless
    of whether the file currently exists on disk.
    """
    run_dir = os.path.join(data_root, rel_path)
    return {
        "run_dir":      run_dir,
        "config":       os.path.join(run_dir, "config.json"),
        "stats":        os.path.join(run_dir, "stats.csv"),
        "test_results": os.path.join(run_dir, "test_results.json"),
        "best_model":   os.path.join(run_dir, "best_model.pt"),
        "checkpoints":  os.path.join(run_dir, "checkpoints"),
    }


# ==========================================
# MODULE-LEVEL UTILITIES
# ==========================================

def index_goal(data_root: str, complete_only: bool = False) -> pd.DataFrame:
    """Crawl one goal folder and return one row per run directory.

    A goal is `data/<goal>/`, holding timestamped sweep folders.  This reads every
    `config.json` (through SingleRunLoader, so old configs get the same
    backward-compatibility fixes as everywhere else) and, where present, the means
    of `test_results.json`.

    Columns
    -------
    path, sweep_folder, sweep_name, sweep_date, run_timestamp, status,
    receptor_type, curation_state, curation_label, run_mtime  — bookkeeping derived
    from the path, the presence of test_results.json, and curation.csv.

    Note there is no `git_hash`. The retired runs.db had one, but it recorded the
    repository HEAD at INDEXING time, not at run time, so a rebuild stamped every
    run with the same current commit. It was misleading rather than useful.
    Every scalar config field, under its own name.
    `<metric>_mean` for each list-valued key of test_results.json.

    There is no stored index.  Crawling 650 runs takes about half a second, which
    is not worth a second copy of the data that can fall out of step with it.  If
    that ever changes, cache this frame with `to_parquet` and delete it whenever
    it looks stale — do NOT reintroduce an incrementally-updated index.
    """
    data_root = os.path.abspath(data_root)
    defaults = _scalar_config_defaults()
    curation: dict[str, tuple[str, str]] = {}
    rows = []
    for root, _dirs, files in os.walk(data_root):
        if "config.json" not in files:
            continue
        rel = os.path.relpath(root, data_root)
        if not _TS_RE.search(rel):
            continue                       # not a run directory; ignore stray json
        test_json = os.path.join(root, "test_results.json")
        complete = os.path.exists(test_json)
        if complete_only and not complete:
            continue
        try:
            cfg = SingleRunLoader(root).load_config_dict()
        except Exception as e:
            warnings.warn(f"Skipping {rel}: {e}")
            continue
        sweep_folder = rel.split(os.sep)[0]
        m = _TS_RE.search(sweep_folder)
        leaf = _TS_RE.search(os.path.basename(root))   # the run_<timestamp> leaf
        if sweep_folder not in curation:    # one lookup per sweep, not per run
            curation[sweep_folder] = sweep_curation(data_root, sweep_folder)
        state, label = curation[sweep_folder]
        row = dict(defaults)
        # Scalar config fields only: list-valued ones (conc_mean, cell_gene_sets,
        # receptor_indices, ...) do not fit one table cell. Read them per run with
        # SingleRunLoader when needed. Bools become ints so integer filters match.
        row.update({k: int(v) if isinstance(v, bool) else v
                    for k, v in cfg.items() if not isinstance(v, (list, dict, tuple))})
        row.update({
            "path":           rel,
            "sweep_folder":   sweep_folder,
            "sweep_name":     sweep_folder[:m.start()].rstrip("_") if m else sweep_folder,
            "sweep_date":     m.group(1) if m else None,
            "run_timestamp":  leaf.group(1) if leaf else None,
            "status":         "complete" if complete else "partial",
            "receptor_type":  "homomer" if cfg.get("n_receptors") is None else "heteromer",
            "curation_state": state,
            "curation_label": label,
            "run_mtime":      os.path.getmtime(root),
        })
        if complete:
            try:
                with open(test_json) as f:
                    results = json.load(f)
                for key, values in results.items():
                    if isinstance(values, list) and values:
                        row[f"{key}_mean"] = float(np.mean(values))
            except (OSError, ValueError, TypeError) as e:
                warnings.warn(f"Unreadable test_results.json in {rel}: {e}")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("path").reset_index(drop=True) if rows else pd.DataFrame()


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
