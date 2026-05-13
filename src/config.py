from dataclasses import dataclass, asdict, field, fields as dc_fields
from typing import Union, List, Dict, Any, Generator, Tuple, Optional
import itertools
import numpy as np


# Fields that belong to RunConfig only and are never forwarded to SingleRunConfig.
_SWEEP_CONTROL_FIELDS = frozenset({
    "n_samples", "sweep_name", "base_folder", "warm_start_axis",
    "conc_mean_range", "conc_std_range", "p_presence_range", "seed",
})
# Fields that are always lists but are NOT sweep axes.
_ALWAYS_LIST_FIELDS = frozenset({"measurement_fns"})
_NON_RUN_FIELDS = _SWEEP_CONTROL_FIELDS | _ALWAYS_LIST_FIELDS


@dataclass
class SingleRunConfig:
    """
    Scalar-only per-run config consumed by SimulationRunner.
    No random side-effects, no derived tensors — pure value object.

    receptor_indices defaults to None, which signals SimulationRunner to build
    the standard homomer layout [[i]*k_sub for i in range(n_genes)].
    Pass explicit indices for heteromers.
    """

    # --- Environment ---
    n_families:              int
    n_ligands:               int
    latent_dim:              int
    family_spread:           float
    average_family_distance: float
    environment_geometry:    str
    distribution_type:       str
    observation_noise_sigma: float

    # --- Concentration ---
    conc_model_type: str
    conc_mean:       List[float]
    conc_std:        List[float]
    p_presence:      List[float]

    # --- Physics ---
    n_genes:               int
    k_sub:                 int
    temperature:           float
    affinity_length_scale: float

    # --- Mixture ---
    batch_size: int

    # --- Loss ---
    entropy:      str
    cov_weight:   Optional[float]
    penalty_type: Optional[str]
    n_c_bins:     int

    # --- Training ---
    epochs:          int
    lr:              float
    use_scheduler:   bool
    test_batch_size:  int
    eval_chunk_size:  Optional[int] = None   # per-forward-pass budget; None → use batch_size
    measurement_fns:  List[str] = field(default_factory=list)
    # None → SimulationRunner builds [[i]*k_sub for i in range(n_genes)]
    receptor_indices: Optional[List[List[int]]] = None

    def __post_init__(self):
        if self.receptor_indices is None:
            self.receptor_indices = [[i] * self.k_sub for i in range(self.n_genes)]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        lines = ["\n=== SingleRunConfig ==="]
        for key, value in asdict(self).items():
            if isinstance(value, list) and len(value) > 15:
                lines.append(f"{key:<25}: <list of {len(value)} items>")
            else:
                lines.append(f"{key:<25}: {value}")
        lines.append("=======================\n")
        return "\n".join(lines)


@dataclass
class RunConfig:
    """
    Unified config for single runs and parameter sweeps.

    Any parameter field accepts Union[T, list[T]] to declare it as a sweep axis.
    All-scalar fields → single run. Any list-valued field → sweep.

    warm_start_axis: when that field is a list, its values are iterated in
    ascending order within each sample and the environment state is passed
    forward between steps (warm-starting). Defaults to "n_genes". Set to None
    to disable warm-starting entirely.

    seed: controls the RNG used to draw conc_mean / conc_std / p_presence per
    trajectory. generate_trajectories() is fully deterministic given this seed,
    so SweepLoader can reconstruct the exact configs used during training.
    """

    # --- Environment ---
    n_families:              Union[int,   List[int]]
    n_ligands:               Union[int,   List[int]]
    latent_dim:              Union[int,   List[int]]
    family_spread:           Union[float, List[float]]
    average_family_distance: Union[float, List[float]]
    environment_geometry:    Union[str,   List[str]]
    distribution_type:       Union[str,   List[str]]
    observation_noise_sigma: Union[float, List[float]]

    # --- Concentration model ---
    conc_model_type: Union[str, List[str]]

    # --- Concentration ranges (sweep-level: one draw per trajectory) ---
    conc_mean_range:  Tuple[float, float]
    conc_std_range:   Tuple[float, float]
    p_presence_range: Tuple[float, float]

    # --- Physics ---
    n_genes:               Union[int,   List[int]]
    k_sub:                 Union[int,   List[int]]
    temperature:           Union[float, List[float]]
    affinity_length_scale: Union[float, List[float]]

    # --- Mixture ---
    batch_size: Union[int, List[int]]

    # --- Loss ---
    entropy:      Union[str,             List[str]]
    cov_weight:   Union[Optional[float], List[Optional[float]]]
    penalty_type: Union[Optional[str],   List[Optional[str]]]
    n_c_bins:     Union[int,             List[int]]

    # --- Training ---
    epochs:          Union[int,   List[int]]
    lr:              Union[float, List[float]]
    use_scheduler:   Union[bool,  List[bool]]
    test_batch_size: Union[int,   List[int]]
    measurement_fns: List[str]   # always a list — not a sweep axis

    # --- Evaluation chunking ---
    eval_chunk_size: Union[Optional[int], List[Optional[int]]] = None  # per-forward-pass budget; None → use batch_size

    # --- Sweep control ---
    n_samples:       int           = 1
    sweep_name:      str           = "run"
    base_folder:     str           = "/app/data"
    warm_start_axis: Optional[str] = "n_genes"
    seed:            int           = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sweep_axes(self) -> Dict[str, list]:
        """Returns {field_name: values} for all list-valued run-parameter fields."""
        return {
            f.name: getattr(self, f.name)
            for f in dc_fields(self)
            if f.name not in _NON_RUN_FIELDS and isinstance(getattr(self, f.name), list)
        }

    def is_sweep(self) -> bool:
        return bool(self._sweep_axes())

    # ------------------------------------------------------------------
    # Core generator
    # ------------------------------------------------------------------

    def generate_trajectories(self) -> Generator[Tuple[Dict, List[SingleRunConfig]], None, None]:
        """
        Yields (meta, trajectory) pairs, fully deterministic from self.seed.

        meta:       dict of independent-axis values + "sample_id" — used for
                    directory path construction and downstream analysis labels.
        trajectory: list of SingleRunConfig ordered ascending along
                    warm_start_axis (a single element when warm_start_axis is
                    scalar or None).
        """
        rng = np.random.default_rng(self.seed)

        # Work on a mutable copy so pop() doesn't mutate self
        sweep = self._sweep_axes()

        # Extract warm-start axis from independent axes
        warm_axis = self.warm_start_axis
        if warm_axis and warm_axis in sweep:
            warm_values = sorted(sweep.pop(warm_axis))
        else:
            warm_values = None

        # Build the dict of fixed scalar params forwarded to SingleRunConfig
        fixed = {
            f.name: getattr(self, f.name)
            for f in dc_fields(self)
            if f.name not in _NON_RUN_FIELDS
            and f.name not in sweep
            and not isinstance(getattr(self, f.name), list)
        }
        fixed["measurement_fns"] = self.measurement_fns

        # Cartesian product of independent sweep axes
        ind_keys = list(sweep.keys())
        ind_combos = list(itertools.product(*sweep.values())) if sweep else [()]

        for combo in ind_combos:
            combo_dict = dict(zip(ind_keys, combo))
            params = {**fixed, **combo_dict}
            n_ligands = params["n_ligands"]

            for sample_id in range(self.n_samples):
                # Draw concentration params once per trajectory (shared across warm steps)
                conc_mean  = rng.uniform(*self.conc_mean_range,  size=n_ligands).tolist()
                conc_std   = rng.uniform(*self.conc_std_range,   size=n_ligands).tolist()
                p_presence = rng.uniform(*self.p_presence_range, size=n_ligands).tolist()

                meta = {**combo_dict, "sample_id": sample_id}

                # Determine the sequence of values along the warm-start axis
                if warm_values is not None:
                    steps = warm_values          # swept list, run sequentially
                elif warm_axis is not None:
                    steps = [params[warm_axis]]  # scalar — single step
                else:
                    steps = [None]               # no warm axis — single step

                trajectory = []
                for val in steps:
                    run_params = {
                        **params,
                        "conc_mean":  conc_mean,
                        "conc_std":   conc_std,
                        "p_presence": p_presence,
                    }
                    if warm_values is not None:  # only inject when it's a real sweep axis
                        run_params[warm_axis] = val
                    trajectory.append(SingleRunConfig(**run_params))

                yield meta, trajectory

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # asdict converts tuples → lists; restore tuple ranges for round-trip safety
        for key in ("conc_mean_range", "conc_std_range", "p_presence_range"):
            if key in d:
                d[key] = list(d[key])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        """Restore from a JSON-loaded dict (converts range lists back to tuples)."""
        d = dict(d)
        for key in ("conc_mean_range", "conc_std_range", "p_presence_range"):
            if key in d and isinstance(d[key], list):
                d[key] = tuple(d[key])
        return cls(**d)

    def __str__(self) -> str:
        lines = ["\n=== RunConfig ==="]
        for f in dc_fields(self):
            value = getattr(self, f.name)
            if isinstance(value, list) and len(value) > 10 and f.name != "measurement_fns":
                lines.append(f"{f.name:<25}: <list of {len(value)} items>")
            else:
                lines.append(f"{f.name:<25}: {value}")
        lines.append("=================\n")
        return "\n".join(lines)
