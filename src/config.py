# Documented in:
#   doc/theory/07_optimization_pipeline.md  (sweep architecture section)
"""
config.py — Run configuration and sweep generation.

SingleRunConfig: scalar-only config consumed by SimulationRunner.
RunConfig: accepts Union[T, List[T]] for any parameter — list-valued fields become
sweep axes. generate_trajectories() yields (meta, trajectory) pairs via Cartesian
product of independent axes, with the warm_start_axis run sequentially within each
trajectory. Concentration draws are seeded per trajectory for full reproducibility.
"""
from dataclasses import dataclass, asdict, field, fields as dc_fields
from typing import Union, List, Dict, Any, Generator, Tuple, Optional
import itertools
import numpy as np
import warnings


# Fields that belong to RunConfig only and are never forwarded to SingleRunConfig.
_SWEEP_CONTROL_FIELDS = frozenset({
    "n_samples", "sweep_name", "base_folder", "warm_start_axis",
    "conc_mean_range", "conc_std_range", "p_presence_range", "seed",
})
# Fields that are always lists but are NOT sweep axes.
_ALWAYS_LIST_FIELDS = frozenset({"measurement_fns", "kernel_params"})
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

    # --- Presence correlation (Gaussian copula) ---
    # n_presence_blocks: number of source blocks; ligands within a block co-occur.
    # rho_block: within-block Gaussian correlation; 0 → independent Bernoulli.
    # block_shared_conc_mean: share one concentration mean per block (ignored when rho_block=0).
    n_presence_blocks:     int
    rho_block:             float
    block_shared_conc_mean: bool

    # --- Concentration ---
    conc_model_type: str
    conc_mean:       List[float]
    conc_std:        List[float]
    p_presence:      List[float]

    # --- Physics ---
    n_genes:               int
    k_sub:                 int
    temperature:           float
    affinity_kernel:       str          # "gaussian" or "quadratic"
    kernel_params:         List[float]  # [lambda] for gaussian, [] for quadratic

    # --- Mixture ---
    # Accepts int or the sentinel "auto" — resolved to an int by SimulationRunner._initialize().
    batch_size: Union[int, str]

    # --- Loss ---
    entropy:      str
    cov_weight:   Optional[float]
    penalty_type: Optional[str]
    n_c_bins:     int

    # --- Training ---
    epochs:          int
    lr:              float
    use_scheduler:   bool
    test_batch_size:  Union[int, str]        # "auto" resolved at init time
    eval_chunk_size:  Optional[int] = None   # per-forward-pass budget; None → use batch_size
    measurement_fns:  List[str] = field(default_factory=list)
    # None → SimulationRunner builds [[i]*k_sub for i in range(n_genes)]
    receptor_indices: Optional[List[List[int]]] = None
    # When set, receptor_indices is auto-generated via build_heteromer_array.
    n_receptors:               Optional[int] = None
    receptor_sampling_strategy: str          = "cascading"
    receptor_sampling_seed:    Optional[int] = None
    # When True, use the per-interface biophysics model (dual-face units, ordered ring).
    use_interface_model: bool = False

    def __post_init__(self):
        if self.receptor_indices is None:
            if self.n_receptors is not None:
                from src.geometry import build_heteromer_array  # local import — avoids circular dep
                tensor = build_heteromer_array(
                    self.n_genes, self.k_sub, self.n_receptors,
                    strategy=self.receptor_sampling_strategy,
                    seed=self.receptor_sampling_seed,
                )
                self.receptor_indices = tensor.tolist()
            else:
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

    warm_start_axis: controls which sweep axis forms the sequential "trajectory"
    within each sample, and determines the warm-start strategy used by
    SweepRunner.execute().  Defaults to "n_genes". Set to None to disable
    warm-starting entirely.

    Warm-start heuristic (applied at every step after the first):
      1. n_genes grew  → chain warm-start: the environment from the immediately
         preceding step is passed forward (existing gene-growth behaviour).
      2. n_genes unchanged → receptor fan-out: branch from the cached "square"
         baseline, i.e. the step in this trajectory where n_genes == n_receptors.
         Every n_receptors > n_genes therefore starts from the same root, NOT
         from the previous n_receptors result.
      3. Neither applies → cold start with a UserWarning.

    "n_genes" and "n_receptors" are mutually exclusive in warm_start_axis;
    SweepRunner.execute() raises ValueError if both are requested together.

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

    # --- Presence correlation (Gaussian copula) ---
    n_presence_blocks:      Union[int,   List[int]]    # sweep axis: number of source blocks
    rho_block:              Union[float, List[float]]  # sweep axis: within-block correlation
    block_shared_conc_mean: Union[bool,  List[bool]]   # sweep axis: block-shared conc mean

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
    affinity_kernel:       Union[str,   List[str]]   # "gaussian" or "quadratic"
    kernel_params:         List[float]               # always a list — not a sweep axis

    # --- Mixture ---
    batch_size: Union[int, str, List[Union[int, str]]]

    # --- Loss ---
    entropy:      Union[str,             List[str]]
    cov_weight:   Union[Optional[float], List[Optional[float]]]
    penalty_type: Union[Optional[str],   List[Optional[str]]]
    n_c_bins:     Union[int,             List[int]]

    # --- Training ---
    epochs:          Union[int,   List[int]]
    lr:              Union[float, List[float]]
    use_scheduler:   Union[bool,  List[bool]]
    test_batch_size: Union[int,   str,          List[Union[int, str]]]
    measurement_fns: List[str]   # always a list — not a sweep axis

    # --- Evaluation chunking ---
    eval_chunk_size: Union[Optional[int], List[Optional[int]]] = None  # per-forward-pass budget; None → use batch_size

    # --- Interface model ---
    use_interface_model: Union[bool, List[bool]] = False

    # --- Receptor sampling (generates receptor_indices in SingleRunConfig.__post_init__) ---
    n_receptors:               Union[Optional[int], List[Optional[int]]] = None
    receptor_sampling_strategy: Union[str,          List[str]]          = "cascading"
    receptor_sampling_seed:    Union[Optional[int], List[Optional[int]]] = None

    # --- Sweep control ---
    n_samples:       int                                   = 1
    sweep_name:      str                                   = "run"
    base_folder:     str                                   = "/app/data"
    # Single string: one warm-start axis.
    # List of strings: joint axes zipped together (sorted by the first listed axis).
    warm_start_axis: Optional[Union[str, List[str]]]       = "n_genes"
    seed:            int                                   = 0

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

        warm_start_axis may be:
          - None            : no warm-starting; trajectory has one element.
          - str             : single axis swept sequentially (existing behaviour).
          - List[str]       : joint axes zipped together, sorted by the first;
                              all listed axes must be list-valued with equal length.
        """
        rng = np.random.default_rng(self.seed)

        # Work on a mutable copy so pop() doesn't mutate self
        sweep = self._sweep_axes()

        # --- Resolve warm-start axes ---
        warm_spec = self.warm_start_axis
        if isinstance(warm_spec, list):
            warm_axes = warm_spec
        elif warm_spec is not None:
            warm_axes = [warm_spec]
        else:
            warm_axes = []

        # Pop all warm axes from sweep; collect their value lists.
        warm_axis_values: Dict[str, list] = {}
        for ax in warm_axes:
            if ax in sweep:
                warm_axis_values[ax] = sweep.pop(ax)

        # Build the sequence of per-step injection dicts, sorted by the first axis.
        if warm_axis_values:
            first_ax = warm_axes[0]
            if first_ax in warm_axis_values:
                order = sorted(
                    range(len(warm_axis_values[first_ax])),
                    key=lambda i: warm_axis_values[first_ax][i],
                )
                warm_steps: List[Dict] = [
                    {ax: warm_axis_values[ax][i]
                     for ax in warm_axes if ax in warm_axis_values}
                    for i in order
                ]
            else:
                warm_steps = [{}]
        else:
            warm_steps = [{}]  # single step; warm axis (if any) is scalar in fixed params

        # Build the dict of fixed scalar params forwarded to SingleRunConfig
        fixed = {
            f.name: getattr(self, f.name)
            for f in dc_fields(self)
            if f.name not in _NON_RUN_FIELDS
            and f.name not in sweep
            and not isinstance(getattr(self, f.name), list)
        }
        fixed["measurement_fns"] = self.measurement_fns
        fixed["kernel_params"]   = self.kernel_params

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

                trajectory = []
                for step_dict in warm_steps:
                    run_params = {
                        **params,
                        **step_dict,   # inject warm-axis values for this step
                        "conc_mean":  conc_mean,
                        "conc_std":   conc_std,
                        "p_presence": p_presence,
                    }
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
        # Backward compat: old configs used affinity_length_scale float
        if "affinity_length_scale" in d and "affinity_kernel" not in d:
            d["affinity_kernel"] = "gaussian"
            d["kernel_params"] = [d.pop("affinity_length_scale")]
        # Backward compat: Gaussian-copula fields added after initial release.
        # Old sweep_config.json files won't have these; default to independent
        # Bernoulli (rho_block=0) so existing sweep results remain loadable.
        d.setdefault("n_presence_blocks", 1)
        d.setdefault("rho_block", 0.0)
        d.setdefault("block_shared_conc_mean", False)
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
