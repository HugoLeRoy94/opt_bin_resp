# Documented in:
#   doc/theory/07_optimization_pipeline.md  (sweep architecture section)
"""
config.py — Run configuration and trajectory generation.

SingleRunConfig: scalar-only config consumed by SimulationRunner.
RunConfig: every parameter field accepts T (fixed) or List[T] (iteration axis).
Fields whose values are semantically arrays (conc_mean, conc_std,
kernel_params, measurement_fns) use Tuple when fixed and List[Tuple] when iterated,
so isinstance(val, list) cleanly identifies all iteration axes with no special-casing.

generate_trajectories() zips all list-valued axes (must share length L), sorts by
n_genes when warm_start=True, and yields a single List[SingleRunConfig] trajectory.
"""
from dataclasses import dataclass, asdict, field, fields as dc_fields
from typing import Union, List, Dict, Any, Generator, Tuple, Optional


_SWEEP_CONTROL_FIELDS = frozenset({"sweep_name", "base_folder", "warm_start"})

# Fields whose values are arrays (tuple = fixed, list-of-tuples = axis).
# Used when converting RunConfig values to lists for SingleRunConfig.
_TUPLE_FIELDS = frozenset({"kernel_params", "measurement_fns", "conc_mean", "conc_std"})


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

    # --- Presence (hierarchical source→ligand sampler) ---
    # n_presence_blocks: K source blocks; partition is deterministic from (n_ligands, K).
    # mu_sources: Poisson rate for the number of active sources per sniff.
    # mu_ligands_per_source: Poisson rate for the number of active ligands per source.
    # block_shared_conc_mean: share one concentration mean per block (gated on n_presence_blocks > 1).
    n_presence_blocks:       int
    mu_sources:              float
    mu_ligands_per_source:   float
    block_shared_conc_mean:  bool

    # --- Concentration ---
    conc_model_type: str
    conc_mean:       List[float]
    conc_std:        List[float]

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
    block_size:       int = 15               # blocked estimator: receptors per block (2^block_size bins)
    n_partitions:     int = 4                # blocked estimator: number of random partitions averaged
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
                    use_interface_model=self.use_interface_model,
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

    Scalar fields are fixed for all steps.  Any field set to a list becomes an
    iteration axis: all axes are zipped (not crossed), so every axis list must
    share the same length L, producing exactly L steps.

    Fields whose values are inherently arrays (conc_mean, conc_std,
    kernel_params, measurement_fns) are typed as Tuple when fixed and
    List[Tuple] when iterated.  isinstance(val, list) therefore cleanly separates
    axes from fixed values with no special-case logic.

    warm_start: when True, steps are sorted by n_genes ascending and a chain
    warm-start is applied whenever n_genes grows between consecutive steps.
    When False, steps run in the order they appear in the lists (no sorting,
    always cold-start).

    Concentration parameters (conc_mean, conc_std) are supplied directly as
    tuples (one vector per ligand).  No RNG-based range sampling is performed;
    the caller is responsible for generating appropriate values.
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

    # --- Presence (hierarchical source→ligand sampler) ---
    n_presence_blocks:      Union[int,   List[int]]
    mu_sources:             Union[float, List[float]]
    mu_ligands_per_source:  Union[float, List[float]]
    block_shared_conc_mean: Union[bool,  List[bool]]

    # --- Concentration model ---
    conc_model_type: Union[str, List[str]]

    # --- Concentration (direct; tuple = fixed, List[Tuple] = axis) ---
    conc_mean: Union[Tuple[float, ...], List[Tuple[float, ...]]]
    conc_std:  Union[Tuple[float, ...], List[Tuple[float, ...]]]

    # --- Physics ---
    n_genes:               Union[int,   List[int]]
    k_sub:                 Union[int,   List[int]]
    temperature:           Union[float, List[float]]
    affinity_kernel:       Union[str,   List[str]]
    kernel_params:         Union[Tuple[float, ...], List[Tuple[float, ...]]]

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
    measurement_fns: Union[Tuple[str, ...], List[Tuple[str, ...]]]

    # --- Evaluation chunking ---
    eval_chunk_size: Union[Optional[int], List[Optional[int]]] = None
    block_size:      Union[int,           List[int]] = 15
    n_partitions:    Union[int,           List[int]] = 4

    # --- Interface model ---
    use_interface_model: Union[bool, List[bool]] = False

    # --- Receptor sampling ---
    n_receptors:               Union[Optional[int], List[Optional[int]]] = None
    receptor_sampling_strategy: Union[str,          List[str]]          = "cascading"
    receptor_sampling_seed:    Union[Optional[int], List[Optional[int]]] = None

    # --- Sweep control (never forwarded to SingleRunConfig) ---
    sweep_name:  str  = "run"
    base_folder: str  = "/app/data"
    warm_start:  bool = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _axes(self) -> Dict[str, list]:
        """Returns {field_name: values} for every list-valued non-control field."""
        return {
            f.name: getattr(self, f.name)
            for f in dc_fields(self)
            if f.name not in _SWEEP_CONTROL_FIELDS
            and isinstance(getattr(self, f.name), list)
        }

    def is_sweep(self) -> bool:
        return bool(self._axes())

    # ------------------------------------------------------------------
    # Core generator
    # ------------------------------------------------------------------

    def generate_trajectories(self) -> Generator[List[SingleRunConfig], None, None]:
        """
        Yields a single List[SingleRunConfig] — the ordered trajectory.

        All list-valued non-control fields are zipped (not crossed).  They must
        all share the same length L; if not, a ValueError is raised naming every
        offending axis and its length.

        When warm_start=True the steps are sorted by (env_group, n_genes) before
        building SingleRunConfig objects.  Env groups are inferred from the input
        order: a new group begins each time n_genes does not strictly increase
        (i.e., the n_genes sweep restarts).  This keeps each fixed-environment
        chain contiguous and sorted by n_genes, so warm-starting is valid within
        a group and resets automatically between groups.
        When warm_start=False steps are emitted in natural (input) order.

        Tuple-typed fields (conc_mean, conc_std, kernel_params, measurement_fns)
        are converted to lists when forwarded to SingleRunConfig.
        """
        axes = self._axes()

        # --- Validate equal lengths ---
        if axes:
            lengths = {k: len(v) for k, v in axes.items()}
            if len(set(lengths.values())) > 1:
                offenders = "\n  ".join(
                    f"{k}: {n}" for k, n in sorted(lengths.items())
                )
                raise ValueError(
                    f"All axis lists must share the same length.\n  {offenders}"
                )
            L = next(iter(lengths.values()))
        else:
            L = 1

        # --- Fixed params forwarded to SingleRunConfig ---
        fixed = {
            f.name: getattr(self, f.name)
            for f in dc_fields(self)
            if f.name not in _SWEEP_CONTROL_FIELDS and f.name not in axes
        }

        # --- Sort indices by (env_group, n_genes) when warm_start is enabled ---
        # Groups are inferred from the input order: a new group starts each time
        # n_genes does not strictly increase (i.e., the sweep restarts).
        order = list(range(L))
        if self.warm_start and "n_genes" in axes:
            ng_vals = axes["n_genes"]
            gid = 0
            group_ids: list[int] = []
            for i, ng in enumerate(ng_vals):
                if i > 0 and ng <= ng_vals[i - 1]:
                    gid += 1
                group_ids.append(gid)
            order.sort(key=lambda i: (group_ids[i], axes["n_genes"][i]))

        # --- Build trajectory ---
        trajectory: List[SingleRunConfig] = []
        for i in order:
            step = {k: v[i] for k, v in axes.items()}
            run_params = {**fixed, **step}
            # Tuple-typed fields must arrive as lists at SingleRunConfig
            for k in _TUPLE_FIELDS:
                if k in run_params and isinstance(run_params[k], tuple):
                    run_params[k] = list(run_params[k])
            trajectory.append(SingleRunConfig(**run_params))

        yield trajectory

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        """Restore from a JSON-loaded dict.

        Strips unknown keys (so old configs with removed fields don't crash),
        converts array fields from JSON lists back to tuples or list-of-tuples,
        and applies backward-compat patches.
        """
        d = dict(d)
        # Strip fields that no longer exist in RunConfig
        valid_fields = {f.name for f in dc_fields(cls)}
        d = {k: v for k, v in d.items() if k in valid_fields}

        # Restore tuple fields from JSON arrays
        for fname in _TUPLE_FIELDS:
            if fname in d and isinstance(d[fname], list):
                if d[fname] and isinstance(d[fname][0], list):
                    # list-of-tuples (axis)
                    d[fname] = [tuple(v) for v in d[fname]]
                else:
                    # single fixed tuple
                    d[fname] = tuple(d[fname])

        # Backward compat: old configs used affinity_length_scale float
        if "affinity_length_scale" in d and "affinity_kernel" not in d:
            d["affinity_kernel"] = "gaussian"
            d["kernel_params"] = (d.pop("affinity_length_scale"),)

        # Backward compat: presence-model fields
        d.setdefault("n_presence_blocks", 1)
        d.setdefault("mu_sources", 1.0)
        d.setdefault("mu_ligands_per_source", 1.0)
        d.setdefault("block_shared_conc_mean", False)

        return cls(**d)

    def __str__(self) -> str:
        lines = ["\n=== RunConfig ==="]
        for f in dc_fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (list, tuple)) and len(value) > 10 and f.name not in ("measurement_fns",):
                lines.append(f"{f.name:<25}: <{type(value).__name__} of {len(value)} items>")
            else:
                lines.append(f"{f.name:<25}: {value}")
        lines.append("=================\n")
        return "\n".join(lines)
