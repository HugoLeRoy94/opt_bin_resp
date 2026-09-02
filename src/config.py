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
_TUPLE_FIELDS = frozenset({"kernel_params", "measurement_fns", "conc_mean", "conc_std",
                           "cell_gene_probs", "cell_size_pmf"})

# cell_gene_sets is nested one level deeper (a tuple of gene tuples), so it needs
# its own round-trip handling rather than the flat _TUPLE_FIELDS rule.
_NESTED_TUPLE_FIELDS = frozenset({"cell_gene_sets"})


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

    # --- Training ---
    epochs:          int
    lr:              float
    use_scheduler:   bool
    test_batch_size:  Union[int, str]        # "auto" resolved at init time

    # --- Loss (optional, estimator-specific) ---
    cov_weight:   Optional[float] = None     # only used by proxy estimator
    penalty_type: Optional[str]   = None     # only used by proxy estimator
    n_c_bins:     int              = 10      # only used by mi_conc loss
    initial_temperature: Union[float, str] = "auto"  # "auto" or explicit float
    block_size:       int = 18               # blocked estimator: receptors per block (2^block_size bins)
    n_partitions:     int = 4                # blocked estimator: number of random partitions averaged
    # When True, gradient-checkpoint the blocked histogram (recompute in backward
    # instead of retaining it): ~+20% step time for a larger auto batch. See
    # resolve_batch_sizes. Blocked-family losses only; no effect on the result.
    recompute_backward: bool = False
    # Cap on the FINAL one-shot measurement batch when test_batch_size="auto" (the
    # per-epoch curve stays 4×train). Bounds the O(B²) KT cost of the closing test at
    # high R without shrinking training. None → no cap (auto uses min(2^R, memory)).
    test_max_batch:   Optional[int] = None
    # When False: skip the per-epoch _eval_stats measurement (log the training loss for
    # free instead) AND run the final test at 4×train. Saves the per-epoch re-sampling
    # cost; re-measure from the saved checkpoint if more samples are needed.
    per_epoch_measure: bool = True
    # torch.compile the KT per-tile kernel (fuses the launch-heavy elementwise loop).
    compile_kt: bool = False
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

    # --- Performance (exact / near-exact; no effect on the model) ---
    # Evaluate each DISTINCT energy source (gene, or unique +/- face pocket) once and
    # contract to receptors with one matmul, instead of gathering k_sub values per
    # receptor. Exact. Removes the (B, L, R, k_sub) intermediate (k_sub x memory) and,
    # in the interface model, ~n_genes^2/(R*k_sub) redundant pocket evaluations.
    use_composition: bool = True
    # Run the energy/EC50 matmuls under torch.autocast(bfloat16). ~2x on the dominant
    # tensors. NOT exact: bf16 has ~3 decimal digits, so entropies shift slightly.
    use_amp: bool = False

    # --- Cell mode (see src/cells.py) ---
    # Setting cell_gene_sets or n_cells switches the array from receptors to CELLS:
    # each cell expresses a set of genes and assembles every receptor those genes
    # allow.  receptor_indices is then DERIVED (the deduplicated union of all cells'
    # repertoires) and the entropy is computed over the C cells, not the R_pool
    # receptors.  Leave both None for the original receptor-array behaviour.
    cell_gene_sets:  Optional[List[List[int]]] = None   # explicit gene set per cell
    n_cells:         Optional[int]   = None             # sample this many gene sets
    cell_sampling_strategy: str      = "bernoulli"      # "bernoulli" or "size_pmf"
    cell_gene_probs: Optional[List[float]] = None       # bernoulli: P(gene u expressed)
    cell_size_pmf:   Optional[List[float]] = None       # size_pmf: P(cell expresses i+1 genes)
    cell_max_genes:  Optional[int]   = None             # bernoulli: reject cells above this
    cell_sampling_seed: Optional[int] = None
    cell_stoichiometry: str          = "multinomial"    # "multinomial" or "uniform"
    cell_readout:    str             = "threshold"      # "threshold", "noisy_or", "mean"
    cell_threshold:  Union[float, str] = "auto"         # "auto" → calibrated median drive
    # One scalar threshold shared by all cells.  Default OFF: the threshold is pinned
    # to the median of the drive and re-pinned every cell_recalibrate_every epochs, so
    # it is determined by the data rather than fitted — no free parameter to justify.
    # A learnable threshold collapses to the fair-coin degeneracy whenever the cell is
    # soft (see doc/theory/07 §3b.6); set True only for that ablation.
    cell_threshold_learnable: bool   = False
    # T_cell at the END of phase 2, as a FRACTION of the live spread of the drive (not
    # an absolute value — the drive is a weighted mean of probabilities, so its scale
    # is set by the environment).  0.01 leaves ~0.8% of (sniff, cell) pairs inside the
    # sigmoid's transition band, i.e. cells are effectively deterministic.  Raising it
    # toward 1.0 leaves cells soft, and a soft cell's output entropy is mostly
    # H(response | sniff), which carries NO information.
    cell_temperature: float          = 0.01
    cell_initial_temperature: Union[float, str] = "auto"
    # Receptors per pool chunk in the fused physics+readout pass. None → one pass.
    # Bounds peak memory at O(B·L·chunk) instead of O(B·L·R_pool); pair with
    # recompute_backward to make the saving hold during training too.
    cell_pool_chunk: Optional[int]   = None
    # Two-phase schedule (cell mode only).  Phase 1 = the first cell_phase_split of the
    # epochs: the RECEPTOR sharpness anneals to its final value while the cell is held
    # soft, so the chemistry arranges the drive around the threshold with live gradients
    # everywhere.  Phase 2 = the rest: the receptor sharpness is held and the CELL
    # sharpness anneals down to cell_temperature x spread.  Annealing both at once makes
    # the cell's operating point chase a drive distribution that is still moving.
    cell_phase_split: float          = 0.5
    # Re-pin the threshold to the median of the drive every N epochs (0 disables).
    # Costs one forward pass at the calibration batch size.  Also refreshes the drive
    # spread, so the phase-2 sharpness target tracks the live distribution instead of
    # one measured at epoch 0.  Ignored when cell_threshold is an explicit float.
    cell_recalibrate_every: int      = 25

    def is_cell_mode(self) -> bool:
        return self.cell_gene_sets is not None or self.n_cells is not None

    def __post_init__(self):
        if self.is_cell_mode():
            # Derive the receptor pool from the cells; sampling (if any) happens
            # once, here, so the resolved gene sets are persisted in config.json
            # and the run is exactly reproducible from it.
            from src.cells import build_cell_array  # local import — avoids circular dep
            cell_array = build_cell_array(
                k_sub=self.k_sub,
                n_genes=self.n_genes,
                gene_sets=self.cell_gene_sets,
                n_cells=self.n_cells,
                strategy=self.cell_sampling_strategy,
                gene_probs=self.cell_gene_probs,
                size_pmf=self.cell_size_pmf,
                max_genes=self.cell_max_genes,
                seed=self.cell_sampling_seed,
                stoichiometry=self.cell_stoichiometry,
                use_interface_model=self.use_interface_model,
            )
            self.cell_gene_sets   = [list(gs) for gs in cell_array.gene_sets]
            self.n_cells          = cell_array.n_cells
            self.receptor_indices = cell_array.receptor_indices.tolist()
            return

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

    # --- Training ---
    epochs:          Union[int,   List[int]]
    lr:              Union[float, List[float]]
    use_scheduler:   Union[bool,  List[bool]]
    test_batch_size: Union[int,   str,          List[Union[int, str]]]
    measurement_fns: Union[Tuple[str, ...], List[Tuple[str, ...]]]

    # --- Loss (optional, estimator-specific) ---
    cov_weight:   Union[Optional[float], List[Optional[float]]] = None
    penalty_type: Union[Optional[str],   List[Optional[str]]]   = None
    n_c_bins:     Union[int,             List[int]]              = 10
    initial_temperature: Union[float, str, List[Union[float, str]]] = "auto"
    eval_chunk_size: Union[Optional[int], List[Optional[int]]] = None
    block_size:      Union[int,           List[int]] = 18
    n_partitions:    Union[int,           List[int]] = 4
    recompute_backward: Union[bool, List[bool]] = False  # checkpoint blocked histogram → larger batch
    test_max_batch: Union[Optional[int], List[Optional[int]]] = None  # cap the final auto measurement (O(B²) KT)
    per_epoch_measure: Union[bool, List[bool]] = True  # False → skip per-epoch eval, final test at 4×train
    compile_kt: Union[bool, List[bool]] = False  # torch.compile the KT tile kernel

    # --- Interface model ---
    use_interface_model: Union[bool, List[bool]] = False

    # --- Performance ---
    use_composition: Union[bool, List[bool]] = True
    use_amp: Union[bool, List[bool]] = False

    # --- Receptor sampling ---
    n_receptors:               Union[Optional[int], List[Optional[int]]] = None
    receptor_sampling_strategy: Union[str,          List[str]]          = "cascading"
    receptor_sampling_seed:    Union[Optional[int], List[Optional[int]]] = None

    # --- Cell mode (see src/cells.py) ---
    # cell_gene_sets / cell_gene_probs / cell_size_pmf are inherently arrays, so
    # they follow the _TUPLE_FIELDS convention: tuple = fixed, list-of-tuples = axis.
    cell_gene_sets:  Union[Optional[Tuple[Tuple[int, ...], ...]],
                           List[Tuple[Tuple[int, ...], ...]]] = None
    n_cells:         Union[Optional[int], List[Optional[int]]] = None
    cell_sampling_strategy: Union[str, List[str]] = "bernoulli"
    cell_gene_probs: Union[Optional[Tuple[float, ...]], List[Tuple[float, ...]]] = None
    cell_size_pmf:   Union[Optional[Tuple[float, ...]], List[Tuple[float, ...]]] = None
    cell_max_genes:  Union[Optional[int], List[Optional[int]]] = None
    cell_sampling_seed: Union[Optional[int], List[Optional[int]]] = None
    cell_stoichiometry: Union[str, List[str]] = "multinomial"
    cell_readout:    Union[str, List[str]] = "threshold"
    cell_threshold:  Union[float, str, List[Union[float, str]]] = "auto"
    cell_threshold_learnable: Union[bool, List[bool]] = False
    cell_temperature: Union[float, List[float]] = 0.01
    cell_initial_temperature: Union[float, str, List[Union[float, str]]] = "auto"
    cell_pool_chunk: Union[Optional[int], List[Optional[int]]] = None
    cell_phase_split: Union[float, List[float]] = 0.5
    cell_recalibrate_every: Union[int, List[int]] = 25

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
            for k in _NESTED_TUPLE_FIELDS:
                if k in run_params and isinstance(run_params[k], tuple):
                    run_params[k] = [list(v) for v in run_params[k]]
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

        # cell_gene_sets: [[0,2],[1]] (fixed) vs [[[0,2],[1]], ...] (axis)
        for fname in _NESTED_TUPLE_FIELDS:
            v = d.get(fname)
            if isinstance(v, list) and v:
                is_axis = isinstance(v[0], list) and v[0] and isinstance(v[0][0], list)
                d[fname] = ([tuple(tuple(g) for g in step) for step in v] if is_axis
                            else tuple(tuple(g) for g in v))

        # Backward compat: old configs used affinity_length_scale float
        if "affinity_length_scale" in d and "affinity_kernel" not in d:
            d["affinity_kernel"] = "gaussian"
            d["kernel_params"] = (d.pop("affinity_length_scale"),)

        # Backward compat: presence-model fields
        d.setdefault("n_presence_blocks", 1)
        d.setdefault("mu_sources", 1.0)
        d.setdefault("mu_ligands_per_source", 1.0)
        d.setdefault("block_shared_conc_mean", False)

        # Backward compat: renyi → collision rename
        if d.get("entropy") == "renyi":
            d["entropy"] = "collision"
        elif isinstance(d.get("entropy"), list):
            d["entropy"] = ["collision" if e == "renyi" else e for e in d["entropy"]]

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
