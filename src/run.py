# Documented in:
#   doc/theory/07_optimization_pipeline.md  (stages 5 & 6: training loop, evaluation)
"""
run.py — Training orchestration (SimulationRunner) and parameter sweep execution (SweepRunner).

SimulationRunner.run() executes: initialize → train → checkpoint → test.
Key behaviours:
  - Temperature annealing: linear from T_init (calibrated) to T_final.
  - Warm-starting: environment state passed forward along the warm_start_axis sweep,
    with LR damped 10× to preserve learned representations.
  - Chunked evaluation: soft metrics (Rényi, distances) on a single chunk; hard
    codeword metrics accumulated across all chunks for the full test_batch_size budget.
  - Measurement dispatch: functions selected by name from MEASUREMENT_REGISTRY,
    called via inspect.signature to inject only the arguments they accept.
"""
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import record_function   # [profiling] inert unless a profiler is active
import inspect
from datetime import datetime
from tqdm import tqdm
from typing import Optional

# --- Local Imports ---
from src.config import SingleRunConfig, RunConfig
from src.IO import ExperimentLogger, SweepLogger

from src import (LigandEnvironment,
                 SymmetricLigandEnvironment,
                 BinaryReceptor,
                 LogNormalConcentration,
                 NormalConcentration)
from src.physics import compute_initial_temperature

from src.analysis_helper import (
    full_array_entropy,
    entropy_collision,
    entropy_blocked,
    entropy_blocked_corrected,
    entropy_kt,
    entropy_kt_upper,
    codeword_entropy,
    conditional_entropy_response,
    mutual_information_kt,
    mutual_information_kt_upper,
    mutual_information_counting,
    response_counting_metrics,
    miller_madow_entropy,
    mean_receptor_distance,
    conditional_entropy_ligand,
    mutual_information_ligand,
    conditional_entropy_concentration,
    mutual_information_concentration,
    concentration_channel,
    identity_channel,
    conditional_entropy_family,
    mutual_information_family,
    conditional_entropy_block,
    mutual_information_block,
    receptor_distances,
    rank_ordered_distances,
    mean_specialization_index,
    receptor_conditioned_entropy
)

from src.bin_loss import (DiscreteExactLoss, KTMutualInformationLoss, KT_EPS,
                          compute_kt_entropy, compute_kt_upper_entropy,
                          compute_response_conditional_entropy)
from src.annealed_loss import AnnealedEntropyLoss, BlockedToCorrectedLoss
from src.family_mi_loss import MaximizeMutualInformationLigandLoss
from src.concentration_mi_loss import MaximizeMutualInformationConcentrationLoss

# ==========================================
# REGISTRIES
# ==========================================

MEASUREMENT_REGISTRY = {
    "full_array_entropy":                full_array_entropy,
    "entropy_collision":                 entropy_collision,
    "entropy_blocked":                   entropy_blocked,
    "entropy_blocked_corrected":         entropy_blocked_corrected,
    "entropy_kt":                        entropy_kt,
    "entropy_kt_upper":                  entropy_kt_upper,
    "codeword_entropy":                  codeword_entropy,
    "conditional_entropy_response":      conditional_entropy_response,
    "mutual_information_kt":             mutual_information_kt,
    "mutual_information_kt_upper":       mutual_information_kt_upper,
    "mutual_information_counting":       mutual_information_counting,
    "mean_receptor_distance":            mean_receptor_distance,
    "conditional_entropy_ligand":        conditional_entropy_ligand,
    "mutual_information_ligand":         mutual_information_ligand,
    "conditional_entropy_concentration": conditional_entropy_concentration,
    "mutual_information_concentration":  mutual_information_concentration,
    "concentration_channel":             concentration_channel,
    "identity_channel":                  identity_channel,
    "conditional_entropy_family":        conditional_entropy_family,
    "mutual_information_family":         mutual_information_family,
    "conditional_entropy_block":         conditional_entropy_block,
    "mutual_information_block":          mutual_information_block,
    "receptor_distances":                receptor_distances,
    "rank_ordered_distances":            rank_ordered_distances,
    "mean_specialization_index":         mean_specialization_index,
    "receptor_conditioned_entropy":      receptor_conditioned_entropy,
}

# ---------------------------------------------------------------------------
# Batch-size auto-scaling
# ---------------------------------------------------------------------------

# collision retains ~n_chunks (R,m,m) blocks in the training graph, so peak is
# n_chunks·R·m²·4 at fixed m. This sets how many blocks coexist: raising it shrinks
# the chunk m (lower log2(m) ceiling) in exchange for more pairs / a bigger batch B
# at the SAME peak memory. The knob to turn when you need larger collision batches.
COLLISION_TARGET_CHUNKS = 4

# KT with recompute_backward is COMPUTE-bound, not memory-bound: gradient
# checkpointing drops the retained graph to O(B·R), but the O(B²·R) work is
# unchanged. This caps the per-step work at KT_COMPUTE_BUDGET Bhattacharyya
# pair-evaluations, so the auto batch is B = sqrt(KT_COMPUTE_BUDGET / R) — the
# same sqrt(1/R) shape as the memory cap, just bounded by time instead of RAM.
# Reference: the memory-bound work is B²·R = mem_budget/(4·SAFETY) ≈ 5.3e9 on an
# 80 GiB A100 (R20≈16k / R45≈11k samples); 4× that ⇒ ~2× the batch, ~4× the step
# time. Independent of GPU size (it is a time budget). Raise it for more samples.
KT_COMPUTE_BUDGET = 4 * 5.3e9   # ≈ 2.1e10 pair-evaluations / training step

# KT training tile (edge length m of the (m,n,R) Bhattacharyya block). During training
# the retained graph is B²·R regardless of m (all tiles kept for backward), so m only
# sets the TRANSIENT (torch.log output + its backward grad) at m²·R — which rides under
# the already-present B²·R. A bigger tile ⇒ fewer, larger (fused) kernels ⇒ fewer launches
# (KT is launch-bound), for a small transient-memory cost. 4096 quarters the tile count
# vs 2048 for ~1 GB extra at R=20. NOT used for eval, where the (m,B) row buffer scales
# with the (huge) eval B — eval keeps its own 2048 tile (see EVAL_TILE / _eval_stats).
KT_TRAIN_TILE = 4096


def resolve_batch_sizes(
    n_receptors: int,
    entropy_type: str = "shannon",
    n_ligands: int = 1,
    k_sub: int = 1,
    mem_budget_bytes: Optional[int] = None,
    block_size: int = 15,
    n_partitions: int = 4,
    recompute_backward: bool = False,
    test_max_batch: Optional[int] = None,
    n_physics_receptors: Optional[int] = None,
) -> tuple:
    """Returns (batch_size, test_perepoch, test_final, collision_chunk_size).

    collision_chunk_size is the largest collision-chunk m that fits in GPU
    memory (the binding tensor is (R, m, m) float32 with a 3× safety factor
    for backward).  Returned only for the collision estimator (None otherwise).
    The batch is rounded up to a multiple of m so chunking wastes no samples.

    The estimator dictates the dominant entropy-side tensor, so each gets its
    own cap (memory model in parentheses):
      shannon   : B = 2^R coverage, capped by (B, 2^R) float32.        — only R<~15
      collision : B = 16·√(2^R), rounded to multiple of m_max, physics-capped.
                  Memory for collision is per-chunk (R, m, m), not (B, B).
      blocked   : capped by ceil(R/block_size)·n_partitions histograms of
                shape (B, 2^block_size) — independent of R, so B stays large.
      proxy / mi_* : O(B·R²)/O(B·R), no exponential or B² term; physics-bound.

    Physics bottleneck cap (all estimators): the interface-model forward+backward
    holds many (B, n_ligands, R·k_sub) float32 tensors at once (see below); a 16×
    safety factor is applied. mem_budget_bytes should be the free GPU memory at
    _initialize time; defaults to 8 GiB when CUDA is unavailable.

    block_size / n_partitions must match the DiscreteExactLoss config (defaults
    15 / 4) for the blocked cap to be correct.

    n_receptors vs n_physics_receptors: in the receptor picture these coincide.
    In the CELL picture they decouple — the entropy is over the C cells while the
    physics runs over the (much larger) receptor pool R_pool.  Pass n_receptors=C
    and n_physics_receptors=R_pool so the estimator caps size on C and only the
    physics cap sees R_pool.  Defaults to n_receptors when omitted.

    When test_batch_size == "auto": the per-epoch measurement uses test_perepoch
    (4 × batch_size, capped at test_final) and the single final measurement uses
    test_final (= min(2^R, memory)). An explicit test_batch_size is used for both.
    """
    # KT entropy and information share the same pairwise work and memory budget.
    if entropy_type == 'kt_mi':
        entropy_type = 'kt'
    B_min = 512
    collision_chunk_size = None
    if mem_budget_bytes is None:
        mem_budget_bytes = 8 * (1 << 30)  # 8 GiB fallback

    # Statistical saturation caps — estimator-specific upper bound on useful B.
    # Beyond these, extra samples yield negligible variance reduction.
    #   shannon / blocked : need ~100 samples per histogram bin → 100 · 2^bin_dim
    #     bin_dim = min(R, block_size) (blocked reduces to exact Shannon when R < block_size)
    #   collision         : 16·√(2^R) pairs; already encoded as the starting value below
    #   proxy / mi_*      : marginal entropies converge fast; 200·R samples is generous
    # bin_dim: effective state-space dimension shared by blocked and collision caps.
    # For R < block_size the blocked estimator is exact Shannon over 2^R states;
    # for R ≥ block_size each block has 2^block_size states. Collision at small R
    # also benefits from the same 100-samples-per-state floor before the
    # large-R heuristic (16·√(2^R)) takes over.
    bin_dim   = min(n_receptors, block_size)
    stats_cap = {
        "shannon":  max(B_min, 100 * (1 << n_receptors)),
        "collision": max(B_min, max(100 * (1 << bin_dim), 16 * int(2 ** (n_receptors / 2)))),
        "kt":        max(B_min, max(100 * (1 << bin_dim), 16 * int(2 ** (n_receptors / 2)))),
        "blocked":              max(B_min, 100 * (1 << bin_dim)),
        "blocked_corrected":    max(B_min, 100 * (1 << bin_dim)),
        "annealed":             max(B_min, 100 * (1 << bin_dim)),
        "blocked_to_corrected": max(B_min, 100 * (1 << bin_dim)),
    }.get(entropy_type, max(B_min, 200 * n_receptors))

    if entropy_type == "shannon":
        b_train = stats_cap
        # Hard memory cap: soft_assign is (B, 2^R) float32; 4× safety for backward.
        entropy_cap = max(B_min, mem_budget_bytes // ((1 << n_receptors) * 4 * 4))
        b_train = min(b_train, entropy_cap)
    elif entropy_type == "collision":
        # Per-block tensor (R,m,m) is fed to torch.log, so autograd RETAINS it for
        # backward. The single adjacent-chunk loop keeps ~n_chunks of them alive ⇒
        # training peak ≈ n_chunks·R·m²·4 = B·R·m·4 (LINEAR in m). Size m so
        # COLLISION_TARGET_CHUNKS blocks fit; B spans that many chunks. Peak stays
        # ≈ mem_budget/SAFETY regardless of the chunk count (see the constant).
        # SAFETY ≈ 3 covers the transient torch.log output + backward buffers.
        SAFETY = 3
        tc = COLLISION_TARGET_CHUNKS
        m_max = max(512, int(math.sqrt(
            mem_budget_bytes / (tc * n_receptors * 4 * SAFETY))))
        collision_chunk_size = m_max
        # B = up to tc chunks (multiple of m_max so chunking wastes no samples).
        b_train = math.ceil(min(stats_cap, tc * m_max) / m_max) * m_max
    elif entropy_type == "kt":
        # The DOUBLE loop over ALL chunk pairs keeps every (m,m,R) block alive for
        # backward ⇒ the RETAINED graph is B²·R·4, independent of chunk size — that
        # sets the batch: b_train = sqrt(mem_budget / (R·4·SAFETY)).  Ceiling log2(B).
        if recompute_backward:
            # Checkpointing the inner loop drops the retained graph to O(B·R), so
            # memory no longer binds — the O(B²·R) COMPUTE does. Cap the work instead
            # (physics_cap below still guards forward memory). Same sqrt(1/R) shape.
            b_cap = max(B_min, int(math.sqrt(KT_COMPUTE_BUDGET / n_receptors)))
        else:
            SAFETY = 3
            b_cap = max(B_min, int(math.sqrt(mem_budget_bytes / (n_receptors * 4 * SAFETY))))
        b_train = min(stats_cap, b_cap)
        # But do NOT run a single b_train-sized chunk: that also materialises a
        # full-size (B,B,R) torch.log transient AND its backward gradient (~3× the
        # retained tensor → OOM). A modest fixed tile keeps those transients at m²·R
        # while the retained B²·R (the real cost, already in b_cap) is unchanged.
        collision_chunk_size = min(b_train, KT_TRAIN_TILE)
    elif entropy_type in ("blocked", "blocked_corrected", "annealed", "blocked_to_corrected"):
        # Blocked Shannon builds (B, 2^block_size) histograms — NOT (B, 2^R).
        # One correlation-aware partition with ceil(R/block_size) blocks is
        # retained for backward (no partition averaging).
        # Annealed shares the same batch since its Rényi term reuses the batch.
        n_blk = (n_receptors + block_size - 1) // block_size
        # SAFETY=4 covers the histogram forward matrix + its retained backward graph.
        # recompute_backward gradient-checkpoints the histogram (recomputed in
        # backward, not retained), so the graph copy drops out → SAFETY≈2, ~2× the
        # batch. NOT applied to 'annealed': it also holds an un-checkpointed collision
        # (R,B,B) block that would then dominate and OOM at the larger B.
        SAFETY = 2 if (recompute_backward and entropy_type != "annealed") else 4
        blocked_mem_cap = max(B_min, mem_budget_bytes // ((1 << block_size) * n_blk * 4 * SAFETY))
        b_train = min(stats_cap, blocked_mem_cap)
    else:
        # proxy / mi_* : O(B·R²) or O(B·R), no exponential or B² memory term.
        b_train = stats_cap

    # Physics cap. The interface-model forward+backward holds *many*
    # (B, n_ligands, R·k_sub) float32 tensors at once: in _compute_energies
    # (ab, dist_sq, exp(·), E_open) and again in p_open (log_terms_open/closed),
    # several retained for backward + their gradients. The retained energy graph
    # also coexists with the collision (R,m,m) chunk during the loss/backward, so the
    # factor must leave headroom for that term too. 16× restores roughly the
    # safety the classic model enjoyed by accident (see below) and clears the
    # OOM that the old 4× hit once the k_sub axis is real (use_interface_model=
    # True). For the classic model the true width is n_genes (no k_sub), so
    # charging R·k_sub here is conservative — exactly that hidden margin.
    # Note s_upper ≤ n_ligands, so n_ligands bounds the 2nd dimension.
    # In cell mode this axis is the receptor POOL, not the number of cells.
    bytes_per_sample = n_ligands * (n_physics_receptors or n_receptors) * k_sub * 4
    physics_cap = max(B_min, mem_budget_bytes // (bytes_per_sample * 16))
    b_train = min(b_train, physics_cap)

    # Two measurement batch sizes (both used only when test_batch_size == "auto"):
    #   test_final   — the absolute maximum useful for the ONE final measurement:
    #                  min(state space 2^R, what memory fits). Evaluation is no_grad
    #                  and the all-pairs estimators (KT) tile internally, so the
    #                  binding tensor is the (EVAL_TILE, B) row buffer →
    #                  B ≤ budget/(EVAL_TILE·4·4); samples above the per-forward limit
    #                  are generated in sub-batches, so this is a pure memory bound.
    #   test_perepoch — a cheaper size (4× the training batch) used for the per-epoch
    #                  convergence curve, so KT's O(B²) cost doesn't dominate every
    #                  logged epoch. Capped at test_final.
    EVAL_TILE = 2048
    eval_mem_cap  = max(B_min, mem_budget_bytes // (EVAL_TILE * 4 * 4))
    test_final    = max(b_train, min(1 << n_receptors, eval_mem_cap))
    # Optional user cap on the final measurement, to bound the O(B²) KT cost at high R
    # (keep it ≥ b_train so the final is never smaller than a training batch).
    if test_max_batch is not None:
        test_final = max(b_train, min(test_final, int(test_max_batch)))
    test_perepoch = min(4 * b_train, test_final)

    return b_train, test_perepoch, test_final, collision_chunk_size

ENV_REGISTRY = {
    "asymmetric": LigandEnvironment,
    "symmetric":  SymmetricLigandEnvironment,
}

def _build_loss(cfg, collision_chunk_size: int = 2048) -> nn.Module:
    """Dispatch on cfg.entropy to construct the appropriate loss module."""
    if cfg.entropy == 'kt_mi':
        return KTMutualInformationLoss(
            collision_chunk_size=collision_chunk_size,
            recompute_backward=cfg.recompute_backward,
            compile_kt=cfg.compile_kt,
        )
    elif cfg.entropy in DiscreteExactLoss._ENTROPY_FNS:
        return DiscreteExactLoss(
            entropy_type=cfg.entropy,
            cov_weight=cfg.cov_weight or 0.0,
            penalty_type=cfg.penalty_type or 'covariance',
            block_size=cfg.block_size,
            n_partitions=cfg.n_partitions,
            collision_chunk_size=collision_chunk_size,
            recompute_backward=cfg.recompute_backward,
            compile_kt=cfg.compile_kt,
        )
    elif cfg.entropy == 'annealed':
        return AnnealedEntropyLoss(
            block_size=cfg.block_size,
            n_partitions=cfg.n_partitions,
            recompute_backward=cfg.recompute_backward,
        )
    elif cfg.entropy == 'blocked_to_corrected':
        return BlockedToCorrectedLoss(
            block_size=cfg.block_size,
            n_partitions=cfg.n_partitions,
            recompute_backward=cfg.recompute_backward,
        )
    elif cfg.entropy == 'mi_ligand':
        return MaximizeMutualInformationLigandLoss(entropy_type='collision')
    elif cfg.entropy == 'mi_conc':
        return MaximizeMutualInformationConcentrationLoss(n_c_bins=cfg.n_c_bins, entropy_type='collision')
    else:
        raise ValueError(f"Unknown entropy: {cfg.entropy!r}. "
                         f"Choose from {DiscreteExactLoss._ENTROPY_FNS} or "
                         f"'kt_mi' / 'annealed' / 'mi_ligand' / 'mi_conc'.")

CONC_REGISTRY = {
    "lognormal": lambda cfg: LogNormalConcentration(
        n_ligands=cfg.n_ligands, init_mean=cfg.conc_mean, init_scale=cfg.conc_std
    ),
    "normal": lambda cfg: NormalConcentration(
        n_ligands=cfg.n_ligands, init_mean=cfg.conc_mean, init_scale=cfg.conc_std
    ),
}

# ==========================================
# SINGLE-RUN MANAGER
# ==========================================

class SimulationRunner:
    """Handles initialisation, training, evaluation, and logging for one run."""

    def __init__(self, config: SingleRunConfig, logger: ExperimentLogger):
        self.config = config
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cell_array = None   # set by _initialize when config.is_cell_mode()
        self.readout    = None

    def _initialize(self, prev_env=None):
        """Builds all components. receptor_indices are always derived from config."""
        receptor_indices = torch.tensor(
            self.config.receptor_indices, dtype=torch.long, device=self.device
        )

        # --- Cell mode: preserve explicit repertoires when rebuilding abundances.
        # The pool order is deterministic (sorted in CellArray), so the columns of W
        # line up with config.receptor_indices, which __post_init__ derived the same way.
        self.cell_array = None
        self.readout = None
        if self.config.is_cell_mode():
            from src.cells import CellArray, CellReadout
            self.cell_array = CellArray(
                (None if self.config.cell_receptors is not None
                 else self.config.cell_gene_sets), self.config.k_sub,
                stoichiometry=self.config.cell_stoichiometry,
                use_interface_model=self.config.use_interface_model,
                repertoires=self.config.cell_receptors,
            ).to(self.device)
            assert torch.equal(self.cell_array.receptor_indices, receptor_indices), (
                "cell pool mismatch: CellArray rebuilt a different receptor pool than "
                "the one stored in config.receptor_indices."
            )
            self.readout = CellReadout(
                self.cell_array.W,
                mode=self.config.cell_readout,
                threshold=(0.5 if self.config.cell_threshold == "auto"
                           else float(self.config.cell_threshold)),
                temperature=self.config.cell_temperature,
                k_sub=self.config.k_sub,
                learnable_threshold=self.config.cell_threshold_learnable,
            ).to(self.device)

        # Always resolve to obtain collision_chunk_size; batch sizes used only when "auto".
        # In cell mode the entropy is over C cells; the physics runs over R_pool.
        n_pool = receptor_indices.shape[0]
        n_r = self.readout.n_cells if self.readout is not None else n_pool
        if torch.cuda.is_available():
            free_mem, _ = torch.cuda.mem_get_info()
            mem_budget = int(free_mem * 0.8)
        else:
            mem_budget = None
        was_auto_test = self.config.test_batch_size == "auto"
        b_auto, test_perepoch, test_final, collision_chunk_size = resolve_batch_sizes(
            n_r, self.config.entropy,
            n_ligands=self.config.n_ligands,
            k_sub=self.config.k_sub,
            mem_budget_bytes=mem_budget,
            block_size=self.config.block_size,
            n_partitions=self.config.n_partitions,
            recompute_backward=self.config.recompute_backward,
            test_max_batch=self.config.test_max_batch,
            n_physics_receptors=n_pool,
        )
        if self.config.batch_size == "auto" or self.config.test_batch_size == "auto":
            if self.config.batch_size == "auto":
                self.config.batch_size = b_auto
            if self.config.test_batch_size == "auto":
                self.config.test_batch_size = test_perepoch      # per-epoch curve (cheap)
            unit = "C" if self.readout is not None else "R"
            pool_note = f"R_pool={n_pool}  " if self.readout is not None else ""
            print(
                f"[auto batch] {unit}={n_r}  "
                + pool_note
                + f"batch_size={self.config.batch_size}  "
                + f"test_batch_size={self.config.test_batch_size}"
                + (f"  gpu_free={mem_budget//(1<<20)} MiB" if mem_budget is not None else "")
            )
        # Final (one-shot) measurement batch. In light mode (per_epoch_measure=False)
        # the closing test uses 4×train (user re-measures from the checkpoint if more
        # samples are wanted). Otherwise: max memory when auto, else the explicit size.
        if not self.config.per_epoch_measure:
            self._final_test_batch = 4 * int(self.config.batch_size)
        else:
            self._final_test_batch = (test_final if was_auto_test
                                      else int(self.config.test_batch_size))
        if self.config.final_test_batch_size is not None:
            self._final_test_batch = int(self.config.final_test_batch_size)
        # config.json was written with "auto" before this resolution; re-save it so
        # the resolved batch_size / test_batch_size (the per-epoch measurement size)
        # are persisted for analysis — e.g. the log2(sample_size) measurement ceilings.
        try:
            self.logger.save_config(self.config)
        except Exception:
            pass
        if collision_chunk_size is not None:
            n_chunks = math.ceil(self.config.batch_size / collision_chunk_size)
            print(
                f"[collision] chunk_size={collision_chunk_size}  "
                f"n_chunks={n_chunks}  batch={self.config.batch_size}"
            )

        if prev_env is not None:
            extra_units = max(0, self.config.n_genes - prev_env.n_genes)
            env = prev_env.clone_with_extra_units(extra_units).to(self.device)
        else:
            conc_model = CONC_REGISTRY[self.config.conc_model_type](self.config)
            env_class  = ENV_REGISTRY[self.config.environment_geometry]
            env = env_class(
                self.config.n_genes,
                self.config.n_families,
                conc_model=conc_model,
                n_ligands=self.config.n_ligands,
                mu_sources=self.config.mu_sources,
                mu_ligands_per_source=self.config.mu_ligands_per_source,
                observation_noise_sigma=self.config.observation_noise_sigma,
                latent_dim=self.config.latent_dim,
                family_spread=self.config.family_spread,
                avg_family_distance=self.config.average_family_distance,
                n_presence_blocks=self.config.n_presence_blocks,
                affinity_kernel=self.config.affinity_kernel,
                kernel_params=self.config.kernel_params,
                distribution_type=self.config.distribution_type,
                use_interface_model=self.config.use_interface_model,
                block_shared_conc_mean=self.config.block_shared_conc_mean,
            ).to(self.device)

        physics = BinaryReceptor(
            self.config.n_genes, self.config.k_sub, temperature=self.config.temperature
        ).to(self.device)

        # Composition binding: evaluate each DISTINCT energy source (gene, or unique
        # +/- face pocket) once and contract to receptors with one matmul.  Removes the
        # (B, L, R, k_sub) intermediate entirely (k_sub x memory) and, in the interface
        # model, the ~n_genes^2/(R*k_sub) pocket redundancy (~400x at cell-mode pool
        # sizes).  Exact — verified equal to the gather path in the self-tests.
        if self.config.use_composition:
            env.bind_receptors(receptor_indices)
            n_src = env.composition.shape[0]
            R_ = receptor_indices.shape[0]
            if self.config.use_interface_model:
                # Genuine compute saving: pocket energies were evaluated per (receptor,
                # ring slot); now once per distinct (+face, -face) gene pair.
                note = (f"pockets {receptor_indices.numel()} slots -> {n_src} distinct"
                        f"  ({receptor_indices.numel()/max(n_src,1):.0f}x fewer evaluations)")
            else:
                # Energies were already per-gene; the saving is the k_sub-wide
                # intermediate that the gather built only to average away.
                note = f"genes={n_src}  (drops the {self.config.k_sub}x-wide gather)"
            print(f"[composition] R={R_}  {note}")
        else:
            env.unbind()
        loss_fn = _build_loss(self.config, collision_chunk_size=collision_chunk_size or 2048).to(self.device)

        # Dampen LR when picking up from a previous env to preserve learned representations
        lr = self.config.lr if prev_env is None else self.config.lr * 0.1
        # In cell mode the readout contributes the shared firing threshold (and nothing
        # else — W is a buffer, T_cell is annealed on a schedule, not optimised).
        trainable = list(env.parameters()) + list(physics.parameters())
        if self.readout is not None:
            trainable += list(self.readout.parameters())
        optimizer = optim.Adam(trainable, lr=lr)

        return env, physics, loss_fn, optimizer, receptor_indices

    def _amp(self):
        """autocast(bfloat16) context for the energy/EC50 matmuls, or a no-op.

        Scoped DELIBERATELY to sampling + physics only, never the loss: the entropy
        estimators do pairwise log/exp arithmetic over the whole batch where bf16's
        ~3 decimal digits would show up directly in the reported bits.  Inside the
        physics, autocast keeps logsumexp in float32 by its own dtype policy, so only
        the einsum/matmul actually drop to bf16 — which is where the memory is.
        """
        from contextlib import nullcontext
        if not self.config.use_amp or self.device != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def _activity(self, physics, energies, concs, receptor_indices, pre_gathered, env=None):
        """(B, R) receptor activity, or (B, C) cell activity when in cell mode.

        Everything downstream (losses, entropy estimators, measurements) consumes a
        (B, N) tensor and is agnostic to whether N counts receptors or cells, so this
        is the ONLY place the two pictures differ.
        """
        comp = getattr(env, "composition", None) if env is not None else None
        if comp is not None:
            pre_gathered = False        # energies are per-source, not per-receptor slot
        if self.readout is None:
            return physics(energies, concs, receptor_indices,
                           pre_gathered=pre_gathered, composition=comp)
        from src.cells import cell_activity
        return cell_activity(
            physics, self.readout, energies, concs, receptor_indices,
            pre_gathered=pre_gathered,
            chunk_size=self.config.cell_pool_chunk,
            recompute=self.config.recompute_backward,
            composition=comp,
        )

    def _eval_stats(self, env, physics, loss_fn, receptor_indices, batch_size, epoch,
                    measurement_fns=None):
        """Evaluation over batch_size total samples, with bounded per-pass memory.

        chunk_size = min(eval_chunk_size or batch_size, batch_size)
          defaults to self.config.batch_size (training batch size), so memory
          per forward pass matches training without any explicit config.

        Soft metrics (Rényi, blocked Shannon, distances …) run on a single
        chunk_size forward pass.

        KT, response conditional entropy, and both counting measurements use the
        same full batch. Counting-only evaluations retain CPU bits, not all soft
        activities, and perform no quadratic KT computation.
        """
        chunk_size = min(self.config.eval_chunk_size or self.config.batch_size, batch_size)
        measurements = self.config.measurement_fns if measurement_fns is None else measurement_fns
        requested = set(measurements)
        batch_metrics = {'entropy_kt', 'entropy_kt_upper', 'conditional_entropy_response',
                         'mutual_information_kt', 'mutual_information_kt_upper',
                         'codeword_entropy', 'mutual_information_counting'}
        want_lower = bool(requested & {'entropy_kt', 'mutual_information_kt'})
        want_upper = bool(requested & {'entropy_kt_upper', 'mutual_information_kt_upper'})
        want_kt = want_lower or want_upper
        want_counting = 'mutual_information_counting' in requested
        want_hard = 'codeword_entropy' in requested

        ri_for_batch = receptor_indices if env.use_interface_model else None
        with torch.no_grad():
            # --- First chunk: soft metrics + soft assignments ---
            # NOTE: no autocast here on purpose. bf16 moves individual activities by up
            # to ~5e-2 and the reported entropy by ~3e-3 bits; that is acceptable as
            # training noise but not in a number we publish. Measurement stays fp32.
            E, concs, masks, concs_dense = env.sample_batch(
                batch_size=chunk_size, receptor_indices=ri_for_batch, return_dense_conc=True)
            activity = self._activity(physics, E, concs, receptor_indices,
                                      env.use_interface_model, env=env)

            stat = {}
            family_labels_cache = None  # computed lazily if any fn requests it
            block_labels_cache  = None  # computed lazily if any fn requests it
            for fn_name in measurements:
                if fn_name in batch_metrics:
                    continue
                fn  = MEASUREMENT_REGISTRY[fn_name]
                sig = inspect.signature(fn)
                kwargs = {}
                if "env"              in sig.parameters: kwargs["env"]              = env
                if "physics"          in sig.parameters: kwargs["physics"]          = physics
                if "receptor_indices" in sig.parameters: kwargs["receptor_indices"] = receptor_indices
                if "loss_fn"          in sig.parameters: kwargs["loss_fn"]          = loss_fn
                if "activity"         in sig.parameters: kwargs["activity"]         = activity
                if "epoch"            in sig.parameters: kwargs["epoch"]            = epoch
                if "concs"            in sig.parameters: kwargs["concs"]            = concs
                if "concs_dense"      in sig.parameters: kwargs["concs_dense"]      = concs_dense
                if "mixture_masks"    in sig.parameters: kwargs["mixture_masks"]    = masks
                if "family_labels"    in sig.parameters:
                    if family_labels_cache is None:
                        import torch.nn.functional as _F
                        one_hot_fam = _F.one_hot(
                            env.ligand_family_assignments.long(), env.n_families
                        ).float()                                         # (L, n_families)
                        family_labels_cache = (masks.float() @ one_hot_fam).bool()  # (B, n_families)
                    kwargs["family_labels"] = family_labels_cache
                if "block_labels"     in sig.parameters:
                    if block_labels_cache is None:
                        import torch.nn.functional as _F
                        one_hot_blk = _F.one_hot(
                            env.presence_block_id.long(), env.n_presence_blocks
                        ).float()                                          # (L, n_presence_blocks)
                        block_labels_cache = (masks.float() @ one_hot_blk).bool()  # (B, n_presence_blocks)
                    kwargs["block_labels"] = block_labels_cache
                result = fn(**kwargs)
                if isinstance(result, dict):
                    stat.update(result)
                else:
                    stat[fn_name] = result

            if requested & batch_metrics:
                acts, hard_codes, sampled_codes = [], [], []
                cond_sum = torch.zeros((), dtype=torch.float64, device=activity.device)
                if want_counting and not hasattr(self, '_response_generator'):
                    # Output sampling must not perturb the world's/training RNG.
                    self._response_generator = torch.Generator(device=activity.device)
                    self._response_generator.manual_seed((torch.initial_seed() + 104729) % (2**63))
                n_have = 0
                while True:
                    n = activity.shape[0]
                    cond_sum += compute_response_conditional_entropy(activity).double() * n
                    if want_kt:
                        acts.append(activity)
                    if want_hard:
                        hard_codes.append((activity > 0.5).cpu())
                    if want_counting:
                        a = activity.clamp(KT_EPS, 1.0 - KT_EPS)
                        sampled_codes.append(torch.bernoulli(
                            a, generator=self._response_generator).bool().cpu())
                    n_have += n
                    if n_have == batch_size:
                        break
                    this = min(chunk_size, batch_size - n_have)
                    E_k, concs_k, _ = env.sample_batch(this, receptor_indices=ri_for_batch)
                    activity = self._activity(physics, E_k, concs_k, receptor_indices,
                                              env.use_interface_model, env=env)
                h_cond = (cond_sum / n_have).item()
                stat['response_evaluation_samples'] = n_have
                if requested & {'conditional_entropy_response', 'mutual_information_kt',
                                'mutual_information_kt_upper', 'mutual_information_counting'}:
                    stat['conditional_entropy_response'] = h_cond
                if want_hard:
                    stat.update(codeword_entropy(torch.cat(hard_codes)))
                if want_counting:
                    stat.update(response_counting_metrics(torch.cat(sampled_codes), h_cond))
                if want_kt:
                    big_soft = loss_fn.compute_soft_assignment(torch.cat(acts))
                    with record_function("prof:eval_kt"):
                        if want_lower:
                            i_lower = compute_kt_entropy(
                                big_soft, chunk_size=2048, return_mi=True,
                                use_compile=getattr(loss_fn, 'compile_kt', False)).item()
                            if 'entropy_kt' in requested:
                                stat['full_array_entropy_kt'] = i_lower + h_cond
                            if 'mutual_information_kt' in requested:
                                stat['mutual_information_kt'] = i_lower
                        if want_upper:
                            i_upper = compute_kt_upper_entropy(
                                big_soft, chunk_size=2048, return_mi=True).item()
                            if 'entropy_kt_upper' in requested:
                                stat['full_array_entropy_kt_upper'] = i_upper + h_cond
                            if 'mutual_information_kt_upper' in requested:
                                stat['mutual_information_kt_upper'] = i_upper

        return stat

    def _train(self, env, physics, loss_fn, optimizer, receptor_indices):
        if self.config.initial_temperature == "auto":
            start_temp = compute_initial_temperature(env, receptor_indices)
        else:
            start_temp = float(self.config.initial_temperature)
        end_temp   = self.config.temperature
        scheduler  = (
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs, eta_min=1e-5)
            if self.config.use_scheduler else None
        )

        # --- Sharpness schedule -------------------------------------------------
        # Receptor-only runs: anneal the receptor temperature over 80% of training,
        # then hold, so the sharp objective (which eval always uses) is optimised to
        # convergence rather than only at the final epoch.
        #
        # Cell mode: two DISTINCT phases, because annealing the receptor and the cell
        # together makes the cell's operating point chase a drive distribution that is
        # still moving underneath it.
        #   Phase 1 (first cell_phase_split of the epochs): the receptor temperature
        #     anneals to its final value; the cell is held SOFT.  Gradients are live
        #     everywhere, so the chemistry can arrange the drive around the threshold.
        #   Phase 2 (the rest): the receptor temperature is HELD; the cell sharpness
        #     anneals down until the cell is effectively deterministic.
        is_cell = self.readout is not None and self.readout.mode == "threshold"
        phase1_epochs = (max(1, int(self.config.cell_phase_split * self.config.epochs))
                         if is_cell else None)
        # phase1_epochs - 1: frac = epoch/anneal_epochs must reach 1.0 on the LAST epoch
        # of phase 1, otherwise the receptor is still annealing one epoch into phase 2
        # and the phases are not clean.  Receptor-only runs keep the original denominator.
        anneal_epochs = (max(1, phase1_epochs - 1) if is_cell
                         else max(1, int(0.8 * self.config.epochs)))

        # --- Cell readout: calibrate theta / T_cell, then anneal T_cell like the
        # receptor temperature.  physics.temperature must be at its START value for
        # the calibration to reflect the drive distribution the first epochs see.
        cell_start_temp = cell_end_temp = None
        recal_every = self.config.cell_recalibrate_every if is_cell else 0
        # Re-pinning the threshold only makes sense when it was pinned to the data in
        # the first place; an explicit float is the user's choice and is left alone.
        recal_theta = recal_every and self.config.cell_threshold == "auto"
        if is_cell:
            from src.cells import calibrate_cell_readout
            physics.temperature = start_temp
            diag = calibrate_cell_readout(
                self.readout, env, physics, receptor_indices,
                chunk_size=self.config.cell_pool_chunk,
                set_threshold=(self.config.cell_threshold == "auto"),
                set_temperature=(self.config.cell_initial_temperature == "auto"),
                n_molecules=self.config.cell_n_molecules,
            )
            # T_cell endpoints are RELATIVE to the calibrated spread of the drive.
            # The drive is a weighted mean of probabilities, so its scale is set by the
            # environment — an absolute T_cell is meaningless against it. Start at 1x
            # the spread (unit-spread sigmoid argument, live gradients everywhere), end
            # at cell_temperature x the spread (default 0.05 => argument spread ~20, so
            # cells are effectively deterministic and H(response|sniff) -> 0, which is
            # what makes the entropy objective a valid proxy for information).
            # Both endpoints are MULTIPLES of the drive spread, so they can be
            # recomputed whenever the spread is re-measured (see the loop).
            start_mult = (1.0 if self.config.cell_initial_temperature == "auto"
                          else float(self.config.cell_initial_temperature))
            end_mult   = self.config.cell_temperature
            # NB the floor is 1e-45 (just off zero), NOT something like 1e-6: the drive
            # can legitimately live at ~1e-30, and an absolute floor above its scale
            # would swamp it and turn every cell back into a fair coin.
            sigma_S = diag["drive_scale"]
            cell_start_temp = max(start_mult * sigma_S, 1e-45)
            cell_end_temp   = max(end_mult   * sigma_S, 1e-45)
            kind = ("learnable" if self.readout.learnable_threshold
                    else (f"re-pinned every {recal_every}" if recal_theta else "fixed"))
            print(f"[cell] C={self.readout.n_cells}  R_pool={receptor_indices.shape[0]}  "
                  f"phase1={phase1_epochs}/{self.config.epochs} epochs  "
                  f"T_cell {cell_start_temp:.4g} → {cell_end_temp:.4g}  "
                  f"theta={diag['theta']:.4g} ({kind})")
            # A shared threshold means cells can start silent or saturated; that is
            # allowed (the chemistry adapts, and the threshold is re-pinned) but a
            # large count means channels are being wasted from epoch 0 — worth seeing.
            if diag["n_silent"] or diag["n_saturated"]:
                print(f"[cell] at init: {diag['n_silent']} silent, "
                      f"{diag['n_saturated']} saturated of {self.readout.n_cells} cells")
            # The plain median sat on a point mass (a sharp receptor drives most cells
            # to exactly zero). theta was stepped above it; the code is sparse, which is
            # honest, but it caps how many bits a cell can carry — worth knowing.
            if diag.get("theta_floored"):
                print(f"[cell] theta hit the physical floor 1/N = "
                      f"{1.0 / self.config.cell_n_molecules:.2e} (one open channel of "
                      f"{self.config.cell_n_molecules:.0e}): the code is sparse, so the "
                      f"population does not fire 50%.")
            if diag.get("on_point_mass"):
                print("[cell] drive has a point mass at the median; theta stepped above "
                      "it (sparse code). See cells.median_threshold.")

        stats = []
        for epoch in range(self.config.epochs):
            optimizer.zero_grad()

            # --- Phase 1: the RECEPTOR sharpens (cell mode: anneal_epochs == phase1) ---
            frac = min(1.0, epoch / anneal_epochs)
            current_temp = (
                end_temp + (start_temp - end_temp) * (1.0 - frac)
                if end_temp < start_temp else end_temp
            )
            if hasattr(physics, "temperature"):
                physics.temperature = current_temp

            # --- Re-pin the threshold to the median of the drive -------------------
            # The drive distribution moves as the chemistry trains, so a threshold
            # measured at epoch 0 goes stale.  Re-pinning keeps it free of any fitted
            # parameter AND keeps the sigmoid's transition band sitting on the densest
            # part of the drive, which is where the phase-2 gradient comes from.
            # Done AFTER physics.temperature is set, so the drive is measured at the
            # receptor sharpness currently in force.
            if recal_theta and epoch > 0 and epoch % recal_every == 0:
                from src.cells import calibrate_cell_readout as _recal
                d = _recal(self.readout, env, physics, receptor_indices,
                           chunk_size=self.config.cell_pool_chunk,
                           set_threshold=True, set_temperature=False,
                           n_molecules=self.config.cell_n_molecules)
                # Refresh the spread too: both sharpness endpoints are multiples of it,
                # so phase 2 targets the live distribution rather than epoch 0's.
                sigma_S = d["drive_scale"]
                cell_start_temp = max(start_mult * sigma_S, 1e-45)
                cell_end_temp   = max(end_mult   * sigma_S, 1e-45)

            # --- Phase 2: the CELL sharpens; held soft for the whole of phase 1 ------
            current_cell_temp = None
            if cell_start_temp is not None:
                phase2_epochs = self.config.epochs - phase1_epochs
                cell_frac = (0.0 if epoch < phase1_epochs else
                             (1.0 if phase2_epochs <= 1 else
                              min(1.0, (epoch - phase1_epochs)
                                  / (phase2_epochs - 1))))
                current_cell_temp = (
                    cell_end_temp + (cell_start_temp - cell_end_temp) * (1.0 - cell_frac)
                    if cell_end_temp < cell_start_temp else cell_end_temp
                )
                self.readout.temperature = current_cell_temp

            ri_for_batch = receptor_indices if env.use_interface_model else None
            # [profiling] record_function labels group ops in the torch.profiler trace /
            # table (physics forward vs loss vs backward). Inert (~free) when no profiler
            # is attached — see tasks/profiling/scripts/profile_run.py.
            with record_function("prof:sample+physics_fwd"), self._amp():
                energies, concs, masks = env.sample_batch(self.config.batch_size, receptor_indices=ri_for_batch)
                activity = self._activity(physics, energies, concs, receptor_indices,
                                          env.use_interface_model, env=env).float()

            with record_function("prof:loss_fwd"):
                if isinstance(loss_fn, MaximizeMutualInformationLigandLoss):
                    loss = loss_fn(activity, mixture_masks=masks)
                elif isinstance(loss_fn, MaximizeMutualInformationConcentrationLoss):
                    loss = loss_fn(activity, concs=concs)
                elif isinstance(loss_fn, (AnnealedEntropyLoss, BlockedToCorrectedLoss)):
                    loss = loss_fn(activity, epoch, self.config.epochs)
                else:
                    loss = loss_fn(activity)

            with record_function("prof:backward"):
                loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            if epoch % max(1, self.config.epochs // 100) == 0:
                if self.config.per_epoch_measure:
                    if hasattr(physics, "temperature"):
                        physics.temperature = end_temp
                    if current_cell_temp is not None:
                        self.readout.temperature = cell_end_temp
                    stat = self._eval_stats(
                        env, physics, loss_fn, receptor_indices,
                        self.config.test_batch_size, epoch
                    )
                    if hasattr(physics, "temperature"):
                        physics.temperature = current_temp
                    if current_cell_temp is not None:
                        self.readout.temperature = current_cell_temp
                else:
                    # Free: reuse the training objective already computed for the
                    # gradient step — no extra sampling / eval. For entropy-maximising
                    # losses (kt, collision, …) the native entropy is -loss.
                    metric = ('train_mutual_information' if isinstance(loss_fn, KTMutualInformationLoss)
                              else 'train_entropy')
                    stat = {"loss": float(loss.item()), metric: float(-loss.item())}
                if isinstance(loss_fn, KTMutualInformationLoss):
                    stat['train_mutual_information'] = float(-loss.item())
                stat["lr"] = optimizer.param_groups[0]["lr"]
                if self.readout is not None and self.readout.mode == "threshold":
                    # Track where the shared threshold drifts: it starts at the
                    # calibrated median (population fires ~50%) and Adam is free to
                    # move it toward a sparser or denser operating point.
                    stat["cell_theta"] = self.readout.theta.detach().item()
                stats.append(stat)

        # Save and test the same endpoint used for periodic evaluation, including
        # very short runs whose schedule has no room for a complete second phase.
        physics.temperature = end_temp
        if cell_end_temp is not None:
            self.readout.temperature = cell_end_temp

        return {key: [s[key] for s in stats] for key in stats[0]} if stats else {}

    def _test(self, env, physics, loss_fn, receptor_indices, n_samples: int, test_epochs: int = 10):
        stats = [
            self._eval_stats(env, physics, loss_fn, receptor_indices, n_samples, i,
                             measurement_fns=self.config.final_measurement_fns)
            for i in range(test_epochs)
        ]
        return {key: [s[key] for s in stats] for key in stats[0]} if stats else {}

    def run(self, prev_env=None):
        """Executes the full training → checkpoint → test pipeline."""
        env, physics, loss_fn, optimizer, receptor_indices = self._initialize(prev_env)

        train_stats = self._train(env, physics, loss_fn, optimizer, receptor_indices)

        if train_stats:
            n_logged = len(next(iter(train_stats.values())))
            for i in range(n_logged):
                self.logger.save_stats(i, {k: train_stats[k][i] for k in train_stats})

        self.logger.save_checkpoint(self.config.epochs, env, physics, receptor_indices,
                                    is_best=True, readout=self.readout)

        test_results = self._test(env, physics, loss_fn, receptor_indices,
                                   n_samples=self._final_test_batch)

        import json as _json
        import os as _os
        with open(_os.path.join(self.logger.run_dir, "test_results.json"), "w") as f:
            from src.IO import CustomJSONEncoder
            _json.dump(test_results, f, indent=4, cls=CustomJSONEncoder)

        return env


# ==========================================
# SWEEP MANAGER
# ==========================================

def _sweep_total_steps(config: RunConfig) -> int:
    """Counts total SimulationRunner.run() calls for the tqdm bar."""
    axes = config._axes()
    return len(next(iter(axes.values()))) if axes else 1


class SweepRunner:
    """Consumes a RunConfig and executes all generated trajectories."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.master_logger = SweepLogger(config)

    def execute(self):
        """Execute the sweep and leave an objective execution-state marker."""
        try:
            result = self._execute()
        except KeyboardInterrupt:
            self.master_logger.set_execution_state("interrupted")
            raise
        except BaseException:
            self.master_logger.set_execution_state("failed")
            raise
        self.master_logger.set_execution_state("complete")
        return result

    def _execute(self):
        total = _sweep_total_steps(self.config)
        print(f"\nInitiating sweep: {self.master_logger.sweep_root}")
        print(f"Total runs: {total}\n")

        axes = self.config._axes()

        with tqdm(total=total, desc="Sweep Progress", dynamic_ncols=True) as pbar:
            for trajectory in self.config.generate_trajectories():
                prev_env = None  # trained env from the immediately preceding step
                prev_cfg = None  # SingleRunConfig of the preceding step

                for run_cfg in trajectory:
                    # --- Build human-readable tqdm label (scalar axes only) ---
                    label_parts = [
                        f"{k}: {getattr(run_cfg, k)}"
                        for k in sorted(axes.keys())
                        if axes[k] and not isinstance(axes[k][0], (list, tuple))
                    ]
                    tqdm.write(f"--- {' | '.join(label_parts)} ---")

                    # --- warm-start: chain only when n_genes strictly increases ---
                    # A decrease means a new env group is starting; reset there.
                    if (prev_cfg is not None
                            and self.config.warm_start
                            and run_cfg.n_genes > prev_cfg.n_genes):
                        warm_env = prev_env
                    else:
                        warm_env = None

                    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    node_logger   = self.master_logger.get_run_logger(run_cfg, run_timestamp)
                    runner        = SimulationRunner(config=run_cfg, logger=node_logger)
                    prev_env      = runner.run(prev_env=warm_env)
                    prev_cfg      = run_cfg

                    # Index the completed run — best-effort, never aborts sweep
                    try:
                        import os as _os
                        from src.db import add_run as _db_add_run
                        _db_add_run(
                            node_logger.run_dir,
                            _os.path.join(self.config.base_folder, "runs.db"),
                        )
                    except Exception:
                        pass

                    # Release this run's GPU memory BEFORE the next run is sized.
                    # The next SimulationRunner reads torch.cuda.mem_get_info() to pick
                    # its batch; PyTorch's caching allocator otherwise holds this run's
                    # freed tensors as a process-level cache that mem_get_info counts as
                    # NOT free — so without this the reported free memory ratchets down
                    # across the sweep and later runs get progressively tiny batches.
                    if not self.config.warm_start:
                        prev_env = None            # not chaining → drop the trained env
                    del runner
                    import gc as _gc
                    _gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    pbar.update(1)
