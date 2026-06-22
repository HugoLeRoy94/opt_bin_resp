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
import torch
import torch.nn as nn
import torch.optim as optim
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
    codeword_entropy,
    miller_madow_entropy,
    mean_receptor_distance,
    conditional_entropy_ligand,
    mutual_information_ligand,
    conditional_entropy_concentration,
    mutual_information_concentration,
    conditional_entropy_family,
    mutual_information_family,
    conditional_entropy_block,
    mutual_information_block,
    receptor_distances,
    rank_ordered_distances,
    mean_specialization_index,
    receptor_conditioned_entropy
)

from src.bin_loss import DiscreteExactLoss
from src.annealed_loss import AnnealedEntropyLoss, BlockedToCorrectedLoss
from src.family_mi_loss import MaximizeMutualInformationLigandLoss
from src.concentration_mi_loss import MaximizeMutualInformationConcentrationLoss

# ==========================================
# REGISTRIES
# ==========================================

MEASUREMENT_REGISTRY = {
    "full_array_entropy":                full_array_entropy,
    "codeword_entropy":                  codeword_entropy,
    "mean_receptor_distance":            mean_receptor_distance,
    "conditional_entropy_ligand":        conditional_entropy_ligand,
    "mutual_information_ligand":         mutual_information_ligand,
    "conditional_entropy_concentration": conditional_entropy_concentration,
    "mutual_information_concentration":  mutual_information_concentration,
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

def resolve_batch_sizes(
    n_receptors: int,
    entropy_type: str = "shannon",
    n_ligands: int = 1,
    k_sub: int = 1,
    mem_budget_bytes: Optional[int] = None,
    block_size: int = 15,
    n_partitions: int = 4,
) -> tuple:
    """Returns (batch_size, test_batch_size) appropriate for the array size.

    The estimator dictates the dominant entropy-side tensor, so each gets its
    own cap (memory model in parentheses):
      shannon   : B = 2^R coverage, capped by (B, 2^R) float32.        — only R<~15
      collision : B = 16·√(2^R), capped by the (B, B) collision matrix.
      blocked   : capped by ceil(R/block_size)·n_partitions histograms of
                shape (B, 2^block_size) — independent of R, so B stays large.
      proxy / mi_* : O(B·R²)/O(B·R), no exponential or B² term; physics-bound.

    Physics bottleneck cap (all estimators): the interface-model forward+backward
    holds many (B, n_ligands, R·k_sub) float32 tensors at once (see below); a 16×
    safety factor is applied. mem_budget_bytes should be the free GPU memory at
    _initialize time; defaults to 8 GiB when CUDA is unavailable.

    block_size / n_partitions must match the DiscreteExactLoss config (defaults
    15 / 4) for the blocked cap to be correct.

    test_batch_size = 4 × batch_size.
    """
    B_min = 512
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
        b_train = stats_cap
        # Collision materialises a (B, B) matrix: B²·4 bytes.
        collision_cap = max(B_min, int((mem_budget_bytes / (4 * 4)) ** 0.5))
        b_train = min(b_train, collision_cap)
    elif entropy_type in ("blocked", "blocked_corrected", "annealed", "blocked_to_corrected"):
        # Blocked Shannon builds (B, 2^block_size) histograms — NOT (B, 2^R).
        # One correlation-aware partition with ceil(R/block_size) blocks is
        # retained for backward (no partition averaging).
        # Annealed shares the same batch since its Rényi term reuses the batch.
        n_blk = (n_receptors + block_size - 1) // block_size
        blocked_mem_cap = max(B_min, mem_budget_bytes // ((1 << block_size) * n_blk * 4 * 4))
        b_train = min(stats_cap, blocked_mem_cap)
    else:
        # proxy / mi_* : O(B·R²) or O(B·R), no exponential or B² memory term.
        b_train = stats_cap

    # Physics cap. The interface-model forward+backward holds *many*
    # (B, n_ligands, R·k_sub) float32 tensors at once: in _compute_energies
    # (ab, dist_sq, exp(·), E_open) and again in p_open (log_terms_open/closed),
    # several retained for backward + their gradients. The retained energy graph
    # also coexists with the collision (B,B) matrix during the loss/backward, so the
    # factor must leave headroom for that term too. 16× restores roughly the
    # safety the classic model enjoyed by accident (see below) and clears the
    # OOM that the old 4× hit once the k_sub axis is real (use_interface_model=
    # True). For the classic model the true width is n_genes (no k_sub), so
    # charging R·k_sub here is conservative — exactly that hidden margin.
    # Note s_upper ≤ n_ligands, so n_ligands bounds the 2nd dimension.
    bytes_per_sample = n_ligands * n_receptors * k_sub * 4
    physics_cap = max(B_min, mem_budget_bytes // (bytes_per_sample * 16))
    b_train = min(b_train, physics_cap)

    return b_train, 4 * b_train

ENV_REGISTRY = {
    "asymmetric": LigandEnvironment,
    "symmetric":  SymmetricLigandEnvironment,
}

def _build_loss(cfg) -> nn.Module:
    """Dispatch on cfg.entropy to construct the appropriate loss module."""
    if cfg.entropy in DiscreteExactLoss._ENTROPY_FNS:
        return DiscreteExactLoss(
            entropy_type=cfg.entropy,
            cov_weight=cfg.cov_weight or 0.0,
            penalty_type=cfg.penalty_type or 'covariance',
            block_size=cfg.block_size,
            n_partitions=cfg.n_partitions,
        )
    elif cfg.entropy == 'annealed':
        return AnnealedEntropyLoss(
            block_size=cfg.block_size,
            n_partitions=cfg.n_partitions,
        )
    elif cfg.entropy == 'blocked_to_corrected':
        return BlockedToCorrectedLoss(
            block_size=cfg.block_size,
            n_partitions=cfg.n_partitions,
        )
    elif cfg.entropy == 'mi_ligand':
        return MaximizeMutualInformationLigandLoss(entropy_type='collision')
    elif cfg.entropy == 'mi_conc':
        return MaximizeMutualInformationConcentrationLoss(n_c_bins=cfg.n_c_bins, entropy_type='collision')
    else:
        raise ValueError(f"Unknown entropy: {cfg.entropy!r}. "
                         f"Choose from {DiscreteExactLoss._ENTROPY_FNS} or "
                         f"'annealed' / 'mi_ligand' / 'mi_conc'.")

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

    def _initialize(self, prev_env=None):
        """Builds all components. receptor_indices are always derived from config."""
        receptor_indices = torch.tensor(
            self.config.receptor_indices, dtype=torch.long, device=self.device
        )

        # Resolve "auto" batch sizes now that receptor_indices is known.
        if self.config.batch_size == "auto" or self.config.test_batch_size == "auto":
            n_r = receptor_indices.shape[0]
            if torch.cuda.is_available():
                free_mem, _ = torch.cuda.mem_get_info()
                mem_budget = int(free_mem * 0.8)
            else:
                mem_budget = None
            b_train, b_eval = resolve_batch_sizes(
                n_r, self.config.entropy,
                n_ligands=self.config.n_ligands,
                k_sub=self.config.k_sub,
                mem_budget_bytes=mem_budget,
                block_size=self.config.block_size,
                n_partitions=self.config.n_partitions,
            )
            if self.config.batch_size == "auto":
                self.config.batch_size = b_train
            if self.config.test_batch_size == "auto":
                self.config.test_batch_size = b_eval
            print(
                f"[auto batch] R={n_r}  "
                f"batch_size={self.config.batch_size}  "
                f"test_batch_size={self.config.test_batch_size}"
                + (f"  gpu_free={mem_budget//(1<<20)} MiB" if mem_budget is not None else "")
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
        loss_fn = _build_loss(self.config).to(self.device)

        # Dampen LR when picking up from a previous env to preserve learned representations
        lr = self.config.lr if prev_env is None else self.config.lr * 0.1
        optimizer = optim.Adam(
            list(env.parameters()) + list(physics.parameters()), lr=lr
        )

        return env, physics, loss_fn, optimizer, receptor_indices

    def _eval_stats(self, env, physics, loss_fn, receptor_indices, batch_size, epoch):
        """Evaluation over batch_size total samples, with bounded per-pass memory.

        chunk_size = min(eval_chunk_size or batch_size, batch_size)
          defaults to self.config.batch_size (training batch size), so memory
          per forward pass matches training without any explicit config.

        Soft metrics (Rényi, blocked Shannon, distances …) run on a single
        chunk_size forward pass.

        Hard-codeword metrics (plug-in, Miller-Madow, K_hat, K_frac) are
        accumulated over ceil(batch_size / chunk_size) forward passes so the
        full batch_size budget drives bias estimation.  When batch_size ==
        chunk_size the single-pass path in full_array_entropy covers both.
        """
        chunk_size = min(self.config.eval_chunk_size or self.config.batch_size, batch_size)
        do_mm_accumulation = (
            'codeword_entropy' in self.config.measurement_fns
            and batch_size > chunk_size
        )

        ri_for_batch = receptor_indices if env.use_interface_model else None
        with torch.no_grad():
            # --- First chunk: soft metrics + soft assignments ---
            E, concs, masks = env.sample_batch(batch_size=chunk_size, receptor_indices=ri_for_batch)
            activity = physics(E, concs, receptor_indices, pre_gathered=env.use_interface_model)

            stat = {}
            family_labels_cache = None  # computed lazily if any fn requests it
            block_labels_cache  = None  # computed lazily if any fn requests it
            for fn_name in self.config.measurement_fns:
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

            if do_mm_accumulation:
                # Accumulate hard codewords on CPU over the full budget.
                # (B_eval, R) bool tensor — 2.5 MB for B=2^20, R=20.
                all_codes = [(activity > 0.5).cpu()]
                n_so_far = chunk_size
                while n_so_far < batch_size:
                    this_chunk = min(chunk_size, batch_size - n_so_far)
                    E_c, concs_c, _ = env.sample_batch(batch_size=this_chunk, receptor_indices=ri_for_batch)
                    act_c = physics(E_c, concs_c, receptor_indices, pre_gathered=env.use_interface_model)
                    all_codes.append((act_c > 0.5).cpu())
                    n_so_far += this_chunk

                all_codes_cat = torch.cat(all_codes, dim=0)  # (batch_size, R) on CPU
                H_plugin, H_MM, K_hat, log2_B, K_frac = miller_madow_entropy(all_codes_cat)
                stat.update({
                    'codeword_entropy_plugin': H_plugin,
                    'codeword_entropy_mm':     H_MM,
                    'codeword_entropy_log2B':  log2_B,
                    'codeword_entropy_K_hat':  float(K_hat),
                    'codeword_entropy_K_frac': K_frac,
                })

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

        stats = []
        for epoch in range(self.config.epochs):
            optimizer.zero_grad()

            current_temp = (
                end_temp + (start_temp - end_temp) * (1.0 - epoch / self.config.epochs)
                if end_temp < start_temp else end_temp
            )
            if hasattr(physics, "temperature"):
                physics.temperature = current_temp

            ri_for_batch = receptor_indices if env.use_interface_model else None
            energies, concs, masks = env.sample_batch(self.config.batch_size, receptor_indices=ri_for_batch)
            activity = physics(energies, concs, receptor_indices, pre_gathered=env.use_interface_model)

            if isinstance(loss_fn, MaximizeMutualInformationLigandLoss):
                loss = loss_fn(activity, mixture_masks=masks)
            elif isinstance(loss_fn, MaximizeMutualInformationConcentrationLoss):
                loss = loss_fn(activity, concs=concs)
            elif isinstance(loss_fn, (AnnealedEntropyLoss, BlockedToCorrectedLoss)):
                loss = loss_fn(activity, epoch, self.config.epochs)
            else:
                loss = loss_fn(activity)

            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            if epoch % max(1, self.config.epochs // 100) == 0:
                if hasattr(physics, "temperature"):
                    physics.temperature = end_temp
                stat = self._eval_stats(
                    env, physics, loss_fn, receptor_indices,
                    self.config.test_batch_size, epoch
                )
                stat["lr"] = optimizer.param_groups[0]["lr"]
                if hasattr(physics, "temperature"):
                    physics.temperature = current_temp
                stats.append(stat)

        return {key: [s[key] for s in stats] for key in stats[0]} if stats else {}

    def _test(self, env, physics, loss_fn, receptor_indices, n_samples: int, test_epochs: int = 10):
        stats = [
            self._eval_stats(env, physics, loss_fn, receptor_indices, n_samples, i)
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

        self.logger.save_checkpoint(self.config.epochs, env, physics, receptor_indices, is_best=True)

        test_results = self._test(env, physics, loss_fn, receptor_indices,
                                   n_samples=self.config.test_batch_size)

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

                    pbar.update(1)
