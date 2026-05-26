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
from functools import reduce
from operator import mul
from tqdm import tqdm

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

def resolve_batch_sizes(n_receptors: int, entropy_type: str = "shannon") -> tuple:
    """Returns (batch_size, test_batch_size) appropriate for the array size.

    Shannon / exact: B_train = 2^R — one sample per histogram bin for good coverage.
    Memory cap: soft_assign tensor is (B, 2^R) float32; budget B×2^R ≤ 2^35 floats
    (~128 GiB), giving ~10^6 max samples at R=15 on an A100.

    Rényi-2: cost is O(B²·R) not O(B·2^R); keep B ~ 16·√(2^R) instead.

    test_batch_size = 4 × batch_size.
    """
    B_min = 512
    if entropy_type != "renyi":
        b_train = max(B_min, 1 << n_receptors)
        # B × 2^R ≤ 2^35  →  B ≤ 2^(35-R)
        mem_cap = max(B_min, 1 << max(0, 35 - n_receptors))
        b_train = min(b_train, mem_cap)
    else:
        b_train = max(B_min, 16 * int(2 ** (n_receptors / 2)))
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
        )
    elif cfg.entropy == 'mi_ligand':
        return MaximizeMutualInformationLigandLoss(entropy_type='renyi')
    elif cfg.entropy == 'mi_conc':
        return MaximizeMutualInformationConcentrationLoss(n_c_bins=cfg.n_c_bins, entropy_type='renyi')
    else:
        raise ValueError(f"Unknown entropy: {cfg.entropy!r}. "
                         f"Choose from {DiscreteExactLoss._ENTROPY_FNS} or 'mi_ligand' / 'mi_conc'.")

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
            b_train, b_eval = resolve_batch_sizes(n_r, self.config.entropy)
            if self.config.batch_size == "auto":
                self.config.batch_size = b_train
            if self.config.test_batch_size == "auto":
                self.config.test_batch_size = b_eval

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
                p_presence=self.config.p_presence,
                observation_noise_sigma=self.config.observation_noise_sigma,
                latent_dim=self.config.latent_dim,
                family_spread=self.config.family_spread,
                avg_family_distance=self.config.average_family_distance,
                n_presence_blocks=self.config.n_presence_blocks,
                rho_block=self.config.rho_block,
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
        start_temp = compute_initial_temperature(env, receptor_indices)
        print(start_temp)
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
    sweep = dict(config._sweep_axes())
    warm_axis = config.warm_start_axis
    if warm_axis and warm_axis in sweep:
        warm_steps = len(sweep.pop(warm_axis))
    else:
        warm_steps = 1
    ind_size = reduce(mul, (len(v) for v in sweep.values()), 1)
    return ind_size * config.n_samples * warm_steps


class SweepRunner:
    """Consumes a RunConfig and executes all generated trajectories."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.master_logger = SweepLogger(config)

    def execute(self):
        total = _sweep_total_steps(self.config)
        print(f"\nInitiating sweep: {self.master_logger.sweep_root}")
        print(f"Total runs: {total}\n")

        # Guard: gene-growth and receptor fan-out warm-starts are mutually exclusive.
        _warm_spec = self.config.warm_start_axis
        _warm_axes = (
            [_warm_spec] if isinstance(_warm_spec, str)
            else (_warm_spec or [])
        )
        if 'n_genes' in _warm_axes and 'n_receptors' in _warm_axes:
            raise ValueError(
                "warm_start_axis cannot include both 'n_genes' and 'n_receptors'. "
                "Gene-growth warm-starting and receptor fan-out warm-starting are "
                "mutually exclusive — use one or the other."
            )

        with tqdm(total=total, desc="Sweep Progress", dynamic_ncols=True) as pbar:
            for meta, trajectory in self.config.generate_trajectories():
                prev_env   = None  # trained env from the immediately preceding step
                prev_cfg   = None  # SingleRunConfig of the preceding step
                square_env = None  # cached env from the step where n_genes == n_receptors

                for run_cfg in trajectory:
                    # --- Build human-readable tqdm label ---
                    meta_str  = " | ".join(f"{k}: {v}" for k, v in sorted(meta.items()))
                    warm_axis = self.config.warm_start_axis
                    if isinstance(warm_axis, str) and isinstance(getattr(self.config, warm_axis, None), list):
                        warm_str = f" | {warm_axis}: {getattr(run_cfg, warm_axis)}"
                    else:
                        warm_str = ""
                    tqdm.write(f"--- {meta_str}{warm_str} ---")

                    # --- 3-way warm-start heuristic ---
                    if prev_cfg is None:
                        # First step in the trajectory: always cold start.
                        warm_env = None
                    elif prev_cfg.n_genes != run_cfg.n_genes:
                        # Case 1: n_genes grew → chain warm-start from the previous step.
                        warm_env = prev_env
                    elif square_env is not None:
                        # Case 2: n_genes unchanged, receptors expanded → fan-out from
                        # the cached "square" baseline (the step where n_genes == n_receptors).
                        warm_env = square_env
                    else:
                        # Case 3: no applicable warm start → cold start.
                        import warnings as _warnings
                        _warnings.warn(
                            f"No warm-start env available for "
                            f"(n_genes={run_cfg.n_genes}, n_receptors={run_cfg.n_receptors}): "
                            f"n_genes did not change and no square baseline "
                            f"(n_genes == n_receptors) has been run yet in this trajectory. "
                            f"Starting cold.",
                            UserWarning, stacklevel=2,
                        )
                        warm_env = None

                    node_logger = self.master_logger.get_run_logger(meta, run_cfg)
                    runner      = SimulationRunner(config=run_cfg, logger=node_logger)
                    prev_env    = runner.run(prev_env=warm_env)

                    # Cache env as the square baseline when n_genes == effective n_receptors.
                    _effective_nr = (
                        run_cfg.n_receptors if run_cfg.n_receptors is not None
                        else run_cfg.n_genes  # homomer default: one receptor per gene
                    )
                    if _effective_nr == run_cfg.n_genes:
                        square_env = prev_env

                    prev_cfg = run_cfg
                    pbar.update(1)
