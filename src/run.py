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
    "full_array_entropy":              full_array_entropy,
    "codeword_entropy":                codeword_entropy,
    "mean_receptor_distance":          mean_receptor_distance,
    "conditional_entropy_ligand":      conditional_entropy_ligand,
    "mutual_information_ligand":       mutual_information_ligand,
    "conditional_entropy_concentration": conditional_entropy_concentration,
    "mutual_information_concentration": mutual_information_concentration,
    "receptor_distances":              receptor_distances,
    "rank_ordered_distances":          rank_ordered_distances,
    "mean_specialization_index":       mean_specialization_index,
    "receptor_conditioned_entropy":    receptor_conditioned_entropy,
}

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
                affinity_kernel=self.config.affinity_kernel,
                kernel_params=self.config.kernel_params,
                distribution_type=self.config.distribution_type,
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

        with torch.no_grad():
            # --- First chunk: soft metrics + soft assignments ---
            E, concs, masks = env.sample_batch(batch_size=chunk_size)
            activity = physics(E, concs, receptor_indices)

            stat = {}
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
                    E_c, concs_c, _ = env.sample_batch(batch_size=this_chunk)
                    act_c = physics(E_c, concs_c, receptor_indices)
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

            energies, concs, masks = env.sample_batch(self.config.batch_size)
            activity = physics(energies, concs, receptor_indices)

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

        with tqdm(total=total, desc="Sweep Progress", dynamic_ncols=True) as pbar:
            for meta, trajectory in self.config.generate_trajectories():
                prev_env = None
                for run_cfg in trajectory:
                    # Build a human-readable label for the progress bar
                    meta_str = " | ".join(f"{k}: {v}" for k, v in sorted(meta.items()))
                    warm_axis = self.config.warm_start_axis
                    if warm_axis and isinstance(getattr(self.config, warm_axis, None), list):
                        warm_str = f" | {warm_axis}: {getattr(run_cfg, warm_axis)}"
                    else:
                        warm_str = ""
                    tqdm.write(f"--- {meta_str}{warm_str} ---")

                    node_logger = self.master_logger.get_run_logger(meta, run_cfg)
                    runner      = SimulationRunner(config=run_cfg, logger=node_logger)
                    prev_env    = runner.run(prev_env=prev_env)
                    pbar.update(1)
