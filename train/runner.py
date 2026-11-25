"""One optimization: build the world, train the array, measure it once.

The loop is short. Everything around it is schedule and measurement.

Sharpness schedule. The receptor activation starts soft (temperature
initial_temperature) and anneals to `temperature` over the first 80% of epochs,
then holds. Starting sharp gives vanishing gradients, and ending soft would mean
reporting a number for a model that is not the one evaluation uses, so the final
measurement is always taken at the sharp endpoint.
"""
import math

import torch
import torch.optim as optim

from estimate.entropy import (compute_kt_entropy, compute_kt_upper_entropy,
                              compute_response_conditional_entropy)
from estimate.grouping import CellGrouping, GroupedResponseCounter
from estimate.losses import (ExactMutualInformationLoss,
                             GroupedCellMutualInformationLoss,
                             GroupedKTMutualInformationLoss, KTMutualInformationLoss)
from estimate.measurements import MEASUREMENTS, codeword_entropy
from model.cells import CellArray, CellReadout, cell_activity
from model.environment import LigandEnvironment, LogNormalConcentration
from model.receptors import BinaryReceptor, compute_initial_temperature
from train.moves import FamilyMoves, place_near_families

KT_TRAIN_TILE = 4096
EVAL_TILE = 2048

def build_loss(config, grouping=None):
    """The objective picked by (config.mi, config.grouped).

    Two independent axes, so four combinations and no special names:

                     grouped=False                grouped=True
        shannon      ExactMutualInformationLoss   GroupedCellMutualInformationLoss
        kt           KTMutualInformationLoss      GroupedKTMutualInformationLoss
    """
    tile = min(int(config.batch_size), KT_TRAIN_TILE)
    common = dict(collision_chunk_size=tile,
                  recompute_backward=config.recompute_backward,
                  compile_kt=config.compile_kt)
    if config.mi == "shannon":
        return (GroupedCellMutualInformationLoss(grouping, config.cell_grouped_max_states)
                if config.grouped else ExactMutualInformationLoss(**common))
    return (GroupedKTMutualInformationLoss(grouping, **common)
            if config.grouped else KTMutualInformationLoss(**common))


class Runner:
    """Trains one Config and writes its results through `logger`."""

    def __init__(self, config, logger, device=None):
        self.config = config
        self.logger = logger
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.readout = None
        self.grouping = None
        self.grouped_estimator = None
        self._response_generator = None
        # Set by build(). A test script loads a checkpoint into these and calls
        # evaluate(), so measuring never re-implements construction.
        self.env = self.physics = self.loss_fn = self.receptor_indices = None

    # -- construction ------------------------------------------------------

    def build(self, verbose=True):
        """Construct the world, the array and the objective. Sets the attributes."""
        cfg = self.config
        receptor_indices = torch.tensor(cfg.receptor_indices, dtype=torch.long,
                                        device=self.device)

        if cfg.is_cell_mode():
            # The pool order is deterministic, so CellArray rebuilds exactly the
            # pool that Config.__post_init__ stored in receptor_indices.
            cells = CellArray(
                (None if cfg.cell_receptors is not None else cfg.cell_gene_sets),
                cfg.k_sub, stoichiometry=cfg.cell_stoichiometry,
                repertoires=cfg.cell_receptors).to(self.device)
            self.readout = CellReadout(cells.W, k_sub=cfg.k_sub).to(self.device)
            self.grouping = CellGrouping(cells.W).to(self.device)
            if verbose:
                print(f"[cells] {self.grouping.n_cells} cells -> "
                      f"{self.grouping.n_groups} groups, "
                      f"{self.grouping.n_states} count states")

        env = LigandEnvironment(
            cfg.n_genes, cfg.n_families,
            conc_model=LogNormalConcentration(cfg.n_ligands, cfg.conc_mean, cfg.conc_std),
            n_ligands=cfg.n_ligands, mu_sources=cfg.mu_sources,
            mu_ligands_per_source=cfg.mu_ligands_per_source,
            observation_noise_sigma=cfg.observation_noise_sigma,
            latent_dim=cfg.latent_dim, family_spread=cfg.family_spread,
            avg_family_distance=cfg.average_family_distance,
            n_presence_blocks=cfg.n_presence_blocks, kernel_params=cfg.kernel_params,
            distribution_type=cfg.distribution_type, max_energy=cfg.max_energy,
            block_shared_conc_mean=cfg.block_shared_conc_mean,
            shared_unit_energies=cfg.shared_unit_energies,
            family_weights=cfg.family_weights).to(self.device)

        physics = BinaryReceptor(cfg.n_genes, cfg.k_sub,
                                 temperature=cfg.temperature).to(self.device)

        # Evaluate each DISTINCT pocket once and contract to receptors with one
        # matmul. Removes the (B, L, R, k_sub) intermediate entirely.
        env.bind_receptors(receptor_indices)
        n_src = env.composition.shape[0]
        if verbose:
            print(f"[composition] R={receptor_indices.shape[0]}  "
                  f"{receptor_indices.numel()} pocket slots -> {n_src} distinct "
                  f"({receptor_indices.numel() / max(n_src, 1):.0f}x fewer)")

        loss_fn = build_loss(cfg, self.grouping).to(self.device)
        if isinstance(loss_fn, GroupedCellMutualInformationLoss):
            self.grouped_estimator = loss_fn

        optimizer = optim.Adam(list(env.parameters()) + list(physics.parameters()),
                               lr=cfg.lr)
        self.env, self.physics = env, physics
        self.loss_fn, self.receptor_indices = loss_fn, receptor_indices
        return env, physics, loss_fn, optimizer, receptor_indices

    def _grouped_estimator_for(self, _):
        """The exact count estimator, built once, shared with the training loss."""
        if self.grouped_estimator is None:
            self.grouped_estimator = GroupedCellMutualInformationLoss(
                self.grouping, self.config.cell_grouped_max_states).to(self.device)
        return self.grouped_estimator

    def _activity(self, physics, energies, concs, receptor_indices, env):
        """(B, R) receptor activity, or (B, C) cell activity in cell mode.

        The only place the two pictures differ. Everything downstream takes a
        (B, N) tensor and does not care what N counts.
        """
        comp = env.composition
        if self.readout is None:
            return physics(energies, concs, comp)
        return cell_activity(physics, self.readout, energies, concs, comp,
                             chunk_size=self.config.cell_pool_chunk,
                             recompute=self.config.recompute_backward)

    # -- evaluation --------------------------------------------------------

    def _eval_stats(self, env, physics, loss_fn, receptor_indices, batch_size,
                    measurement_fns=None):
        """Measure over batch_size samples, with per-pass memory bounded by chunks.

        Measurement is always float32 and never under autocast: a few thousandths
        of a bit is acceptable training noise but not in a published number.
        """
        cfg = self.config
        chunk = min(cfg.eval_chunk_size or cfg.batch_size, batch_size)
        requested = set(cfg.measurement_fns if measurement_fns is None else measurement_fns)
        unknown = requested - MEASUREMENTS
        if unknown:
            raise ValueError(f"Unknown measurements {sorted(unknown)}. "
                             f"Choose from {sorted(MEASUREMENTS)}.")
        if not requested:
            raise ValueError("No measurement requested.")

        native_grouped_kt = isinstance(loss_fn, GroupedKTMutualInformationLoss)
        want_grouped = "grouped_information" in requested
        want_lower = bool(requested & {"entropy_kt", "mutual_information_kt"})
        want_upper = bool(requested & {"entropy_kt_upper", "mutual_information_kt_upper"})
        want_grouped_counting = "grouped_counting" in requested
        want_hard = "codeword_entropy" in requested
        grouped_est = self._grouped_estimator_for(loss_fn) if want_grouped else None
        counter = GroupedResponseCounter(self.grouping) if want_grouped_counting else None

        stat = {}
        with torch.no_grad():
            E, concs, _ = env.sample_batch(chunk)
            activity = self._activity(physics, E, concs, receptor_indices, env)

            if want_grouped_counting and self._response_generator is None:
                # Output sampling must not perturb the world's or training RNG.
                self._response_generator = torch.Generator(device=activity.device)
                self._response_generator.manual_seed((torch.initial_seed() + 104729) % (2 ** 63))

            acts, hard_codes = [], []
            cond_sum = torch.zeros((), dtype=torch.float64, device=activity.device)
            if want_grouped:
                prob_sum = torch.zeros(grouped_est.n_states, dtype=torch.float64,
                                       device=activity.device)
                grouped_cond_sum = torch.zeros((), dtype=torch.float64, device=activity.device)
            n_have = 0
            while True:
                n = activity.shape[0]
                cond_sum += compute_response_conditional_entropy(activity).double() * n
                if want_grouped:
                    p_sum, c_sum = grouped_est.sufficient_statistics(activity)
                    prob_sum += p_sum.double()
                    grouped_cond_sum += c_sum.double()
                if want_lower or want_upper:
                    acts.append(loss_fn.grouping.probabilities(activity)
                                if native_grouped_kt else activity)
                if want_grouped_counting:
                    counter.update(activity, self._response_generator)
                if want_hard:
                    hard_codes.append((activity > 0.5).cpu())
                n_have += n
                if n_have >= batch_size:
                    break
                E_k, concs_k, _ = env.sample_batch(min(chunk, batch_size - n_have))
                activity = self._activity(physics, E_k, concs_k, receptor_indices, env)

            h_cond = (cond_sum / n_have).item()
            stat["response_evaluation_samples"] = n_have

            if want_grouped:
                metrics = grouped_est.metrics_from_statistics(prob_sum, grouped_cond_sum, n_have)
                if "grouped_information" in requested:
                    stat.update({k: v.item() for k, v in metrics.items()})
                    stat.update(grouped_n_groups=grouped_est.n_groups,
                                grouped_n_states=grouped_est.n_states,
                                grouped_count_entropy_upper=math.log2(grouped_est.n_states))
            if requested & {"conditional_entropy_response", "mutual_information_kt",
                            "mutual_information_kt_upper"}:
                stat["conditional_entropy_response"] = h_cond
            if want_hard:
                stat.update(codeword_entropy(torch.cat(hard_codes)))
            if want_grouped_counting:
                stat.update(counter.metrics())
            if want_lower or want_upper:
                big_soft = loss_fn.compute_soft_assignment(torch.cat(acts))
                mult = loss_fn.grouping.group_sizes if native_grouped_kt else None
                if want_lower:
                    i_lower = compute_kt_entropy(
                        big_soft, chunk_size=EVAL_TILE, return_mi=True,
                        use_compile=self.config.compile_kt, multiplicities=mult).item()
                    if "entropy_kt" in requested:
                        stat["full_array_entropy_kt"] = i_lower + h_cond
                    if "mutual_information_kt" in requested:
                        stat["mutual_information_kt"] = i_lower
                if want_upper:
                    i_upper = compute_kt_upper_entropy(
                        big_soft, chunk_size=EVAL_TILE, return_mi=True,
                        multiplicities=mult).item()
                    if "entropy_kt_upper" in requested:
                        stat["full_array_entropy_kt_upper"] = i_upper + h_cond
                    if "mutual_information_kt_upper" in requested:
                        stat["mutual_information_kt_upper"] = i_upper
        return stat

    # -- training ----------------------------------------------------------

    def _train(self, env, physics, loss_fn, optimizer, receptor_indices):
        cfg = self.config
        is_mi = True        # every objective is an MI objective now
        if cfg.family_moves:
            place_near_families(env, torch.arange(env.n_genes))
        start_temp = (compute_initial_temperature(env, receptor_indices)
                      if cfg.initial_temperature == "auto"
                      else float(cfg.initial_temperature))
        end_temp = cfg.temperature
        anneal_epochs = max(1, int(0.8 * cfg.epochs))
        log_every = max(1, cfg.epochs // 100)
        moves = None
        if cfg.family_moves:
            physics.temperature = start_temp
            moves = FamilyMoves(self, env, physics, loss_fn, optimizer, anneal_epochs)

        stats = []
        for epoch in range(cfg.epochs):
            optimizer.zero_grad()
            frac = min(1.0, epoch / anneal_epochs)
            current_temp = (end_temp + (start_temp - end_temp) * (1.0 - frac)
                            if end_temp < start_temp else end_temp)
            physics.temperature = current_temp

            energies, concs, _ = env.sample_batch(cfg.batch_size)
            activity = self._activity(physics, energies, concs, receptor_indices, env).float()
            loss = loss_fn(activity)
            loss.backward()
            optimizer.step()
            if moves is not None and moves.due(epoch):
                moves.step(epoch)

            if epoch % log_every:
                continue
            if cfg.per_epoch_measure:
                # Measure at the SHARP endpoint, which is what the final number
                # uses; otherwise the curve describes a model nobody reports.
                physics.temperature = end_temp
                stat = self._eval_stats(env, physics, loss_fn, receptor_indices,
                                        cfg.test_batch_size)
                physics.temperature = current_temp
            else:
                metric = "train_mutual_information" if is_mi else "train_entropy"
                stat = {"loss": float(loss.item()), metric: float(-loss.item())}
            if is_mi:
                stat["train_mutual_information"] = float(-loss.item())
            stat["lr"] = optimizer.param_groups[0]["lr"]
            if moves is not None:
                stat["moves_accepted"] = moves.accepted
            stats.append(stat)

        physics.temperature = end_temp
        return {k: [s[k] for s in stats] for k in stats[0]} if stats else {}

    def evaluate(self, measurements, batch_size, eval_chunk_size=None):
        """Measure the built model over batch_size inputs. One measurement, not repeated.

        Repeating would show evaluation NOISE, but a frequency-based entropy
        estimate is biased downward by an amount fixed by batch_size, so ten
        repeats of a small batch carry exactly the bias of one. At equal cost,
        put everything into batch_size.

        Values are stored as one-element lists so every consumer keeps one shape.
        """
        if self.env is None:
            raise RuntimeError("Call build() first, or load a checkpoint into it.")
        if eval_chunk_size is not None:
            self.config.eval_chunk_size = eval_chunk_size
        stat = self._eval_stats(self.env, self.physics, self.loss_fn,
                                self.receptor_indices, batch_size,
                                measurement_fns=tuple(measurements))
        return {k: [v] for k, v in stat.items()}


    def run(self):
        """Train and checkpoint. Deliberately does NOT measure.

        Measuring is a separate step because different estimators want
        different batch sizes: sampled counting needs a budget that grows with
        the number of cells, while KT costs O(B^2) and wants a modest one.
        Folding one of them into training would silently privilege it, and
        re-measuring would mean retraining. See tasks/_lib/measure.py.
        """
        # SweepLogger already wrote this, but a Runner built directly would
        # otherwise leave a directory nothing can load.
        self.logger.save_config(self.config)
        env, physics, _loss, optimizer, receptor_indices = self.build()
        history = self._train(env, physics, self.loss_fn, optimizer, receptor_indices)
        for i in range(len(next(iter(history.values()), []))):
            self.logger.save_stats(i, {k: v[i] for k, v in history.items()})
        self.logger.save_checkpoint(self.config.epochs, env, physics,
                                    receptor_indices, self.readout)
        return env
