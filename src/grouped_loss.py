"""Exact, KT, and sampled-count estimators sharing identical-cell grouping.

See theory notes 04, 05, 06, 07, 08 (cell bounds), and 09. Groups are fixed by
identical abundance rows, never inferred from similar sampled activities.
"""
import math

import torch

from src.bin_loss import (DiscreteExactLoss, KTMutualInformationLoss, compute_kt_entropy,
                          compute_kt_upper_entropy)
from src.response_groups import CellGrouping
from src.counting import SymbolCounter


class GroupedCellMutualInformationLoss(DiscreteExactLoss):
    """Cell-only count enumeration; ``compute_entropy`` still returns full H(Y).

    Inputs remain (batch, cells), with one probability per labeled cell. Identical
    W rows must use the same readout parameters and conditionally independent
    output noise. The runner guarantees this for its shared cell readout.
    """

    def __init__(self, W: torch.Tensor, max_states: int = 65536):
        super().__init__(entropy_type="shannon")
        self.entropy_type = "grouped_mi"
        if max_states < 1:
            raise ValueError("cell_grouped_max_states must be positive.")
        self.grouping = W if isinstance(W, CellGrouping) else CellGrouping(W)
        self.n_cells = self.grouping.n_cells
        self.n_groups = self.grouping.n_groups
        self.n_states = self.grouping.n_states
        sizes = self.grouping.group_sizes
        if self.n_states > max_states:
            raise ValueError(
                f"Grouped cell alphabet has {self.n_states} states, exceeding "
                f"cell_grouped_max_states={max_states}. Select grouped_kt_mi with "
                "grouped_counting instead of exact grouped_information, or raise "
                "the limit after checking memory.")
        radices = sizes + 1
        strides = torch.cat((radices.new_ones(1), radices.cumprod(0)[:-1]))
        states = (torch.arange(self.n_states, device=sizes.device)[:, None]
                  // strides[None, :]) % radices[None, :]
        remaining = sizes[None, :] - states
        # Cache coefficients in double precision; cast to the activity dtype at
        # evaluation. A joint count has prod_j choose(n_j, k_j) labeled patterns.
        log_multiplicity = self.grouping.log_choose(sizes[None, :], states).sum(1)
        self.register_buffer("count_states", states)
        self.register_buffer("remaining_counts", remaining)
        self.register_buffer("log_multiplicity", log_multiplicity)

    def sufficient_statistics(self, activity: torch.Tensor):
        """Return sums of P(K|x) and H(K|x), mergeable across input chunks.

        No binary draws, no B² comparisons, and no (B,S,J) broadcast. The main
        work is two (B,J) @ (J,S) products, and storage is O(BS + SJ + BC).
        """
        p = self.grouping.probabilities(activity)
        log_joint = (p.log() @ self.count_states.to(p.dtype).T
                     + torch.log1p(-p) @ self.remaining_counts.to(p.dtype).T
                     + self.log_multiplicity.to(p.dtype)[None, :])
        # The exact rows sum to one; normalization suppresses floating-point
        # drift in large binomial coefficients without changing the channel.
        log_joint = log_joint - torch.logsumexp(log_joint, dim=1, keepdim=True)
        joint = log_joint.exp()
        conditional_sum = -(joint * log_joint).sum() / math.log(2)
        return joint.sum(0), conditional_sum

    def metrics_from_statistics(self, probability_sum, conditional_sum, n_samples):
        """Restore both entropies of labeled outputs; only count keys mean H(K)."""
        marginal = probability_sum / n_samples
        h_count = -(marginal * marginal.clamp_min(
            torch.finfo(marginal.dtype).tiny).log2()).sum()
        h_count_cond = conditional_sum / n_samples
        multiplicity = (marginal * self.log_multiplicity.to(marginal.dtype)).sum() / math.log(2)
        return {
            "grouped_count_entropy": h_count,
            "grouped_count_conditional_entropy": h_count_cond,
            "grouped_label_entropy": multiplicity,
            "response_entropy_grouped": h_count + multiplicity,
            "conditional_entropy_response_grouped": h_count_cond + multiplicity,
            "mutual_information_grouped": h_count - h_count_cond,
        }

    def compute_metrics(self, activity):
        return self.metrics_from_statistics(
            *self.sufficient_statistics(activity), activity.shape[0])

    def compute_entropy(self, activity, entropy_type=None, use_cache=True):
        if entropy_type in (None, "grouped_mi", "shannon"):
            return self.compute_metrics(activity)["response_entropy_grouped"]
        return super().compute_entropy(activity, entropy_type, use_cache)

    def forward(self, activity):
        return -self.compute_metrics(activity)["mutual_information_grouped"]


class GroupedKTMutualInformationLoss(KTMutualInformationLoss):
    """Same KT bound and labeled-entropy convention, with weighted group columns."""
    def __init__(self, W, **kwargs):
        super().__init__(**kwargs)
        self.grouping = W if isinstance(W, CellGrouping) else CellGrouping(W)

    def bound(self, activity, *, upper=False, return_mi=False):
        p = self.grouping.probabilities(activity)
        return self.bound_from_probabilities(p, upper=upper, return_mi=return_mi)

    def bound_from_probabilities(self, p, *, upper=False, return_mi=False):
        soft = torch.stack((1 - p, p), -1)
        kwargs = dict(chunk_size=self.collision_chunk_size, return_mi=return_mi,
                      multiplicities=self.grouping.group_sizes)
        if upper:
            return compute_kt_upper_entropy(soft, **kwargs)
        return compute_kt_entropy(soft, recompute=self.recompute_backward,
                                  use_compile=self.compile_kt, **kwargs)

    def compute_entropy(self, activity, entropy_type=None, use_cache=True):
        if entropy_type in (None, 'kt', 'grouped_kt_mi'):
            return self.bound(activity)
        return super().compute_entropy(activity, entropy_type, use_cache)

    def forward(self, activity):
        return -self.bound(activity, return_mi=True)


class GroupedResponseCounter:
    """Stream binomial draws and analytical noise; never enumerate joint states."""
    def __init__(self, grouping):
        self.grouping = grouping
        self.counter = SymbolCounter()
        self.conditional_sum = 0.0
        self.response_conditional_sum = 0.0

    @torch.no_grad()
    def update(self, activity, generator=None):
        p = self.grouping.probabilities(activity)
        h_count, h_response, _ = self.grouping.conditional_entropies(p)
        n = activity.shape[0]
        self.conditional_sum += n * h_count.item()
        self.response_conditional_sum += n * h_response.item()
        self.counter.update(self.grouping.sample(p, generator))

    def metrics(self):
        plugin, mm, unique, log_n = self.counter.entropy()
        n = self.counter.n_samples
        hc = self.conditional_sum / n
        hyc = self.response_conditional_sum / n
        d = hyc - hc
        return dict(
            grouped_count_entropy_counting_plugin=plugin,
            grouped_count_entropy_counting_mm=mm,
            grouped_count_conditional_entropy_counting=hc,
            grouped_label_entropy_counting=d,
            response_entropy_grouped_counting_plugin=plugin + d,
            response_entropy_grouped_counting_mm=mm + d,
            conditional_entropy_response_grouped_counting=hyc,
            mutual_information_grouped_counting_plugin=plugin - hc,
            mutual_information_grouped_counting_mm=mm - hc,
            grouped_counting_K_hat=unique,
            grouped_counting_unique_fraction=unique / n,
            grouped_counting_samples=n,
            grouped_counting_log2B=log_n,
            **self.grouping.diagnostics())
