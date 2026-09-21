"""Exact cell MI by enumerating counts of conditionally independent identical cells.

See theory notes 04, 05, 06, 07, 08 (cell bounds), and 09. Groups are fixed by
identical abundance rows, never inferred from similar sampled activities.
"""
import math

import torch

from src.bin_loss import DiscreteExactLoss, KT_EPS


class GroupedCellMutualInformationLoss(DiscreteExactLoss):
    """Cell-only count enumeration; ``compute_entropy`` still returns full H(Y).

    Inputs remain (batch, cells), with one probability per labeled cell. Identical
    W rows must use the same readout parameters and conditionally independent
    output noise. The runner guarantees this for its shared cell readout.
    """

    def __init__(self, W: torch.Tensor, max_states: int = 65536):
        super().__init__(entropy_type="shannon")
        self.entropy_type = "grouped_mi"
        if W.ndim != 2 or min(W.shape) == 0:
            raise ValueError("Grouping requires a nonempty cell abundance matrix W.")
        if max_states < 1:
            raise ValueError("cell_grouped_max_states must be positive.")
        _, inverse, sizes = torch.unique(
            W.detach(), dim=0, return_inverse=True, return_counts=True)
        self.n_cells = W.shape[0]
        self.n_groups = sizes.numel()
        # Python integers avoid overflow before the allocation guard. This is
        # one-time shape metadata, not a loop over stimuli or response states.
        self.n_states = math.prod((sizes + 1).tolist())
        if self.n_states > max_states:
            raise ValueError(
                f"Grouped cell alphabet has {self.n_states} states, exceeding "
                f"cell_grouped_max_states={max_states}. Use kt_mi / counting, "
                "or explicitly raise the limit after checking memory.")
        radices = sizes + 1
        strides = torch.cat((radices.new_ones(1), radices.cumprod(0)[:-1]))
        states = (torch.arange(self.n_states, device=W.device)[:, None]
                  // strides[None, :]) % radices[None, :]
        remaining = sizes[None, :] - states
        # Cache coefficients in double precision; cast to the activity dtype at
        # evaluation. A joint count has prod_j choose(n_j, k_j) labeled patterns.
        log_multiplicity = (
            torch.lgamma(sizes.double() + 1)[None, :]
            - torch.lgamma(states.double() + 1)
            - torch.lgamma(remaining.double() + 1)
        ).sum(1)
        self.register_buffer("cell_to_group", inverse)
        self.register_buffer("group_sizes", sizes)
        self.register_buffer("count_states", states)
        self.register_buffer("remaining_counts", remaining)
        self.register_buffer("log_multiplicity", log_multiplicity)

    def sufficient_statistics(self, activity: torch.Tensor):
        """Return sums of P(K|x) and H(K|x), mergeable across input chunks.

        No binary draws, no B² comparisons, and no (B,S,J) broadcast. The main
        work is two (B,J) @ (J,S) products, and storage is O(BS + SJ + BC).
        """
        if activity.ndim != 2 or activity.shape[1] != self.n_cells or not activity.shape[0]:
            raise ValueError(f"Expected nonempty (batch, {self.n_cells}) cell activity.")
        # Losses run outside autocast; also support direct calls with half inputs.
        a = activity if activity.dtype == torch.float64 else activity.float()
        a = a.clamp(KT_EPS, 1.0 - KT_EPS)
        p = a.new_zeros((a.shape[0], self.n_groups)).scatter_add(
            1, self.cell_to_group[None, :].expand(a.shape[0], -1), a)
        p = (p / self.group_sizes).clamp(KT_EPS, 1.0 - KT_EPS)
        # Averaging identical probabilities distributes gradients symmetrically
        # across copies, matching the full binary channel's activity gradients.
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
