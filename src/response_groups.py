"""Exact cell equivalence, shared by enumeration, KT and sampled counting.

Only identical abundance rows qualify. All cells must share readout parameters
and have independent response noise conditional on the complete input.
"""
import math

import torch
from torch import nn

from src.bin_loss import KT_EPS, compute_response_conditional_entropy


class CellGrouping(nn.Module):
    """Grouping metadata and per-group operations; never builds joint states."""

    def __init__(self, W):
        super().__init__()
        if W.ndim != 2 or min(W.shape) == 0:
            raise ValueError("Grouping requires a nonempty cell abundance matrix W.")
        _, inverse, sizes = torch.unique(W.detach(), dim=0, return_inverse=True,
                                        return_counts=True)
        self.n_cells = W.shape[0]
        self.n_groups = sizes.numel()
        self.n_states = math.prod((sizes + 1).tolist())  # Python int: no overflow
        self.register_buffer("cell_to_group", inverse)
        self.register_buffer("group_sizes", sizes)
        # Flat individual binomial supports: C+J entries, not product(n_j+1).
        ids = torch.arange(self.n_groups, device=W.device).repeat_interleave(sizes + 1)
        starts = (sizes + 1).cumsum(0) - (sizes + 1)
        k = torch.arange(self.n_cells + self.n_groups, device=W.device) - starts[ids]
        self.register_buffer("support_groups", ids)
        self.register_buffer("support_counts", k)
        self.register_buffer("support_log_choose", self.log_choose(sizes[ids], k))

    @staticmethod
    def log_choose(n, k):
        return (torch.lgamma(n.double() + 1) - torch.lgamma(k.double() + 1)
                - torch.lgamma((n - k).double() + 1))

    def probabilities(self, activity):
        if activity.ndim != 2 or activity.shape[1] != self.n_cells or not activity.shape[0]:
            raise ValueError(f"Expected nonempty (batch, {self.n_cells}) cell activity.")
        a = activity if activity.dtype == torch.float64 else activity.float()
        a = a.clamp(KT_EPS, 1 - KT_EPS)
        # Average duplicates so partial gradients agree with the ungrouped loss.
        total = a.new_zeros((a.shape[0], self.n_groups)).scatter_add(
            1, self.cell_to_group[None, :].expand(a.shape[0], -1), a)
        return (total / self.group_sizes).clamp(KT_EPS, 1 - KT_EPS)

    def conditional_entropies(self, p):
        """Mean count H(K|X), labeled H(Y|X), and their difference D, in bits."""
        ids, k = self.support_groups, self.support_counts
        log_pmf = (self.support_log_choose.to(p.dtype)
                   + p[:, ids].log() * k
                   + torch.log1p(-p[:, ids]) * (self.group_sizes[ids] - k))
        h_count = -(log_pmf.exp() * log_pmf).sum(1).mean() / math.log(2)
        h_response = compute_response_conditional_entropy(p, multiplicities=self.group_sizes)
        return h_count, h_response, h_response - h_count

    def sample(self, p, generator=None):
        return torch.binomial(self.group_sizes.to(p.dtype).expand_as(p), p,
                              generator=generator).to(torch.int64)

    def diagnostics(self):
        return dict(grouped_n_groups=self.n_groups, grouped_n_states=self.n_states,
                    grouped_count_entropy_upper=math.log2(self.n_states))
