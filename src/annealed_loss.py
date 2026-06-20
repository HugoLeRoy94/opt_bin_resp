# Documented in:
#   doc/theory/07_optimization_pipeline.md  (stage 4d: AnnealedEntropyLoss)
#   doc/theory/05_optimization.md           (§5.4 annealed loss schedule)
"""
annealed_loss.py — Annealed blocked→Rényi entropy loss.

Linearly interpolates between the blocked Shannon estimator (good gradients,
fast early convergence) and the Rényi H2 estimator (faithful lower bound,
cannot be gamed) over the course of training:

    loss = -[(1 - lam) * H_blocked + lam * H_renyi],   lam = epoch / epochs.

At epoch 0 the loss is pure blocked; at the final epoch it is pure Rényi.
No extra weight knob — the schedule is fully determined by training progress.

The blocked term uses correlation-aware partitioning (see bin_loss.py) with a
cached partition refreshed every block_refresh_interval training steps.
"""
import torch
import torch.nn as nn

from .bin_loss import (
    compute_blocked_entropy,
    compute_correlation_aware_partition,
    compute_renyi_joint_entropy,
    compute_shannon_joint_entropy,
)


class AnnealedEntropyLoss(nn.Module):
    """Annealed blocked→Rényi entropy maximization loss.

    Args:
        block_size:              receptor block width for the blocked Shannon estimator.
        n_partitions:            kept for config compatibility (unused).
        block_refresh_interval:  training steps between partition recomputation.
    """

    def __init__(self, block_size: int = 15, n_partitions: int = 4,
                 block_refresh_interval: int = 50):
        super().__init__()
        self.block_size = block_size
        self.n_partitions = n_partitions
        self.block_refresh_interval = block_refresh_interval
        self._cached_partition = None
        self._steps_since_refresh = 0

    def compute_soft_assignment(self, activity: torch.Tensor) -> torch.Tensor:
        return torch.stack([1.0 - activity, activity], dim=-1)

    def _get_blocked_entropy(self, activity: torch.Tensor, soft_assign: torch.Tensor,
                             use_cache: bool) -> torch.Tensor:
        if not use_cache:
            return compute_blocked_entropy(soft_assign, self.block_size)

        self._steps_since_refresh += 1
        if (self._cached_partition is None
                or self._steps_since_refresh >= self.block_refresh_interval):
            self._cached_partition = compute_correlation_aware_partition(
                activity.detach(), self.block_size
            )
            self._steps_since_refresh = 0
        return compute_blocked_entropy(soft_assign, self.block_size, self._cached_partition)

    def compute_entropy(
        self, activity: torch.Tensor, entropy_type: str = "blocked",
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Positive scalar entropy in bits — passthrough for measurement helpers."""
        soft = self.compute_soft_assignment(activity)
        if entropy_type == "shannon":
            return compute_shannon_joint_entropy(soft)
        elif entropy_type == "renyi":
            return compute_renyi_joint_entropy(soft)
        elif entropy_type == "blocked":
            return self._get_blocked_entropy(activity, soft, use_cache)
        raise ValueError(
            f"Unknown entropy_type: {entropy_type!r}. "
            "Choose 'shannon', 'renyi', or 'blocked'."
        )

    def forward(
        self, activity: torch.Tensor, epoch: int, epochs: int
    ) -> torch.Tensor:
        soft = self.compute_soft_assignment(activity)
        h_blocked = self._get_blocked_entropy(activity, soft, use_cache=True)
        h_renyi = compute_renyi_joint_entropy(soft)
        lam = epoch / max(1, epochs)
        return -((1.0 - lam) * h_blocked + lam * h_renyi)
