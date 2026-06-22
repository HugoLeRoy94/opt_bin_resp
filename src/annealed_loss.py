# Documented in:
#   doc/theory/07_optimization_pipeline.md  (stage 4d: AnnealedEntropyLoss)
#   doc/theory/05_optimization.md           (§5.4 annealed loss schedule)
"""
annealed_loss.py — Annealed blocked->collision entropy loss.

Linearly interpolates between the blocked Shannon estimator (good gradients,
fast early convergence) and the collision H2 estimator (faithful lower bound,
cannot be gamed) over the course of training:

    loss = -[(1 - lam) * H_blocked + lam * H_collision],   lam = epoch / epochs.

At epoch 0 the loss is pure blocked; at the final epoch it is pure collision.
No extra weight knob — the schedule is fully determined by training progress.

The blocked term uses correlation-aware partitioning (see bin_loss.py) with a
cached partition refreshed every block_refresh_interval training steps.
"""
import torch
import torch.nn as nn

from .bin_loss import (
    compute_blocked_corrected_entropy,
    compute_blocked_entropy,
    compute_collision_entropy,
    compute_correlation_aware_partition,
    compute_shannon_joint_entropy,
)


class AnnealedEntropyLoss(nn.Module):
    """Annealed blocked->collision entropy maximization loss.

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

    def _refresh_partition(self, activity: torch.Tensor, use_cache: bool):
        if not use_cache:
            return compute_correlation_aware_partition(activity.detach(), self.block_size)
        self._steps_since_refresh += 1
        if (self._cached_partition is None
                or self._steps_since_refresh >= self.block_refresh_interval):
            self._cached_partition = compute_correlation_aware_partition(
                activity.detach(), self.block_size
            )
            self._steps_since_refresh = 0
        return self._cached_partition

    def _get_blocked_entropy(self, activity: torch.Tensor, soft_assign: torch.Tensor,
                             use_cache: bool) -> torch.Tensor:
        partition = self._refresh_partition(activity, use_cache)
        return compute_blocked_entropy(soft_assign, self.block_size, partition)

    def compute_entropy(
        self, activity: torch.Tensor, entropy_type: str = "blocked",
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Positive scalar entropy in bits — passthrough for measurement helpers."""
        soft = self.compute_soft_assignment(activity)
        if entropy_type == "shannon":
            return compute_shannon_joint_entropy(soft)
        elif entropy_type == "collision":
            return compute_collision_entropy(soft)
        elif entropy_type == "blocked":
            return self._get_blocked_entropy(activity, soft, use_cache)
        elif entropy_type == "blocked_corrected":
            partition = self._refresh_partition(activity, use_cache)
            return compute_blocked_corrected_entropy(soft, self.block_size, partition)
        raise ValueError(
            f"Unknown entropy_type: {entropy_type!r}. "
            "Choose 'shannon', 'collision', 'blocked', or 'blocked_corrected'."
        )

    def forward(
        self, activity: torch.Tensor, epoch: int, epochs: int
    ) -> torch.Tensor:
        soft = self.compute_soft_assignment(activity)
        h_blocked = self._get_blocked_entropy(activity, soft, use_cache=True)
        h_collision = compute_collision_entropy(soft)
        lam = epoch / max(1, epochs)
        return -((1.0 - lam) * h_blocked + lam * h_collision)


class BlockedToCorrectedLoss(nn.Module):
    """Annealed blocked -> blocked-corrected Shannon loss.

    loss = -[(1 - lam) * H_blocked + lam * H_blocked_corrected]
    where lam = epoch / epochs (0 at start, 1 at end).

    Set ``lam_override=1.0`` to use pure blocked_corrected at all epochs.

    Args:
        block_size:              receptor block width.
        n_partitions:            kept for config compat (unused).
        block_refresh_interval:  training steps between partition recomputation.
        lam_override:            if not None, fixes lambda (bypasses epoch schedule).
    """

    def __init__(self, block_size: int = 15, n_partitions: int = 4,
                 block_refresh_interval: int = 50, lam_override: float = None):
        super().__init__()
        self.block_size = block_size
        self.n_partitions = n_partitions
        self.block_refresh_interval = block_refresh_interval
        self.lam_override = lam_override
        self._cached_partition = None
        self._steps_since_refresh = 0

    def compute_soft_assignment(self, activity: torch.Tensor) -> torch.Tensor:
        return torch.stack([1.0 - activity, activity], dim=-1)

    def _refresh_partition(self, activity: torch.Tensor, use_cache: bool):
        if not use_cache:
            return compute_correlation_aware_partition(activity.detach(), self.block_size)
        self._steps_since_refresh += 1
        if (self._cached_partition is None
                or self._steps_since_refresh >= self.block_refresh_interval):
            self._cached_partition = compute_correlation_aware_partition(
                activity.detach(), self.block_size
            )
            self._steps_since_refresh = 0
        return self._cached_partition

    def compute_entropy(
        self, activity: torch.Tensor, entropy_type: str = "blocked_corrected",
        use_cache: bool = True,
    ) -> torch.Tensor:
        soft = self.compute_soft_assignment(activity)
        if entropy_type == "shannon":
            return compute_shannon_joint_entropy(soft)
        elif entropy_type == "collision":
            return compute_collision_entropy(soft)
        elif entropy_type == "blocked":
            partition = self._refresh_partition(activity, use_cache)
            return compute_blocked_entropy(soft, self.block_size, partition)
        elif entropy_type == "blocked_corrected":
            partition = self._refresh_partition(activity, use_cache)
            return compute_blocked_corrected_entropy(soft, self.block_size, partition)
        raise ValueError(f"Unknown entropy_type: {entropy_type!r}")

    def forward(
        self, activity: torch.Tensor, epoch: int, epochs: int
    ) -> torch.Tensor:
        soft = self.compute_soft_assignment(activity)
        partition = self._refresh_partition(activity, use_cache=True)
        h_blocked = compute_blocked_entropy(soft, self.block_size, partition)
        h_corrected = compute_blocked_corrected_entropy(soft, self.block_size, partition)
        lam = self.lam_override if self.lam_override is not None else (epoch / max(1, epochs))
        return -((1.0 - lam) * h_blocked + lam * h_corrected)
