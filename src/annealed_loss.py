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
"""
import torch
import torch.nn as nn

from .bin_loss import (
    compute_blocked_entropy,
    compute_renyi_joint_entropy,
    compute_shannon_joint_entropy,
)


class AnnealedEntropyLoss(nn.Module):
    """Annealed blocked→Rényi entropy maximization loss.

    Args:
        block_size:    receptor block width for the blocked Shannon estimator.
        n_partitions:  number of random partitions averaged in blocked Shannon.
    """

    def __init__(self, block_size: int = 15, n_partitions: int = 4):
        super().__init__()
        self.block_size = block_size
        self.n_partitions = n_partitions

    def compute_soft_assignment(self, activity: torch.Tensor) -> torch.Tensor:
        return torch.stack([1.0 - activity, activity], dim=-1)

    def compute_entropy(
        self, activity: torch.Tensor, entropy_type: str = "blocked"
    ) -> torch.Tensor:
        """Positive scalar entropy in bits — passthrough for measurement helpers."""
        soft = self.compute_soft_assignment(activity)
        if entropy_type == "shannon":
            return compute_shannon_joint_entropy(soft)
        elif entropy_type == "renyi":
            return compute_renyi_joint_entropy(soft)
        elif entropy_type == "blocked":
            return compute_blocked_entropy(soft, self.block_size, self.n_partitions)
        raise ValueError(
            f"Unknown entropy_type: {entropy_type!r}. "
            "Choose 'shannon', 'renyi', or 'blocked'."
        )

    def forward(
        self, activity: torch.Tensor, epoch: int, epochs: int
    ) -> torch.Tensor:
        soft = self.compute_soft_assignment(activity)
        h_blocked = compute_blocked_entropy(soft, self.block_size, self.n_partitions)
        h_renyi = compute_renyi_joint_entropy(soft)
        lam = epoch / max(1, epochs)
        return -((1.0 - lam) * h_blocked + lam * h_renyi)
