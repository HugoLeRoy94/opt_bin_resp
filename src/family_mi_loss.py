import torch
import torch.nn as nn
from .bin_loss import compute_shannon_joint_entropy, compute_renyi_joint_entropy

class MaximizeMutualInformationLigandLoss(nn.Module):
    """
    Maximizes the mutual information between the array activity and the ligand mixture identity:
    I(A ; M) = H(A) - H(A | M)
    
    This is achieved by minimizing the loss:
    Loss = H(A | M) - H(A)
    """
    def __init__(self, entropy_type: str):
        super().__init__()
        self.entropy_type = entropy_type

    def compute_soft_assignment(self, activity: torch.Tensor) -> torch.Tensor:
        # Binary system: [P(inactive), P(active)]
        return torch.stack([1.0 - activity, activity], dim=-1)

    def forward(self, activity: torch.Tensor, mixture_masks: torch.Tensor):
        # 1. Compute H(A) on the current mixed training batch
        soft_assign = self.compute_soft_assignment(activity)
        
        if self.entropy_type == 'shannon':
            entropy_fn = compute_shannon_joint_entropy
        elif self.entropy_type == 'renyi':
            entropy_fn = compute_renyi_joint_entropy
        else:
            raise ValueError(f"Unknown entropy_type: {self.entropy_type}. Choose 'shannon' or 'renyi'.")
            
        h_a = entropy_fn(soft_assign)
        
        # 2. Compute H(A | M) by grouping the existing batch by exact mixture identity
        powers = 2 ** torch.arange(mixture_masks.shape[1], device=mixture_masks.device, dtype=mixture_masks.dtype)
        mixture_ids = (mixture_masks * powers).sum(dim=-1).long()
        
        unique_mixtures = torch.unique(mixture_ids)
        total_cond_h = 0.0
        
        for m_idx in unique_mixtures:
            mask = (mixture_ids == m_idx)
            soft_assign_m = soft_assign[mask]
            total_cond_h = total_cond_h + entropy_fn(soft_assign_m)
            
        h_a_given_m = total_cond_h / len(unique_mixtures)
        
        # Maximize I(A; M) -> Minimize H(A|M) - H(A)
        return h_a_given_m - h_a