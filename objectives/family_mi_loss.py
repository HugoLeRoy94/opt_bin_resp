import torch
import torch.nn as nn
from .bin_loss import compute_shannon_joint_entropy, compute_renyi_joint_entropy

class MaximizeMutualInformationFamilyLoss(nn.Module):
    """
    Maximizes the mutual information between the array activity and the ligand family identity:
    I(A ; F) = H(A) - H(A | F)
    
    This is achieved by minimizing the loss:
    Loss = H(A | F) - H(A)
    """
    def __init__(self, entropy_type: str = 'renyi'):
        super().__init__()
        self.entropy_type = entropy_type

    def compute_soft_assignment(self, activity: torch.Tensor) -> torch.Tensor:
        # Binary system: [P(inactive), P(active)]
        return torch.stack([1.0 - activity, activity], dim=-1)

    def forward(self, activity: torch.Tensor, family_ids: torch.Tensor):
        # 1. Compute H(A) on the current mixed training batch
        soft_assign = self.compute_soft_assignment(activity)
        
        if self.entropy_type == 'shannon':
            entropy_fn = compute_shannon_joint_entropy
        elif self.entropy_type == 'renyi':
            entropy_fn = compute_renyi_joint_entropy
        else:
            raise ValueError(f"Unknown entropy_type: {self.entropy_type}. Choose 'shannon' or 'renyi'.")
            
        h_a = entropy_fn(soft_assign)
        
        # 2. Compute H(A | F) by grouping the existing batch by family
        unique_families = torch.unique(family_ids)
        total_cond_h = 0.0
        
        for f_idx in unique_families:
            mask = (family_ids == f_idx)
            soft_assign_f = soft_assign[mask]
            total_cond_h = total_cond_h + entropy_fn(soft_assign_f)
            
        h_a_given_f = total_cond_h / len(unique_families)
        
        # Maximize I(A; F) -> Minimize H(A|F) - H(A)
        return h_a_given_f - h_a