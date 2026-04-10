import torch
import torch.nn as nn
from .bin_loss import compute_discrete_joint_entropy

class MaximizeMutualInformationFamilyLoss(nn.Module):
    """
    Maximizes the mutual information between the array activity and the ligand family identity:
    I(A ; F) = H(A) - H(A | F)
    
    This is achieved by minimizing the loss:
    Loss = H(A | F) - H(A)
    """
    def __init__(self, env, physics, receptor_indices, n_samples_per_family=2048):
        super().__init__()
        self.env = env
        self.physics = physics
        self.receptor_indices = receptor_indices
        self.n_samples = n_samples_per_family

    def compute_soft_assignment(self, activity: torch.Tensor) -> torch.Tensor:
        # Binary system: [P(inactive), P(active)]
        return torch.stack([1.0 - activity, activity], dim=-1)

    def forward(self, activity: torch.Tensor):
        # 1. Compute H(A) on the current mixed training batch
        soft_assign = self.compute_soft_assignment(activity)
        h_a = compute_discrete_joint_entropy(soft_assign)
        
        # 2. Compute H(A | F) by sampling each family
        n_families = self.env.n_families
        if n_families == 0:
            return -h_a
            
        total_cond_h = 0.0
        for f_idx in range(n_families):
            energies, concs, _ = self.env.sample_specific_family(batch_size=self.n_samples, family_id=f_idx)
            act_f = self.physics(energies, concs, self.receptor_indices)
            soft_assign_f = self.compute_soft_assignment(act_f)
            total_cond_h = total_cond_h + compute_discrete_joint_entropy(soft_assign_f)
            
        h_a_given_f = total_cond_h / n_families
        
        # Maximize I(A; F) -> Minimize H(A|F) - H(A)
        return h_a_given_f - h_a