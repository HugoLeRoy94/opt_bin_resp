import torch
import torch.nn as nn
from .bin_loss import compute_discrete_joint_entropy

class MaximizeMutualInformationConcentrationLoss(nn.Module):
    """
    Maximizes the mutual information between the array activity and the ligand concentration:
    I(A ; C) = H(A) - H(A | C)
    
    This is achieved by minimizing the loss:
    Loss = H(A | C) - H(A)
    """
    def __init__(self, env, physics, receptor_indices, n_samples=2048, n_c_bins=10):
        super().__init__()
        self.env = env
        self.physics = physics
        self.receptor_indices = receptor_indices
        self.n_samples = n_samples
        self.n_c_bins = n_c_bins

    def compute_soft_assignment(self, activity: torch.Tensor) -> torch.Tensor:
        # Binary system: [P(inactive), P(active)]
        return torch.stack([1.0 - activity, activity], dim=-1)

    def forward(self, activity: torch.Tensor):
        # 1. Compute H(A) on the current mixed training batch
        soft_assign = self.compute_soft_assignment(activity)
        h_a = compute_discrete_joint_entropy(soft_assign)
        
        # 2. Compute H(A | C) by binning concentration
        # Sample a large batch to bin by concentration
        energies, concs, _ = self.env.sample_batch(batch_size=self.n_samples * self.n_c_bins)
        
        # Sort by concentration to group them into continuous bins
        sorted_concs, indices = torch.sort(concs)
        sorted_energies = energies[indices]
        
        bin_size = len(concs) // self.n_c_bins
        total_cond_h = 0.0
        
        for b in range(self.n_c_bins):
            start_idx = b * bin_size
            end_idx = start_idx + bin_size
            
            bin_energies = sorted_energies[start_idx:end_idx]
            bin_concs = sorted_concs[start_idx:end_idx]
            
            act_c = self.physics(bin_energies, bin_concs, self.receptor_indices)
            soft_assign_c = self.compute_soft_assignment(act_c)
            total_cond_h = total_cond_h + compute_discrete_joint_entropy(soft_assign_c)
            
        h_a_given_c = total_cond_h / self.n_c_bins
        
        # Maximize I(A; C) -> Minimize H(A|C) - H(A)
        return h_a_given_c - h_a