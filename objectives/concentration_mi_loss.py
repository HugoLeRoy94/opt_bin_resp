import torch
import torch.nn as nn
from .bin_loss import compute_shannon_joint_entropy, compute_renyi_joint_entropy

class MaximizeMutualInformationConcentrationLoss(nn.Module):
    """
    Maximizes the mutual information between the array activity and the ligand concentration:
    I(A ; C) = H(A) - H(A | C)
    
    This is achieved by minimizing the loss:
    Loss = H(A | C) - H(A)
    """
    def __init__(self, n_c_bins=10, entropy_type: str = 'renyi'):
        super().__init__()
        self.n_c_bins = n_c_bins
        self.entropy_type = entropy_type

    def compute_soft_assignment(self, activity: torch.Tensor) -> torch.Tensor:
        # Binary system: [P(inactive), P(active)]
        return torch.stack([1.0 - activity, activity], dim=-1)

    def forward(self, activity: torch.Tensor, concs: torch.Tensor):
        # 1. Compute H(A) on the current mixed training batch
        soft_assign = self.compute_soft_assignment(activity)
        
        if self.entropy_type == 'shannon':
            entropy_fn = compute_shannon_joint_entropy
        elif self.entropy_type == 'renyi':
            entropy_fn = compute_renyi_joint_entropy
        else:
            raise ValueError(f"Unknown entropy_type: {self.entropy_type}. Choose 'shannon' or 'renyi'.")
            
        h_a = entropy_fn(soft_assign)
        
        # 2. Compute H(A | C) by sorting and binning the existing batch's concentrations
        sorted_concs, indices = torch.sort(concs)
        sorted_assign = soft_assign[indices]
        
        B = activity.shape[0]
        bin_size = max(1, B // self.n_c_bins)
        total_cond_h = 0.0
        
        for b in range(self.n_c_bins):
            start_idx = b * bin_size
            end_idx = start_idx + bin_size if b < self.n_c_bins - 1 else B
            
            soft_assign_c = sorted_assign[start_idx:end_idx]
            total_cond_h = total_cond_h + entropy_fn(soft_assign_c)
            
        h_a_given_c = total_cond_h / self.n_c_bins
        
        # Maximize I(A; C) -> Minimize H(A|C) - H(A)
        return h_a_given_c - h_a