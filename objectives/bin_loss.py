import torch
import torch.nn as nn
import math

def compute_discrete_joint_entropy(soft_assign: torch.Tensor) -> torch.Tensor:
    """
    Computes the joint entropy of a discrete system from soft assignments.
    Switches between exact enumeration for small state spaces and Monte Carlo
    estimation for large state spaces.
    """
    # B : batch size
    # R : # of receptors 
    # K : # of activity bins = 2
    B, R, K = soft_assign.shape 
    
    # Dynamic switch based on computational complexity
    if K ** R <= 1024:
        # ------------------------------------------------------
        # METHOD A: Exact Enumeration (For small arrays)
        # ------------------------------------------------------
        # We iteratively build the exact (B, K^R) probability tensor
        joint_p = soft_assign[:, 0, :] # Start with receptor 0: (B, K)
        
        for r in range(1, R):
            # Multiply current combinations by the probabilities of the next receptor
            # (B, K^{r}, 1) * (B, 1, K) -> flat to (B, K^{r+1})
            joint_p = (joint_p.unsqueeze(-1) * soft_assign[:, r, :].unsqueeze(1)).view(B, -1)
        
        # Average across the batch to get the true probability of every possible state
        p_a = joint_p.mean(dim=0) # Shape: (K^R,)
        
        # Use clamp to prevent log2(0) while maintaining stable gradients
        p_a_safe = torch.clamp(p_a, min=1e-12)
        log_2_p = torch.log2(p_a_safe)
        joint_h = -torch.sum(p_a * log_2_p)
        
    else:
        # ------------------------------------------------------
        # METHOD B: Exact Pairwise Collision (Rényi Entropy H2)
        # ------------------------------------------------------
        # Bypasses the O(K^R) wall entirely by calculating the probability 
        # that two random ligands produce the EXACT same array state. 
        # H_2 = -log2( P(A = B) ). This is highly correlated with Shannon entropy,
        # fully differentiable, and scales linearly with R!
        
        chunk_size = 2048
        
        if B <= chunk_size:
            S_R = soft_assign.permute(1, 0, 2)
            match_probs = torch.bmm(S_R, S_R.permute(0, 2, 1))
            collision_prob = torch.prod(match_probs, dim=0).mean()
        else:
            # Chunked evaluation: average the collision probability over multiple 
            # independent sub-batches. This reduces estimator variance
            # while keeping autograd memory usage strictly bounded.
            max_chunks = 8 # Process up to 8 chunks (16,384 ligands) per step
            n_chunks = min(B // chunk_size, max_chunks)
            
            indices = torch.randperm(B, device=soft_assign.device)
            total_collision_prob = 0.0
            
            for i in range(n_chunks):
                chunk_idx = indices[i * chunk_size : (i + 1) * chunk_size]
                S_R = soft_assign[chunk_idx].permute(1, 0, 2)
                
                match_probs = torch.bmm(S_R, S_R.permute(0, 2, 1))
                total_collision_prob = total_collision_prob + torch.prod(match_probs, dim=0).mean()
                
            collision_prob = total_collision_prob / n_chunks
        
        # Rényi Entropy of order 2 (in bits)
        collision_prob_safe = torch.clamp(collision_prob, min=1e-12)
        joint_h = -torch.log2(collision_prob_safe)

    return joint_h

class DiscreteExactLoss(nn.Module):
    """
    Maximizes the exact discrete binary joint entropy of the array.
    Ideal for systems where components must be correlated (like a thermometer code).
    """
    def __init__(self):
        super().__init__()

    def compute_soft_assignment(self, activity: torch.Tensor) -> torch.Tensor:
        # For binary systems, activity is exactly P(fire). This avoids vanishing gradients.
        return torch.stack([1.0 - activity, activity], dim=-1)

    def forward(self, activity: torch.Tensor):
        soft_assign = self.compute_soft_assignment(activity)
        joint_h = compute_discrete_joint_entropy(soft_assign)
        return -joint_h  # Maximize joint entropy

class DiscreteProxyLoss(nn.Module):
    """
    Maximizes the discrete binary Shannon entropy of the marginals,
    while minimizing a penalty term to encourage receptor independence or diversity.
    """
    def __init__(self, cov_weight: float = 1.0, penalty_type: str = 'repulsion'):
        """
        Args:
            cov_weight: Weight for the penalty term.
            penalty_type: The type of penalty to apply. Can be 'repulsion' (penalizes
                          similar activation profiles) or 'covariance' (penalizes linear
                          correlation). Defaults to 'repulsion'.
        """
        super().__init__()
        self.cov_weight = cov_weight
        self.penalty_type = penalty_type

        if self.penalty_type not in ['repulsion', 'covariance']:
            raise ValueError("penalty_type must be 'repulsion' or 'covariance'")

    def compute_soft_assignment(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Computes binary soft assignments from continuous activity.

        Args:
            activity (torch.Tensor): Continuous activity tensor of shape (Batch, R).

        Returns:
            torch.Tensor: Soft assignment tensor of shape (Batch, R, 2).
        """
        return torch.stack([1.0 - activity, activity], dim=-1)

    def compute_soft_marginal_probabilities(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Computes marginal probabilities for binary states.

        Args:
            activity (torch.Tensor): Continuous activity tensor of shape (Batch, R).

        Returns:
            torch.Tensor: Marginal probabilities of shape (R, 2).
        """
        soft_assign = self.compute_soft_assignment(activity)
        p_marginal = soft_assign.mean(dim=0)
        return p_marginal

    def _compute_soft_histogram_entropy(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Computes the discrete binary Shannon entropy.
        activity shape: (Batch, R)
        Returns shape: (R,) - entropy in bits.
        """
        # 1. Get marginal probabilities from the shared function
        p_marginal = self.compute_soft_marginal_probabilities(activity)
        
        # 2. Clamp to prevent log2(0) crashes
        p_marginal = torch.clamp(p_marginal, min=1e-12)
        
        # 3. Calculate log2(p)
        log_2_p = torch.log2(p_marginal)
        
        # 4. Exact Shannon Entropy in bits
        entropy = -torch.sum(p_marginal * log_2_p, dim=-1)
        
        return entropy

    def _compute_repulsion_penalty(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Penalizes receptors for having identical continuous activation profiles.
        A perfectly shifted thermometer code has low overlap (low penalty),
        while identical receptors have max overlap (high penalty).
        """
        B, R = activity.shape
        A = activity.T # (R, B)
        
        # Pairwise squared Euclidean distance: ||A_i - A_j||^2
        A_sq = (A ** 2).sum(dim=1, keepdim=True) # (R, 1)
        dist_matrix = (A_sq + A_sq.T - 2.0 * torch.matmul(A, A.T)) / B
        
        # Repulsion kernel: exp(-dist / tau)
        tau = 0.05 
        repulsion = torch.exp(-dist_matrix / tau)
        
        mask = ~torch.eye(R, dtype=torch.bool, device=activity.device)
        
        return repulsion[mask].sum()

    def _compute_covariance_penalty(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Minimizes the off-diagonal terms of the covariance matrix 
        to encourage independent receptors.
        """
        B, R = activity.shape
        
        # Center the probabilities
        mean = activity.mean(dim=0, keepdim=True)
        centered = activity - mean
        
        # Compute Covariance Matrix: (R, B) @ (B, R) -> (R, R)
        cov_matrix = (centered.T @ centered) / (B - 1)
        
        # Create a mask to ignore the diagonal (variance)
        mask = ~torch.eye(R, dtype=torch.bool, device=activity.device)
        
        # Sum of squared off-diagonal elements
        return (cov_matrix[mask] ** 2).sum()

    def forward(self, activity: torch.Tensor):
        # 1. Compute Entropy using Differentiable Bins
        marginals = self._compute_soft_histogram_entropy(activity)
        loss_entropy = -marginals.mean() # Maximize entropy
        
        # 2. Compute selected penalty
        if self.penalty_type == 'repulsion':
            penalty = self._compute_repulsion_penalty(activity)
        else: # covariance
            penalty = self._compute_covariance_penalty(activity)
        
        # 3. Total Loss
        total_loss = loss_entropy + (self.cov_weight * penalty)
            
        return total_loss
