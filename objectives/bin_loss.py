import torch
import torch.nn as nn
import math

def compute_shannon_joint_entropy(soft_assign: torch.Tensor) -> torch.Tensor:
    """
    Computes the Shannon joint entropy of a discrete system from soft assignments
    using exact enumeration. Ideal for small state spaces.
    """
    # B : batch size
    # R : # of receptors 
    # K : # of activity bins = 2
    B, R, K = soft_assign.shape 
    
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
    
    return joint_h

def compute_renyi_joint_entropy(soft_assign: torch.Tensor) -> torch.Tensor:
    """
    Computes the Rényi joint entropy (H2) using exact pairwise collision.
    Bypasses the O(K^R) wall entirely by calculating the probability 
    that two random ligands produce the EXACT same array state. 
    """
    B, R, K = soft_assign.shape 
    chunk_size = 2048
    
    if B <= chunk_size:
        S_R = soft_assign.permute(1, 0, 2)
        # Compute all pairs, then multiply along the receptor dimension
        match_probs = torch.prod(torch.bmm(S_R, S_R.permute(0, 2, 1)), dim=0) # (B, B)
        # Remove the diagonal (self-collisions) which artificially inflates the probability
        mask = ~torch.eye(B, dtype=torch.bool, device=soft_assign.device)
        collision_prob = match_probs[mask].mean()
    else:
        # Chunked evaluation: average the collision probability over multiple 
        # independent sub-batches. This reduces estimator variance
        # while keeping autograd memory usage strictly bounded.
        max_chunks = 8 # Process up to 8 chunks (16,384 ligands) per step
        n_chunks = min(B // chunk_size, max_chunks)
        
        indices = torch.randperm(B, device=soft_assign.device)
        total_collision_prob = 0.0
        
        # Calculate cross-chunk collisions (A vs B) to completely avoid self-collisions
        comparisons = 0
        for i in range(n_chunks - 1):
            idx_A = indices[i * chunk_size : (i + 1) * chunk_size]
            idx_B = indices[(i + 1) * chunk_size : (i + 2) * chunk_size]
            
            S_A = soft_assign[idx_A].permute(1, 0, 2) # (R, chunk, K)
            S_B = soft_assign[idx_B].permute(1, 0, 2) # (R, chunk, K)
            
            # BMM yields (R, chunk, chunk), prod yields (chunk, chunk)
            match_probs = torch.prod(torch.bmm(S_A, S_B.permute(0, 2, 1)), dim=0)
            total_collision_prob = total_collision_prob + match_probs.mean()
            comparisons += 1
            
        collision_prob = total_collision_prob / comparisons
    
    # Rényi Entropy of order 2 (in bits)
    collision_prob_safe = torch.clamp(collision_prob, min=1e-12)
    joint_h = -torch.log2(collision_prob_safe)

    return joint_h

class DiscreteExactLoss(nn.Module):
    """
    Maximizes the exact discrete binary joint entropy of the array.
    Ideal for systems where components must be correlated (like a thermometer code).
    """
    def __init__(self, entropy_type: str = 'shannon'):
        super().__init__()
        self.entropy_type = entropy_type

    def compute_soft_assignment(self, activity: torch.Tensor) -> torch.Tensor:
        # For binary systems, activity is exactly P(fire). This avoids vanishing gradients.
        return torch.stack([1.0 - activity, activity], dim=-1)

    def forward(self, activity: torch.Tensor):
        soft_assign = self.compute_soft_assignment(activity)
        if self.entropy_type == 'shannon':
            joint_h = compute_shannon_joint_entropy(soft_assign)
        elif self.entropy_type == 'renyi':
            joint_h = compute_renyi_joint_entropy(soft_assign)
        else:
            raise ValueError(f"Unknown entropy_type: {self.entropy_type}. Choose 'shannon' or 'renyi'.")
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
