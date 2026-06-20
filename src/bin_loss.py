# Documented in:
#   doc/theory/07_optimization_pipeline.md  (stage 4a: DiscreteExactLoss)
#   doc/theory/06_computational_limits.md  (memory scaling of each estimator)
"""
bin_loss.py — Discrete joint entropy estimators for binary receptor arrays.

Four estimators selectable via entropy_type in DiscreteExactLoss:
  'shannon' : exact enumeration, O(B·2^R) — only for R < ~15.
  'renyi'   : exact Rényi H2, O(B²·R) — default scalable choice.
  'blocked' : correlation-aware blocked Shannon, O(B·2^block_size + R²).
  'proxy'   : Σ H_r − cov_weight·penalty, O(B·R²) — fastest pairwise approximation.

Blocked estimator: partitions receptors by |Pearson correlation| affinity
(greedy clustering), then computes exact Shannon entropy within each block.
Partition is cached and refreshed every block_refresh_interval training steps.

Rényi trick: log P(collision) = Σ_r log P_r(collision) computed in log-space
via logsumexp, diagonal (self-collision) masked out before averaging.
"""
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
        match_prob_per_receptor = torch.bmm(S_R, S_R.permute(0, 2, 1)) # (R, B, B)

        # Switch to log-space to avoid numerical underflow from product over R
        # This is the log-sum-exp trick for stable computation of log(mean(prod(P)))
        # Add a tiny epsilon to prevent log(0)
        log_match_prob_per_receptor = torch.log(match_prob_per_receptor + 1e-40)
        log_match_probs = torch.sum(log_match_prob_per_receptor, dim=0) # (B, B)

        # Remove the diagonal (self-collisions) which artificially inflates the probability
        mask = ~torch.eye(B, dtype=torch.bool, device=soft_assign.device)
        log_match_probs_flat = log_match_probs[mask]

        # H2 = -log2(mean(exp(log_probs)))
        #    = -[ log(mean(exp(log_probs))) / log(2) ]
        #    = -[ (logsumexp(log_probs) - log(N)) / log(2) ]
        num_pairs = log_match_probs_flat.numel()
        log_mean_coll_prob_nats = torch.logsumexp(log_match_probs_flat, dim=0) - math.log(num_pairs)
        joint_h = -log_mean_coll_prob_nats / math.log(2)

    else:
        # Chunked evaluation: average the collision probability over multiple
        # independent sub-batches. This reduces estimator variance
        # while keeping autograd memory usage strictly bounded.
        max_chunks = 8 # Process up to 8 chunks (16,384 ligands) per step
        n_chunks = min(math.ceil(B / chunk_size), max_chunks)
        indices = torch.randperm(B, device=soft_assign.device)

        # Collect all cross-chunk log-probabilities to average them correctly at the end
        all_log_match_probs = []

        # Calculate cross-chunk collisions (A vs B) to completely avoid self-collisions
        for i in range(n_chunks - 1):
            idx_A = indices[i * chunk_size : (i + 1) * chunk_size]
            idx_B = indices[(i + 1) * chunk_size : (i + 2) * chunk_size]

            S_A = soft_assign[idx_A].permute(1, 0, 2) # (R, chunk, K)
            S_B = soft_assign[idx_B].permute(1, 0, 2) # (R, chunk, K)

            match_prob_per_receptor = torch.bmm(S_A, S_B.permute(0, 2, 1))
            log_match_prob_per_receptor = torch.log(match_prob_per_receptor + 1e-40)
            log_match_probs = torch.sum(log_match_prob_per_receptor, dim=0)
            all_log_match_probs.append(log_match_probs)

        full_log_match_probs = torch.cat([t.flatten() for t in all_log_match_probs])
        num_pairs = full_log_match_probs.numel()
        log_mean_coll_prob_nats = torch.logsumexp(full_log_match_probs, dim=0) - math.log(num_pairs)
        joint_h = -log_mean_coll_prob_nats / math.log(2)

    return joint_h


def compute_correlation_aware_partition(
    activity: torch.Tensor,
    block_size: int = 15,
) -> list:
    """Partition receptors into blocks that group correlated receptors together.

    Uses |Pearson correlation| as an affinity measure computed via a single
    matmul on batch-centered activity.  Greedy grouping: each block is seeded
    with the highest-affinity remaining pair, then grown by picking the
    receptor with the maximum affinity to the current block until the block
    reaches ``block_size``.  The procedure repeats until all receptors are
    assigned.

    Args:
        activity:   (B, R) tensor, already detached (stop-grad).
        block_size: maximum number of receptors per block.

    Returns:
        List of 1-D ``LongTensor`` index vectors, one per block.
    """
    B, R = activity.shape
    device = activity.device

    centered = activity - activity.mean(dim=0, keepdim=True)
    std = centered.norm(dim=0).clamp(min=1e-12)           # (R,)
    corr = (centered.T @ centered) / (B - 1)              # (R, R)
    corr = corr / (std.unsqueeze(1) * std.unsqueeze(0))   # normalise to Pearson
    affinity = corr.abs()
    affinity.fill_diagonal_(0.0)

    remaining = set(range(R))
    blocks: list = []

    while remaining:
        if len(remaining) <= block_size:
            blocks.append(torch.tensor(sorted(remaining), dtype=torch.long, device=device))
            break

        rem_list = sorted(remaining)
        rem_t = torch.tensor(rem_list, dtype=torch.long, device=device)
        sub_aff = affinity[rem_t][:, rem_t]                # (|rem|, |rem|)
        best = sub_aff.argmax().item()
        i_local, j_local = best // len(rem_list), best % len(rem_list)
        block = [rem_list[i_local], rem_list[j_local]]
        remaining.discard(block[0])
        remaining.discard(block[1])

        while len(block) < block_size and remaining:
            block_t = torch.tensor(block, dtype=torch.long, device=device)
            rem_list_inner = sorted(remaining)
            rem_t_inner = torch.tensor(rem_list_inner, dtype=torch.long, device=device)
            aff_to_block = affinity[rem_t_inner][:, block_t].sum(dim=1)  # (|rem|,)
            best_idx = aff_to_block.argmax().item()
            chosen = rem_list_inner[best_idx]
            block.append(chosen)
            remaining.discard(chosen)

        blocks.append(torch.tensor(block, dtype=torch.long, device=device))

    return blocks


def compute_blocked_entropy(
    soft_assign: torch.Tensor,
    block_size: int = 15,
    partition: list = None,
) -> torch.Tensor:
    """Approximates the joint Shannon entropy via correlation-aware blocking.

    Partitions receptors into blocks that cluster correlated receptors
    together, computes exact Shannon entropy within each block, and sums
    under a between-block independence assumption:

        H_blocked = Σ_k H_exact(block_k)

    Correlated receptors sharing a block means the within-block entropy
    already accounts for their joint distribution, yielding a tighter upper
    bound than random partitioning.  Cross-block correlations are the only
    source of error, and that error vanishes at the entropy-maximizing
    solution (independent receptors).

    The partition is computed from the detached activity (no gradient) so
    the grouping does not contribute to the computational graph; gradients
    flow only through the within-block Shannon entropy terms.

    Memory: O(B * 2^block_size) per block.

    Args:
        soft_assign: (B, R, K) soft-assignment tensor.
        block_size:  max receptors per block.
        partition:   optional precomputed partition (list of LongTensor).
                     When ``None`` a fresh correlation-aware partition is
                     built from ``soft_assign[:, :, 1]`` (the activity).
    """
    if partition is None:
        activity_detached = soft_assign[:, :, 1].detach()
        partition = compute_correlation_aware_partition(activity_detached, block_size)

    h = soft_assign.new_zeros(())
    for block_idx in partition:
        h = h + compute_shannon_joint_entropy(soft_assign[:, block_idx, :])
    return h


def compute_proxy_entropy(
    soft_assign: torch.Tensor,
    cov_weight: float,
    penalty_type: str,
) -> torch.Tensor:
    """
    Proxy approximation to the joint Shannon entropy:

        H_proxy = Σ_r H(X_r) − cov_weight · penalty

    The sum of marginal entropies (Σ H_r) equals the joint entropy when receptors
    are independent, and exceeds it otherwise. The penalty term corrects for
    correlation, pushing toward independence during optimisation.

    penalty_type options:
      'covariance' — Σ_{i≠j} Cov(X_i, X_j)²   penalises linear correlation
      'repulsion'  — Σ_{i≠j} exp(−‖A_i−A_j‖²/τ)  penalises similar profiles

    O(B·R²), fully differentiable, no exponential memory.
    The return value is on the same scale as the other entropy functions (up to R bits).
    """
    B, R, _ = soft_assign.shape
    activity = soft_assign[:, :, 1]  # (B, R)

    p = soft_assign.mean(dim=0).clamp(min=1e-12)          # (R, 2)
    h_marginals = -(p * torch.log2(p)).sum()               # scalar, sum over R receptors

    if penalty_type == 'covariance':
        centered = activity - activity.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / (B - 1)            # (R, R)
        mask = ~torch.eye(R, dtype=torch.bool, device=soft_assign.device)
        penalty = (cov[mask] ** 2).sum()
    elif penalty_type == 'repulsion':
        A    = activity.T                                   # (R, B)
        A_sq = (A ** 2).sum(dim=1, keepdim=True)
        dist = (A_sq + A_sq.T - 2.0 * (A @ A.T)) / B      # normalised squared distance (R, R)
        mask = ~torch.eye(R, dtype=torch.bool, device=soft_assign.device)
        penalty = torch.exp(-dist / 0.05)[mask].sum()
    else:
        raise ValueError(f"Unknown penalty_type: {penalty_type!r}. Choose 'covariance' or 'repulsion'.")

    return h_marginals - cov_weight * penalty


class DiscreteExactLoss(nn.Module):
    """
    Maximizes the discrete binary joint entropy of the receptor array.

    entropy_type selects the estimator used for training:
      'shannon' : exact enumeration         O(B · 2^R)            only for small R
      'renyi'   : exact Rényi H2            O(B² · R)             scalable, good gradients
      'blocked' : correlation-aware blocked Shannon  O(B · 2^block_size)
      'proxy'   : Σ H_r − cov_weight·penalty  O(B · R²)           fast pairwise approximation

    Training with one estimator and evaluating with another is supported via
    compute_entropy(activity, entropy_type='blocked').  Typical pattern:

        criterion = DiscreteExactLoss(entropy_type='renyi')

        loss      = criterion(activity)                          # training
        h_renyi   = criterion.compute_entropy(activity)          # Rényi in bits
        h_blocked = criterion.compute_entropy(activity, 'blocked')  # blocked in bits

    The blocked estimator caches its correlation-aware partition and refreshes
    it every ``block_refresh_interval`` training steps.  Evaluation calls
    should pass ``use_cache=False`` to get a fresh partition for the eval
    batch without touching the training cache.
    """
    _ENTROPY_FNS = ('shannon', 'renyi', 'blocked', 'proxy')

    def __init__(
        self,
        entropy_type: str,
        block_size:   int   = 15,
        n_partitions: int   = 4,
        cov_weight:   float = 0.0,
        penalty_type: str   = 'covariance',
        block_refresh_interval: int = 50,
    ):
        super().__init__()
        if entropy_type not in self._ENTROPY_FNS:
            raise ValueError(f"Unknown entropy_type: {entropy_type!r}. Choose from {self._ENTROPY_FNS}.")
        if penalty_type not in ('covariance', 'repulsion'):
            raise ValueError(f"Unknown penalty_type: {penalty_type!r}. Choose 'covariance' or 'repulsion'.")
        self.entropy_type = entropy_type
        self.block_size   = block_size
        self.n_partitions = n_partitions
        self.cov_weight   = cov_weight
        self.penalty_type = penalty_type
        self.block_refresh_interval = block_refresh_interval
        self._cached_partition = None
        self._steps_since_refresh = 0

    def compute_soft_assignment(self, activity: torch.Tensor) -> torch.Tensor:
        return torch.stack([1.0 - activity, activity], dim=-1)

    def _compute_soft_histogram_entropy(self, activity: torch.Tensor) -> torch.Tensor:
        """Per-receptor marginal binary Shannon entropy, shape (R,) in bits."""
        p = self.compute_soft_assignment(activity).mean(dim=0).clamp(min=1e-12)  # (R, 2)
        return -(p * torch.log2(p)).sum(dim=-1)                                   # (R,)

    def _get_blocked_entropy(self, activity: torch.Tensor, soft_assign: torch.Tensor,
                             use_cache: bool) -> torch.Tensor:
        """Blocked entropy with optional partition caching."""
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

    def compute_entropy(self, activity: torch.Tensor, entropy_type: str = None,
                        use_cache: bool = True) -> torch.Tensor:
        """
        Returns the joint entropy in bits (positive scalar).

        Args:
            activity:     Soft binary activity, shape (B, R).
            entropy_type: Override the instance entropy_type for this call.
                          Useful for computing a more accurate estimate at eval time.
            use_cache:    If True (default), use/maintain the cached partition for
                          the blocked estimator.  Eval callers should pass False.
        """
        etype = entropy_type if entropy_type is not None else self.entropy_type
        if etype not in self._ENTROPY_FNS:
            raise ValueError(f"Unknown entropy_type: {etype!r}. Choose from {self._ENTROPY_FNS}.")

        soft_assign = self.compute_soft_assignment(activity)
        if etype == 'shannon':
            return compute_shannon_joint_entropy(soft_assign)
        elif etype == 'renyi':
            return compute_renyi_joint_entropy(soft_assign)
        elif etype == 'blocked':
            return self._get_blocked_entropy(activity, soft_assign, use_cache)
        else:  # proxy
            return compute_proxy_entropy(soft_assign, self.cov_weight, self.penalty_type)

    def forward(self, activity: torch.Tensor) -> torch.Tensor:
        return -self.compute_entropy(activity, use_cache=True)
