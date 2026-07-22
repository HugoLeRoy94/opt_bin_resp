# Documented in:
#   doc/theory/07_optimization_pipeline.md  (stage 4a: DiscreteExactLoss)
#   doc/theory/06_computational_limits.md  (memory scaling of each estimator)
"""
bin_loss.py — Discrete joint entropy estimators for binary receptor arrays.

Estimators selectable via entropy_type in DiscreteExactLoss:
  'shannon'   : exact enumeration, O(B·2^R) — only for R < ~15.
  'collision'  : exact collision H2 (was 'renyi'), O(B²·R) — default scalable choice.
  'kt'        : Kolchinsky-Tracey Bhattacharyya LOWER bound on Shannon H(s),
                O(B²·R) — certified lower bound, tight when components separate.
  'blocked'   : correlation-aware blocked Shannon, O(B·2^block_size + R²).
  'proxy'     : Σ H_r − cov_weight·penalty, O(B·R²) — fastest pairwise approximation.

Blocked estimator: partitions receptors by |Pearson correlation| affinity
(greedy clustering), then computes exact Shannon entropy within each block.
Partition is cached and refreshed every block_refresh_interval training steps.

Collision trick: log P(collision) = Σ_r log P_r(collision) computed in log-space
via logsumexp, diagonal (self-collision) masked out before averaging.
"""
import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint

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


def compute_collision_entropy(
    soft_assign: torch.Tensor,
    return_collision_prob: bool = False,
    collision_chunk_size: int = 2048,
) -> torch.Tensor:
    """Collision entropy H2 = -log2(C) where C is the collision probability.

    Processes the full batch by splitting into chunks of ``collision_chunk_size``
    and accumulating cross-chunk collision log-probabilities via a single
    logsumexp.  The chunk size sets the pairwise-collision window (quadratic
    memory: the (R, m, m) binding matrix); the number of chunks only reduces
    variance (linear memory when looped).

    Honest scope: removing the old silent discard and sizing chunks to GPU
    memory raises the usable-sample wall from ~2^14 toward ~2^17-18 (a few
    more honest bits).  It does NOT remove the log2(collision_chunk_size)
    ceiling on resolvable entropy; that ceiling is set by chunk size (quadratic
    memory), not chunk count (linear).

    When ``return_collision_prob=True`` returns C directly (for use as a
    training loss: minimising C is equivalent to maximising H2 but the
    gradient has no 1/C blow-up at low collision probability).

    Default (``False``) returns H2 in bits.
    """
    B, R, K = soft_assign.shape

    if B <= collision_chunk_size:
        S_R = soft_assign.permute(1, 0, 2)
        match_prob_per_receptor = torch.bmm(S_R, S_R.permute(0, 2, 1))  # (R, B, B)

        log_match_prob_per_receptor = torch.log(match_prob_per_receptor + 1e-40)
        log_match_probs = torch.sum(log_match_prob_per_receptor, dim=0)  # (B, B)

        mask = ~torch.eye(B, dtype=torch.bool, device=soft_assign.device)
        log_match_probs_flat = log_match_probs[mask]

        num_pairs = log_match_probs_flat.numel()
        log_mean_coll_prob_nats = torch.logsumexp(log_match_probs_flat, dim=0) - math.log(num_pairs)

    else:
        n_chunks = math.ceil(B / collision_chunk_size)
        indices = torch.randperm(B, device=soft_assign.device)

        all_log_match_probs = []
        for i in range(n_chunks - 1):
            idx_A = indices[i * collision_chunk_size : (i + 1) * collision_chunk_size]
            idx_B = indices[(i + 1) * collision_chunk_size : (i + 2) * collision_chunk_size]

            S_A = soft_assign[idx_A].permute(1, 0, 2)
            S_B = soft_assign[idx_B].permute(1, 0, 2)

            match_prob_per_receptor = torch.bmm(S_A, S_B.permute(0, 2, 1))
            log_match_prob_per_receptor = torch.log(match_prob_per_receptor + 1e-40)
            log_match_probs = torch.sum(log_match_prob_per_receptor, dim=0)
            all_log_match_probs.append(log_match_probs)

        full_log_match_probs = torch.cat([t.flatten() for t in all_log_match_probs])
        num_pairs = full_log_match_probs.numel()
        log_mean_coll_prob_nats = torch.logsumexp(full_log_match_probs, dim=0) - math.log(num_pairs)

    if return_collision_prob:
        return torch.exp(log_mean_coll_prob_nats)
    return -log_mean_coll_prob_nats / math.log(2)


def _kt_row_contribution(sqrt_A_i, sqrt_1A_i, sqrt_A, sqrt_1A, log_B, chunk_size):
    """Sum over the rows in one i-chunk of ( logsumexp_j logBC[i,:] - log_B ).

    Factored out of compute_kt_entropy so it can be gradient-checkpointed: its
    (m, n, R) Bhattacharyya blocks and (m, B) row are the memory bottleneck, and
    when wrapped in torch.utils.checkpoint they are recomputed in backward instead
    of retained — dropping the retained graph from O(B²·R) to O(B·R).
    sqrt_A_i / sqrt_1A_i are (m, R); sqrt_A / sqrt_1A are the full (B, R).
    """
    B = sqrt_A.shape[0]
    row_blocks = []
    for j0 in range(0, B, chunk_size):
        j1 = min(j0 + chunk_size, B)
        bc = (sqrt_A_i.unsqueeze(1) * sqrt_A[j0:j1].unsqueeze(0)
              + sqrt_1A_i.unsqueeze(1) * sqrt_1A[j0:j1].unsqueeze(0))   # (m, n, R)
        row_blocks.append(torch.log(bc).sum(dim=2))                     # (m, n)
    full_row = torch.cat(row_blocks, dim=1)                            # (m, B)
    return (torch.logsumexp(full_row, dim=1) - log_B).sum()


def compute_kt_entropy(
    soft_assign: torch.Tensor,
    chunk_size: int = 512,
    eps: float = 1e-6,
    recompute: bool = False,
) -> torch.Tensor:
    """Kolchinsky-Tracey Bhattacharyya LOWER bound on the joint Shannon entropy.

    Certified lower bound on the SAME H(s) that compute_shannon_joint_entropy
    computes exactly: the entropy of the uniform B-component mixture
    p(s) = (1/B) Σ_b Π_r Bernoulli(A_br), s ∈ {0,1}^R, A_br = soft_assign[b,r,1].
    Tight in the well-separated / noiseless regime.
    Ref: Kolchinsky & Tracey, "Estimating Mixture Entropy with Pairwise
    Distances", Entropy 2017 (Bhattacharyya-affinity pairwise bound).

    Formula (bits):
      h2(a)  = -a·log2(a) - (1-a)·log2(1-a)            binary Shannon entropy
      H_cond = mean_b Σ_r h2(A_br)                     mean component entropy
      logBC[i,j] = Σ_r log( √(A_i A_j) + √((1-A_i)(1-A_j)) )   (natural log)
      H_lb   = H_cond - (1/B) Σ_i ( logsumexp_j logBC[i,:] - log B ) / log 2

    BC[i,j] is the Bhattacharyya affinity between components i and j;
    BC[i,i]=1, so the diagonal anchors the inner sum at 1/B and the
    resolvable-entropy ceiling is log2(TOTAL batch B) — NOT log2(chunk).

    ALL-PAIRS, DIAGONAL INCLUDED, NORMALIZED BY TOTAL B. The inner j-loop
    spans the ENTIRE batch and the diagonal (logBC=0) is kept. This is NOT
    the adjacent-chunk / diagonal-masked scheme of the collision estimator
    (which caps its ceiling at log2(chunk)); collapsing KT to that would be
    wrong. Peak memory O(chunk²·R); time O(B²·R/chunk), quadratic in B.

    Args:
        soft_assign: (B, R, 2) tensor; channel 1 is A_br, channel 0 is 1-A_br.
        chunk_size:  edge length m of the (m, n) logBC blocks.
        eps:         clamp A into [eps, 1-eps] for log/sqrt stability.
        recompute:   if True, gradient-checkpoint each i-chunk's contribution so its
                     (m, n, R) / (m, B) tensors are recomputed in backward instead of
                     retained. Retained graph drops O(B²·R) → O(B·R), so the training
                     batch is bounded by compute, not memory (see resolve_batch_sizes,
                     KT_COMPUTE_BUDGET). Exact — no effect on value or gradients.
                     Costs ~one extra forward of the inner loop in backward.

    Returns:
        Scalar lower bound in bits, differentiable w.r.t. soft_assign.
    """
    B, R, _ = soft_assign.shape
    A = soft_assign[:, :, 1].clamp(eps, 1.0 - eps)   # (B, R)

    # Conditional (mean per-component) entropy in bits.
    h_cond = (-(A * torch.log2(A) + (1.0 - A) * torch.log2(1.0 - A))).sum(dim=1).mean()

    sqrt_A  = torch.sqrt(A)                            # (B, R)
    sqrt_1A = torch.sqrt(1.0 - A)                      # (B, R)
    log_B = math.log(B)

    # Σ_i ( logsumexp_j logBC[i,:] - log B ), all pairs, diagonal kept.
    # Per i-chunk contribution is optionally gradient-checkpointed (recompute).
    do_ckpt = recompute and soft_assign.requires_grad
    inter_nats = soft_assign.new_zeros(())
    for i0 in range(0, B, chunk_size):
        i1 = min(i0 + chunk_size, B)
        sa_i, s1a_i = sqrt_A[i0:i1], sqrt_1A[i0:i1]   # (m, R)
        if do_ckpt:
            contrib = checkpoint(_kt_row_contribution, sa_i, s1a_i, sqrt_A, sqrt_1A,
                                 log_B, chunk_size, use_reentrant=False)
        else:
            contrib = _kt_row_contribution(sa_i, s1a_i, sqrt_A, sqrt_1A, log_B, chunk_size)
        inter_nats = inter_nats + contrib

    inter_bits = (inter_nats / B) / math.log(2)
    return h_cond - inter_bits


def compute_kt_upper_entropy(
    soft_assign: torch.Tensor,
    chunk_size: int = 512,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Kolchinsky-Tracey UPPER bound on the joint Shannon entropy H(s).

    Companion to compute_kt_entropy (the Bhattacharyya LOWER bound): same
    pairwise-distance structure, but the component affinity uses the KL divergence
    instead of the Bhattacharyya coefficient, which flips the bound direction.

        H(s) <= H_cond - (1/B) Sum_i log( (1/B) Sum_j exp(-KL(p_i || p_j)) )

    For product-Bernoulli components (A_ir = P(receptor r active | sample i)):
        KL(i||j) = Sum_r [ A_ir log(A_ir/A_jr) + (1-A_ir) log((1-A_ir)/(1-A_jr)) ]
                 = ent_i - cross(i,j)
    with the i-only self term  ent_i = Sum_r [A_ir log A_ir + (1-A_ir) log(1-A_ir)]
    and the cross term  cross(i,j) = Sum_r [A_ir log A_jr + (1-A_ir) log(1-A_jr)].
    The affinity kernel is -KL(i||j) = cross(i,j) - ent_i.

    Certified upper bound (never below the true H(s)); tight in the well-separated
    regime. The diagonal (i=j, KL=0) is kept and the sum normalised by the total
    batch B, so the resolvable-entropy ceiling is log2(B) — same as the lower bound.
    Peak memory O(chunk^2 * R); time O(B^2 * R / chunk), quadratic in B.

    Args:
        soft_assign: (B, R, 2); channel 1 is A_ir, channel 0 is 1-A_ir.
        chunk_size:  edge length m of the (m, n) blocks.
        eps:         clamp A into [eps, 1-eps] for log stability.
    """
    B, R, _ = soft_assign.shape
    A = soft_assign[:, :, 1].clamp(eps, 1.0 - eps)            # (B, R)

    h_cond = (-(A * torch.log2(A) + (1.0 - A) * torch.log2(1.0 - A))).sum(dim=1).mean()

    log_A  = torch.log(A)                                     # (B, R)
    log_1A = torch.log1p(-A)                                  # (B, R) = log(1-A)
    ent    = (A * log_A + (1.0 - A) * log_1A).sum(dim=1)      # (B,)  self term (nats)
    log_B  = math.log(B)

    inter_nats = soft_assign.new_zeros(())
    for i0 in range(0, B, chunk_size):
        i1 = min(i0 + chunk_size, B)
        A_i     = A[i0:i1].unsqueeze(1)                       # (m, 1, R)
        one_A_i = (1.0 - A[i0:i1]).unsqueeze(1)               # (m, 1, R)
        ent_i   = ent[i0:i1].unsqueeze(1)                     # (m, 1)
        row_blocks = []
        for j0 in range(0, B, chunk_size):
            j1 = min(j0 + chunk_size, B)
            log_A_j  = log_A[j0:j1].unsqueeze(0)              # (1, n, R)
            log_1A_j = log_1A[j0:j1].unsqueeze(0)             # (1, n, R)
            cross = (A_i * log_A_j + one_A_i * log_1A_j).sum(dim=2)   # (m, n)
            row_blocks.append(cross - ent_i)                 # (m, n) = -KL(i||j)
        full_row = torch.cat(row_blocks, dim=1)              # (m, B)
        inter_nats = inter_nats + (torch.logsumexp(full_row, dim=1) - log_B).sum()

    inter_bits = (inter_nats / B) / math.log(2)
    # H(s) of R binary receptors is <= R bits, so clamp: min of two valid upper
    # bounds is still an upper bound, and tighter in the overlapping regime.
    return (h_cond - inter_bits).clamp(max=float(R))


def compute_correlation_aware_partition(
    activity: torch.Tensor,
    block_size: int = 18,
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
    block_size: int = 18,
    partition: list = None,
    recompute: bool = False,
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
        recompute:   if True, wrap each block's Shannon term in gradient
                     checkpointing so its (B, 2^block_size) joint tensor is
                     recomputed in backward instead of retained. Trades ~one
                     extra forward for a smaller peak → allows a larger B (see
                     resolve_batch_sizes, recompute_backward). No effect on the
                     value or gradients (checkpointing is exact).
    """
    if partition is None:
        activity_detached = soft_assign[:, :, 1].detach()
        partition = compute_correlation_aware_partition(activity_detached, block_size)

    h = soft_assign.new_zeros(())
    for block_idx in partition:
        sa = soft_assign[:, block_idx, :]
        if recompute and sa.requires_grad:
            h = h + checkpoint(compute_shannon_joint_entropy, sa, use_reentrant=False)
        else:
            h = h + compute_shannon_joint_entropy(sa)
    return h


def compute_pairwise_mi(activity: torch.Tensor) -> torch.Tensor:
    """Binary pairwise mutual information matrix I(A_i; A_j) in bits.

    Args:
        activity: (B, R) soft binary activity in [0, 1].

    Returns:
        (R, R) symmetric MI matrix.  Diagonal entries equal H(A_i).
    """
    B = activity.shape[0]
    eps = 1e-12

    marginals = activity.mean(dim=0)                        # (R,)
    joint_11 = (activity.T @ activity) / B                  # (R, R)

    p_1 = marginals.unsqueeze(1)                            # (R, 1)
    p_2 = marginals.unsqueeze(0)                            # (1, R)

    p_11 = joint_11
    p_10 = (p_1 - p_11).clamp(min=eps)
    p_01 = (p_2 - p_11).clamp(min=eps)
    p_00 = (1.0 - p_1 - p_2 + p_11).clamp(min=eps)
    p_11 = p_11.clamp(min=eps)

    m_11 = (p_1 * p_2).clamp(min=eps)
    m_10 = (p_1 * (1 - p_2)).clamp(min=eps)
    m_01 = ((1 - p_1) * p_2).clamp(min=eps)
    m_00 = ((1 - p_1) * (1 - p_2)).clamp(min=eps)

    mi = (p_11 * torch.log2(p_11 / m_11)
          + p_10 * torch.log2(p_10 / m_10)
          + p_01 * torch.log2(p_01 / m_01)
          + p_00 * torch.log2(p_00 / m_00))
    return mi.clamp(min=0.0)


def compute_blocked_corrected_entropy(
    soft_assign: torch.Tensor,
    block_size: int = 18,
    partition: list = None,
    recompute: bool = False,
) -> torch.Tensor:
    """Corrected blocked Shannon: H_blocked - cross-block pairwise MI.

    H_blocked_corrected = H_blocked - Sigma_{(i,j) cross-block} I(A_i; A_j)

    Within-block joint distributions are already exact; this subtracts the
    pairwise redundancy between receptors in *different* blocks that plain
    blocked entropy ignores.  The correction is tight when cross-block
    dependence is dominated by pairwise terms (most environments); only
    higher-order cross-block interactions escape it.

    Gradients flow through both the blocked term (via soft_assign) and the
    MI correction (via the activity column of soft_assign).
    """
    activity_detached = soft_assign[:, :, 1].detach()
    if partition is None:
        partition = compute_correlation_aware_partition(activity_detached, block_size)

    h_blocked = compute_blocked_entropy(soft_assign, block_size, partition, recompute=recompute)

    activity = soft_assign[:, :, 1]
    mi_matrix = compute_pairwise_mi(activity)

    R = soft_assign.shape[1]
    device = soft_assign.device
    block_id = torch.zeros(R, dtype=torch.long, device=device)
    for bid, block_idx in enumerate(partition):
        block_id[block_idx] = bid
    cross_block = block_id.unsqueeze(0) != block_id.unsqueeze(1)
    upper = torch.triu(torch.ones(R, R, dtype=torch.bool, device=device), diagonal=1)
    mi_correction = mi_matrix[cross_block & upper].sum()

    return h_blocked - mi_correction


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
      'shannon'   : exact enumeration         O(B · 2^R)            only for small R
      'collision' : collision H2              O(B² · R)             scalable, good gradients
      'kt'        : Kolchinsky-Tracey lower bound  O(B² · R)        certified H(s) lower bound
      'blocked'   : correlation-aware blocked Shannon  O(B · 2^block_size)
      'proxy'     : Σ H_r − cov_weight·penalty  O(B · R²)           fast pairwise approximation

    Training with one estimator and evaluating with another is supported via
    compute_entropy(activity, entropy_type='blocked').  Typical pattern:

        criterion = DiscreteExactLoss(entropy_type='collision')

        loss        = criterion(activity)                             # training
        h_collision = criterion.compute_entropy(activity)             # collision H2 in bits
        h_blocked   = criterion.compute_entropy(activity, 'blocked') # blocked in bits

    The blocked estimator caches its correlation-aware partition and refreshes
    it every ``block_refresh_interval`` training steps.  Evaluation calls
    should pass ``use_cache=False`` to get a fresh partition for the eval
    batch without touching the training cache.
    """
    _ENTROPY_FNS = ('shannon', 'collision', 'kt', 'blocked', 'blocked_corrected', 'proxy')

    def __init__(
        self,
        entropy_type: str,
        block_size:   int   = 18,
        n_partitions: int   = 4,
        cov_weight:   float = 0.0,
        penalty_type: str   = 'covariance',
        block_refresh_interval: int = 50,
        collision_chunk_size: int = 2048,
        recompute_backward: bool = False,
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
        self.collision_chunk_size = collision_chunk_size
        self.recompute_backward = recompute_backward   # checkpoint blocked histogram
        self._cached_partition = None
        self._steps_since_refresh = 0

    def compute_soft_assignment(self, activity: torch.Tensor) -> torch.Tensor:
        return torch.stack([1.0 - activity, activity], dim=-1)

    def _compute_soft_histogram_entropy(self, activity: torch.Tensor) -> torch.Tensor:
        """Per-receptor marginal binary Shannon entropy, shape (R,) in bits."""
        p = self.compute_soft_assignment(activity).mean(dim=0).clamp(min=1e-12)  # (R, 2)
        return -(p * torch.log2(p)).sum(dim=-1)                                   # (R,)

    def _refresh_partition(self, activity: torch.Tensor, use_cache: bool):
        """Return partition, refreshing the cache if needed."""
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
        return compute_blocked_entropy(soft_assign, self.block_size, partition,
                                       recompute=self.recompute_backward)

    def _get_blocked_corrected_entropy(self, activity: torch.Tensor,
                                       soft_assign: torch.Tensor,
                                       use_cache: bool) -> torch.Tensor:
        partition = self._refresh_partition(activity, use_cache)
        return compute_blocked_corrected_entropy(soft_assign, self.block_size, partition,
                                                 recompute=self.recompute_backward)

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
        elif etype == 'collision':
            return compute_collision_entropy(soft_assign,
                                             collision_chunk_size=self.collision_chunk_size)
        elif etype == 'kt':
            return compute_kt_entropy(soft_assign, chunk_size=self.collision_chunk_size,
                                      recompute=self.recompute_backward)
        elif etype == 'blocked':
            return self._get_blocked_entropy(activity, soft_assign, use_cache)
        elif etype == 'blocked_corrected':
            return self._get_blocked_corrected_entropy(activity, soft_assign, use_cache)
        else:  # proxy
            return compute_proxy_entropy(soft_assign, self.cov_weight, self.penalty_type)

    def forward(self, activity: torch.Tensor) -> torch.Tensor:
        if self.entropy_type == 'collision':
            soft = self.compute_soft_assignment(activity)
            return compute_collision_entropy(soft, return_collision_prob=True,
                                             collision_chunk_size=self.collision_chunk_size)
        return -self.compute_entropy(activity, use_cache=True)


# backward-compat alias
compute_renyi_joint_entropy = compute_collision_entropy


def _kt_reference(soft_assign: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Un-chunked (single dense (B, B) block) KT lower bound — validation ref."""
    B, R, _ = soft_assign.shape
    A = soft_assign[:, :, 1].clamp(eps, 1.0 - eps)
    h_cond = (-(A * torch.log2(A) + (1.0 - A) * torch.log2(1.0 - A))).sum(dim=1).mean()
    sqrt_A, sqrt_1A = torch.sqrt(A), torch.sqrt(1.0 - A)
    bc = (sqrt_A.unsqueeze(1) * sqrt_A.unsqueeze(0)
          + sqrt_1A.unsqueeze(1) * sqrt_1A.unsqueeze(0))          # (B, B, R)
    log_bc = torch.log(bc).sum(dim=2)                             # (B, B)
    inter = (torch.logsumexp(log_bc, dim=1) - math.log(B)).mean() / math.log(2)
    return h_cond - inter


def _exact_mixture_entropy(soft_assign: torch.Tensor) -> torch.Tensor:
    """Exact H(s) by enumerating all 2^R states (small R only)."""
    return compute_shannon_joint_entropy(soft_assign)


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1) chunked vs un-chunked reference agree.
    B, R = 300, 17
    A = torch.rand(B, R)
    sa = torch.stack([1 - A, A], dim=-1)
    h_chunk = compute_kt_entropy(sa, chunk_size=64)
    h_ref = _kt_reference(sa)
    err = (h_chunk - h_ref).abs().item()
    print(f"[1] chunked vs reference: |Δ|={err:.2e}  (chunk={h_chunk:.6f}, ref={h_ref:.6f})")
    assert err < 1e-5, f"chunked mismatch {err}"

    # 2) lower-bound property: KT <= exact H(s) over enumerated states.
    for (bb, rr) in [(64, 12), (32, 10), (48, 8)]:
        A2 = torch.rand(bb, rr)
        sa2 = torch.stack([1 - A2, A2], dim=-1)
        h_kt = compute_kt_entropy(sa2, chunk_size=16).item()
        h_ex = _exact_mixture_entropy(sa2).item()
        print(f"[2] B={bb} R={rr}: H_kt={h_kt:.4f} <= H_exact={h_ex:.4f}")
        assert h_kt <= h_ex + 1e-4, f"lower-bound violated: {h_kt} > {h_ex}"

    # 3a) B distinct near-deterministic codewords -> H_kt ≈ log2(B) (+ small H_cond).
    Bd, Rd = 32, 12
    codes = (torch.rand(Bd, Rd) > 0.5).float()
    A3 = codes * (1 - 1e-3) + (1 - codes) * 1e-3      # near-deterministic distinct
    sa3 = torch.stack([1 - A3, A3], dim=-1)
    h3 = compute_kt_entropy(sa3, chunk_size=8).item()
    # Well-separated: inter-term ≈ -log2(B), so H_kt ≈ log2(B) + small H_cond.
    h_cond3 = (-(A3 * torch.log2(A3) + (1 - A3) * torch.log2(1 - A3))).sum(1).mean().item()
    print(f"[3a] distinct codewords: H_kt={h3:.4f}  log2(B)+H_cond={math.log2(Bd)+h_cond3:.4f}")
    assert abs(h3 - (math.log2(Bd) + h_cond3)) < 0.01, f"expected ≈log2(B)+H_cond, got {h3}"

    # 3b) B identical sniffs at A=0.5 -> inter-term = 0 -> H_kt == R.
    Bi, Ri = 40, 9
    A4 = torch.full((Bi, Ri), 0.5)
    sa4 = torch.stack([1 - A4, A4], dim=-1)
    h4 = compute_kt_entropy(sa4, chunk_size=8).item()
    print(f"[3b] identical A=0.5: H_kt={h4:.4f}  R={Ri}")
    assert abs(h4 - Ri) < 1e-4, f"expected R={Ri}, got {h4}"

    # 4) finite-difference gradcheck (float64) through compute_soft_assignment.
    loss_mod = DiscreteExactLoss(entropy_type='kt', collision_chunk_size=8)

    def _kt_from_activity(act):
        sa_ = loss_mod.compute_soft_assignment(act)
        return compute_kt_entropy(sa_, chunk_size=8)

    act0 = torch.rand(20, 6, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(_kt_from_activity, (act0,), eps=1e-6, atol=1e-4)
    print(f"[4] gradcheck through compute_soft_assignment: {ok}")
    assert ok

    print("all KT self-tests passed.")
