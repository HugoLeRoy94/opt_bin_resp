# Documented in:
#   doc/theory/07_optimization_pipeline.md  (receptor sampling strategies section)
"""
geometry.py — Receptor array composition strategies.

Provides several functions for selecting which heteromeric combinations populate
the receptor array:
  generate_receptor_indices         — random sample from all combinations_with_replacement
  generate_targeted_receptors       — explicit count per complexity level (n_unique_subunits)
  generate_cascading_receptors      — fill quota by complexity: homomers first, then 2-mers, …
  generate_exp_distributed_receptors — complexity drawn from exponential distribution
  generate_bernoulli_receptors      — each gene present via Bernoulli(gene_probs)
  build_heteromer_array             — unified entry-point: uniform_random or cascading, seeded
"""
import torch
import itertools
import random
import numpy as np
from typing import Optional

def exp_distrib(l,beta=0.371177):
    return beta*np.exp(-beta * l )

# ---------------------------------------------------------------------------
# Interface-model helpers: ordered cyclic equivalence classes
# ---------------------------------------------------------------------------

def _canonical_rotation(seq: tuple) -> tuple:
    """
    Canonical representative of a cyclic equivalence class.

    Returns the lexicographically minimum rotation of `seq`.  Two tuples that
    are cyclic rotations of each other map to the same canonical form, while
    reflections (reversed order) map to different canonical forms — which is
    the correct semantics for the interface model, where the (+/−) face
    asymmetry makes clockwise ≠ counter-clockwise.
    """
    k = len(seq)
    return min(seq[i:] + seq[:i] for i in range(k))


def generate_ordered_receptor_indices(n_genes: int, k_sub: int, n_sensors: int) -> torch.Tensor:
    """
    Interface-model variant of generate_receptor_indices.

    Generates ordered ring arrangements (cyclic equivalence classes).
    Two arrangements are the same receptor iff one is a cyclic rotation of the
    other.  Reflections are DISTINCT because the +/− face asymmetry breaks
    mirror symmetry.

    Algorithm:
      1. Enumerate all multisets via combinations_with_replacement.
      2. For each multiset, generate all distinct permutations.
      3. Canonicalise each permutation (minimum rotation) and deduplicate.
      4. Random-sample n_sensors from the resulting set.

    Cost: at most 142 506 × 120 = ~17 M operations for n_genes=26, k_sub=5
    (feasible at init time; not on the hot path).

    Returns:
        (n_sensors, k_sub) long tensor of canonical ring arrangements.
    """
    all_combos = list(itertools.combinations_with_replacement(range(n_genes), k_sub))
    canonical_set: set = set()
    for combo in all_combos:
        for perm in set(itertools.permutations(combo)):
            canonical_set.add(_canonical_rotation(perm))
    all_ordered = list(canonical_set)

    if n_sensors > len(all_ordered):
        selected = all_ordered
    else:
        selected = random.sample(all_ordered, n_sensors)
    return torch.tensor(selected, dtype=torch.long)


def generate_targeted_ordered_receptors(n_genes: int, k_sub: int, composition_targets: dict) -> torch.Tensor:
    """
    Interface-model variant of generate_targeted_receptors.

    Generates ordered cyclic arrangements bucketed by number of unique subunits,
    sampling the requested count from each bucket.

    Args:
        composition_targets: {num_unique_units: count | 'all'}
    """
    all_combos = list(itertools.combinations_with_replacement(range(n_genes), k_sub))

    # Bucket canonical forms by complexity
    buckets: dict = {}
    for combo in all_combos:
        for perm in set(itertools.permutations(combo)):
            canon = _canonical_rotation(perm)
            n_unique = len(set(canon))
            buckets.setdefault(n_unique, set()).add(canon)

    selected_combos = []
    for n_unique in sorted(composition_targets.keys()):
        target_k = composition_targets[n_unique]
        if n_unique not in buckets:
            continue
        available = list(buckets[n_unique])
        if target_k == 'all' or target_k >= len(available):
            selected_combos.extend(available)
        else:
            selected_combos.extend(random.sample(available, target_k))

    return torch.tensor(selected_combos, dtype=torch.long)


def generate_cascading_ordered_receptors(n_genes: int, k_sub: int, n_sensors: int) -> torch.Tensor:
    """
    Interface-model variant of generate_cascading_receptors.

    Fills the quota ascending by complexity (homomers → 2-mers → …) using
    ordered cyclic arrangements.
    """
    all_combos = list(itertools.combinations_with_replacement(range(n_genes), k_sub))

    buckets: dict = {}
    for combo in all_combos:
        for perm in set(itertools.permutations(combo)):
            canon = _canonical_rotation(perm)
            n_unique = len(set(canon))
            buckets.setdefault(n_unique, set()).add(canon)

    selected_combos = []
    remaining = n_sensors
    for n_unique in sorted(buckets.keys()):
        if remaining <= 0:
            break
        available = list(buckets[n_unique])
        random.shuffle(available)
        take_n = min(remaining, len(available))
        selected_combos.extend(available[:take_n])
        remaining -= take_n

    return torch.tensor(selected_combos, dtype=torch.long)

def generate_receptor_indices(n_genes, k_sub, n_sensors):
    """
    Generates the identity of the receptors in our array.
    Since 26^5 is huge, we randomly sample 'n_sensors' unique combinations 
    (with replacement, e.g., AABBB) to simulate the array.
    """
    # 1. Generate all possible combinations (approx 142k for 26 choose 5)
    # combinations_with_replacement handles stoichiometry (AAAAA, AAAAB, etc.)
    all_combos = list(itertools.combinations_with_replacement(range(n_genes), k_sub))
    
    # 2. Select a subset to simulate the "Octopus Nose"
    if n_sensors > len(all_combos):
        selected = all_combos
    else:
        selected = random.sample(all_combos, n_sensors)
        
    return torch.tensor(selected, dtype=torch.long)

def generate_targeted_receptors(n_genes, k_sub, composition_targets):
    """
    Generates combinations prioritized by their complexity (number of unique subunits).
    
    Args:
        n_genes (int): Total number of available sub-units.
        k_sub (int): Number of sub-units in a receptor (e.g., 5).
        composition_targets (dict): Defines how many to draw per complexity level.
                                    Format: {num_unique_units: count}
                                    Example: {1: 'all', 2: 10, 3: 5}
    """
    all_combos = list(itertools.combinations_with_replacement(range(n_genes), k_sub))
    
    # 1. Bucket the combinations by the number of unique subunits
    buckets = {}
    for combo in all_combos:
        n_unique = len(set(combo))
        if n_unique not in buckets:
            buckets[n_unique] = []
        buckets[n_unique].append(combo)
        
    selected_combos = []
    
    # 2. Iterate through targets in sorted order to maintain priority in the final tensor
    for n_unique in sorted(composition_targets.keys()):
        target_k = composition_targets[n_unique]
        
        if n_unique not in buckets:
            continue # Skip if this complexity is mathematically impossible 
            
        available = buckets[n_unique]
        
        if target_k == 'all' or target_k >= len(available):
            # Take all available combinations in this bucket
            selected_combos.extend(available)
        else:
            # Randomly sample 'target_k' combinations
            selected_combos.extend(random.sample(available, target_k))
            
    return torch.tensor(selected_combos, dtype=torch.long)


def generate_cascading_receptors(n_genes, k_sub, n_sensors, seed: Optional[int] = None):
    """
    Automatically prioritizes simpler combinations (homomers > 2-mixes > 3-mixes)
    until the 'n_sensors' quota is reached.

    Args:
        seed: When provided, the intra-tier shuffle is deterministic.
              Same (n_genes, k_sub, n_sensors, seed) → identical tensor.
    """
    all_combos = list(itertools.combinations_with_replacement(range(n_genes), k_sub))

    buckets = {}
    for combo in all_combos:
        n_unique = len(set(combo))
        if n_unique not in buckets:
            buckets[n_unique] = []
        buckets[n_unique].append(combo)

    rng = random.Random(seed)
    selected_combos = []
    remaining = n_sensors

    for n_unique in sorted(buckets.keys()):
        if remaining <= 0:
            break

        available = buckets[n_unique]
        # Shuffle within the tier so we don't always bias towards lower subunit indices
        rng.shuffle(available)

        take_n = min(remaining, len(available))
        selected_combos.extend(available[:take_n])
        remaining -= take_n

    return torch.tensor(selected_combos, dtype=torch.long)


def build_heteromer_array(
    n_genes: int,
    k_sub: int,
    R_target: int,
    strategy: str = "uniform_random",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Unified entry-point for building a heteromer receptor array.

    Returns a (R_target, k_sub) long tensor of unit indices.  When R_target
    exceeds the combinatorial pool the full pool is returned (with a warning).

    Args:
        n_genes:   Number of available gene units.
        k_sub:     Subunits per receptor.
        R_target:  Desired number of receptors.
        strategy:  "uniform_random" — reservoir-sample from the full
                       combinations_with_replacement pool (O(R_target) memory).
                   "cascading" — fill by complexity tier: homomers first,
                       then 2-mers, etc. (delegates to generate_cascading_receptors).
        seed:      RNG seed for reproducibility.
                   Same arguments + seed → identical output tensor.
    """
    if strategy == "cascading":
        return generate_cascading_receptors(n_genes, k_sub, R_target, seed=seed)

    if strategy == "uniform_random":
        rng = random.Random(seed)
        reservoir: list = []
        for i, combo in enumerate(
            itertools.combinations_with_replacement(range(n_genes), k_sub)
        ):
            if i < R_target:
                reservoir.append(combo)
            else:
                j = rng.randint(0, i)
                if j < R_target:
                    reservoir[j] = combo

        if len(reservoir) < R_target:
            import warnings
            pool_size = len(reservoir)
            warnings.warn(
                f"build_heteromer_array: R_target={R_target} exceeds the full pool "
                f"(C({n_genes}+{k_sub}-1,{k_sub}) = {pool_size}). "
                f"Returning all {pool_size} combinations.",
                stacklevel=2,
            )

        return torch.tensor(reservoir, dtype=torch.long)

    raise ValueError(
        f"Unknown strategy {strategy!r}. Choose 'uniform_random' or 'cascading'."
    )

def generate_exp_distributed_receptors(N_receptors, n_genes, k_sub):
    """
    Generates a list of receptor combinations where the number of unique genes
    (subunits) in each combination follows the exponential distribution.
    """
    # 1. Calculate probabilities for each possible number of unique genes (1 to k_sub)
    l_values = np.arange(1, k_sub + 1)
    probs = exp_distrib(l_values)
    probs = probs / np.sum(probs)  # Normalize probabilities
    
    # 2. Sample the number of unique genes for N_receptors based on the distribution
    sampled_genes = np.random.choice(l_values, size=N_receptors, p=probs)
    unique, counts = np.unique(sampled_genes, return_counts=True)
    composition_targets = {int(k): int(v) for k, v in zip(unique, counts)}
    
    # 3. Use the existing targeted generator to fetch the combinations
    return generate_targeted_receptors(n_genes, k_sub, composition_targets)

def generate_bernoulli_receptors(N_receptors, n_genes, k_sub, gene_probs):
    """
    Generates combinations (cells) where each gene's presence is determined by an 
    independent Bernoulli trial using `gene_probs`. 
    """
    selected_combos = set()
    combos_list = []
    gene_probs = np.asarray(gene_probs)
    
    retries = 0
    max_retries = 50 # Limit retries to prevent infinite loops and limit sampling bias
    
    while len(combos_list) < N_receptors:
        # 1. Determine which genes are expressed in this cell via Bernoulli trials
        expressed_mask = np.random.rand(n_genes) < gene_probs
        expressed_genes = np.where(expressed_mask)[0]
        n_expressed = len(expressed_genes)
        
        # Constraint 1: Must express at least 1 gene, and cannot express more than k_sub
        if n_expressed == 0 or n_expressed > k_sub:
            continue
            
        # 2. Form a combo of length k_sub using EXACTLY the expressed genes
        combo = list(expressed_genes)
        if k_sub > n_expressed:
            # Fill remaining slots weighted by the relative expression probabilities of the active genes
            relative_probs = gene_probs[expressed_genes]
            relative_probs = relative_probs / np.sum(relative_probs)
            
            remaining_slots = k_sub - n_expressed
            combo.extend(np.random.choice(expressed_genes, size=remaining_slots, p=relative_probs))
            
        # Sort to match the standard tuple format (e.g. non-decreasing: [0, 1, 1, 1, 1])
        combo = tuple(sorted(combo))
        
        # Constraint 2: Try to avoid duplicates, but accept if we're struggling to find novel ones
        if combo in selected_combos and retries < max_retries:
            retries += 1
            continue
            
        # Accept the combination
        selected_combos.add(combo)
        combos_list.append(combo)
        retries = 0 # Reset for the next combo
        
    return torch.tensor(combos_list, dtype=torch.long)