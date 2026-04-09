import torch
import itertools
import random
import numpy as np

def exp_distrib(l,beta=0.371177):
    return beta*np.exp(-beta * l )

def generate_receptor_indices(n_units, k_sub, n_sensors):
    """
    Generates the identity of the receptors in our array.
    Since 26^5 is huge, we randomly sample 'n_sensors' unique combinations 
    (with replacement, e.g., AABBB) to simulate the array.
    """
    # 1. Generate all possible combinations (approx 142k for 26 choose 5)
    # combinations_with_replacement handles stoichiometry (AAAAA, AAAAB, etc.)
    all_combos = list(itertools.combinations_with_replacement(range(n_units), k_sub))
    
    # 2. Select a subset to simulate the "Octopus Nose"
    if n_sensors > len(all_combos):
        selected = all_combos
    else:
        selected = random.sample(all_combos, n_sensors)
        
    return torch.tensor(selected, dtype=torch.long)

def generate_targeted_receptors(n_units, k_sub, composition_targets):
    """
    Generates combinations prioritized by their complexity (number of unique subunits).
    
    Args:
        n_units (int): Total number of available sub-units.
        k_sub (int): Number of sub-units in a receptor (e.g., 5).
        composition_targets (dict): Defines how many to draw per complexity level.
                                    Format: {num_unique_units: count}
                                    Example: {1: 'all', 2: 10, 3: 5}
    """
    all_combos = list(itertools.combinations_with_replacement(range(n_units), k_sub))
    
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


def generate_cascading_receptors(n_units, k_sub, n_sensors):
    """
    Automatically prioritizes simpler combinations (homomers > 2-mixes > 3-mixes)
    until the 'n_sensors' quota is reached.
    """
    all_combos = list(itertools.combinations_with_replacement(range(n_units), k_sub))
    
    buckets = {}
    for combo in all_combos:
        n_unique = len(set(combo))
        if n_unique not in buckets:
            buckets[n_unique] = []
        buckets[n_unique].append(combo)
        
    selected_combos = []
    remaining = n_sensors
    
    for n_unique in sorted(buckets.keys()):
        if remaining <= 0:
            break
            
        available = buckets[n_unique]
        # Shuffle within the tier so we don't always bias towards lower subunit indices
        random.shuffle(available) 
        
        take_n = min(remaining, len(available))
        selected_combos.extend(available[:take_n])
        remaining -= take_n
        
    return torch.tensor(selected_combos, dtype=torch.long)

def generate_exp_distributed_receptors(N_receptors, n_units, k_sub):
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
    return generate_targeted_receptors(n_units, k_sub, composition_targets)

def generate_bernoulli_receptors(N_receptors, n_units, k_sub, gene_probs):
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
        expressed_mask = np.random.rand(n_units) < gene_probs
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