"""Generators for cell gene sets and receptor lists. No config, no CLI, no runner.

This is the ONLY shared code under tasks/cells/. Everything else — every parameter
of every experiment — is written out in the script that runs it. If you are looking
for what a run did, open its script; nothing is hidden here except the combinatorics
of picking which genes land in which cell.

Everything below is pure: same arguments, same output, no global state.
"""
import math
import random

import torch

from src.geometry import _canonical_rotation

__all__ = ["complete_coverage_gene_sets", "complete_coverage_gene_sets_mixed",
           "random_gene_sets", "homomers_only",
           "receptor_pool", "shared_subsets",
           "count_alphabet", "heterogeneous_conc_means"]


def complete_coverage_gene_sets(n_genes, n_cells, genes_per_cell, seed):
    """Random gene sets, but every gene is guaranteed to appear in at least one cell.

    Each cell gets one mandatory ANCHOR gene, dealt round-robin over the genes so
    multiplicities are near-balanced, then fills the rest by random priority. This
    needs cells to know what other cells expressed, which a developing tissue cannot
    do — it is a controlled baseline, not a biological model. Use random_gene_sets
    for the independent version.

    Reusing one seed across genes_per_cell makes the sets NESTED: the g=2 array is
    the g=1 array with one gene added per cell. That is what makes an expression
    sweep a sweep rather than five unrelated arrays.

    Returns a tuple of n_cells sorted gene tuples.
    """
    if not 1 <= genes_per_cell <= n_genes:
        raise ValueError(f"Need 1 <= genes_per_cell <= n_genes, got {genes_per_cell}/{n_genes}.")
    if n_cells < n_genes:
        raise ValueError(f"Complete coverage needs n_cells >= n_genes, got {n_cells} < {n_genes}.")
    generator = torch.Generator().manual_seed(seed)
    priority = torch.rand(n_cells, n_genes, generator=generator)
    permutation = torch.randperm(n_genes, generator=generator)
    anchors = permutation[torch.arange(n_cells) % n_genes]
    anchors = anchors[torch.randperm(n_cells, generator=generator)]
    priority.scatter_(1, anchors[:, None], -1.)
    genes = priority.topk(genes_per_cell, dim=1, largest=False).indices.sort(dim=1).values
    return tuple(map(tuple, genes.tolist()))


def complete_coverage_gene_sets_mixed(n_genes, n_cells, mean, seed):
    """Complete coverage at a NON-INTEGER mean genes per cell.

    complete_coverage_gene_sets gives every cell exactly `genes_per_cell` genes, so
    the mean is forced to an integer and the covering arm cannot share an x axis with
    a continuously-parameterised one. Here `n_big = round(frac * n_cells)` cells get
    floor(mean)+1 genes and the rest get floor(mean).

    Coverage survives because the anchor mechanism is untouched: every cell still
    gets one mandatory gene dealt round-robin, so with n_cells >= n_genes every gene
    is expressed somewhere whatever the sizes are. Which cells get the extra gene is
    itself shuffled, so the big cells are not systematically the ones holding any
    particular anchor.

    Returns (gene_sets, realised_mean). The realised mean is exact when frac*n_cells
    is a whole number and rounds otherwise, so plot the realised value, not the
    target. At integer `mean` this reduces exactly to complete_coverage_gene_sets.
    """
    if not 1 <= mean <= n_genes:
        raise ValueError(f"mean must lie in [1, {n_genes}], got {mean}.")
    if n_cells < n_genes:
        raise ValueError(f"Complete coverage needs n_cells >= n_genes, got {n_cells} < {n_genes}.")
    base = int(math.floor(mean + 1e-9))
    n_big = int(round((mean - base) * n_cells))
    if n_big == 0:
        return complete_coverage_gene_sets(n_genes, n_cells, base, seed), float(base)
    if base + 1 > n_genes:
        raise ValueError(f"mean {mean} needs {base + 1} genes but only {n_genes} exist.")

    generator = torch.Generator().manual_seed(seed)
    priority = torch.rand(n_cells, n_genes, generator=generator)
    permutation = torch.randperm(n_genes, generator=generator)
    anchors = permutation[torch.arange(n_cells) % n_genes]
    anchors = anchors[torch.randperm(n_cells, generator=generator)]
    priority.scatter_(1, anchors[:, None], -1.)

    sizes = torch.full((n_cells,), base, dtype=torch.long)
    sizes[torch.randperm(n_cells, generator=generator)[:n_big]] = base + 1
    gene_sets = tuple(
        tuple(sorted(priority[c].topk(int(sizes[c]), largest=False).indices.tolist()))
        for c in range(n_cells))
    return gene_sets, sum(map(len, gene_sets)) / n_cells


def random_gene_sets(n_genes, n_cells, genes_per_cell, seed):
    """Independent gene sets: no anchor, so a gene can end up expressed nowhere.

    Same nesting property as complete_coverage_gene_sets when the seed is reused.
    """
    if not 1 <= genes_per_cell <= n_genes:
        raise ValueError(f"Need 1 <= genes_per_cell <= n_genes, got {genes_per_cell}/{n_genes}.")
    generator = torch.Generator().manual_seed(seed)
    priority = torch.rand(n_cells, n_genes, generator=generator)
    genes = priority.topk(genes_per_cell, dim=1, largest=False).indices.sort(dim=1).values
    return tuple(map(tuple, genes.tolist()))


def homomers_only(gene_sets, k_sub):
    """Turn gene sets into explicit receptor lists containing only homomers.

    A cell expressing {u, v} gets [(u,)*k_sub, (v,)*k_sub] and nothing else. Pass the
    result as RunConfig(cell_receptors=...), which bypasses repertoire expansion and
    weights the listed receptors uniformly.

    The gene-set path CANNOT express this: expressing u and v necessarily also builds
    every mixed receptor they allow.
    """
    return tuple(tuple((gene,) * k_sub for gene in genes) for genes in gene_sets)


def count_alphabet(gene_sets):
    """Size of the grouped count alphabet these cells produce: prod_j (n_j + 1).

    Cells with an IDENTICAL gene set are one group, and the grouped estimator keeps
    only how many of them fired. That loses nothing: the count vector is a sufficient
    statistic of the labelled response for the input, and the label term cancels
    between H(Y) = H(K) + d and H(Y|X) = H(K|X) + d, so grouped MI equals the
    labelled-codeword MI exactly.

    So log2(count_alphabet) is a real ceiling on what the ARRAY can carry, however
    good the receptors are. Thirty cells with one shared repertoire are thirty noisy
    copies of one measurement. The ceiling is not monotonic in genes per cell because
    the number of distinct repertoires available, C(G, g), is not — print it next to
    any grouped-MI curve to tell "saturating its combinatorics" from "limited by the
    receptors".
    """
    from collections import Counter
    import math
    return math.prod(n + 1 for n in Counter(gene_sets).values())


def heterogeneous_conc_means(n_points, n_ligands, log10_span, seed):
    """One per-ligand log10 concentration-mean vector per sweep point, in +/- span.

    log10_span = 0 gives the flat channel (every ligand the same typical abundance).
    The environment initialises its EC50 from the MEAN of the vector, so a symmetric
    spread moves which ligands sit near saturation rather than moving the whole
    dynamic range.
    """
    generator = torch.Generator().manual_seed(seed)
    return [tuple(((torch.rand(n_ligands, generator=generator) * 2.0 - 1.0)
                   * log10_span).tolist())
            for _ in range(n_points)]


def bernoulli_p_for_mean(n_genes, mean, p_floor=0.02):
    """Per-gene expression probability giving the requested mean genes per cell.

    Each gene is expressed independently at probability p, so the NUMBER per cell is
    induced (Binomial), never chosen. Cells expressing nothing are redrawn, so the
    realised mean is the ZERO-TRUNCATED one, G*p / (1 - (1-p)^G), not G*p. Setting
    p = mean/G would overshoot badly at small means.

    p_floor exists because mean -> 1 needs p -> 0, and at p -> 0 almost every draw is
    an empty cell that gets rejected, so the sampler stalls. The floor bounds the
    rejection rate (acceptance ~ G*p_floor) at the price of a realised mean a little
    above 1. Plot the REALISED mean, which bernoulli_gene_sets reports.
    """
    if not 1 <= mean <= n_genes:
        raise ValueError(f"mean must lie in [1, {n_genes}], got {mean}.")
    realised = lambda p: n_genes * p / (1 - (1 - p) ** n_genes)
    if mean <= realised(p_floor):
        return p_floor
    lo, hi = p_floor, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if realised(mid) < mean else (lo, mid)
    return 0.5 * (lo + hi)


def bernoulli_gene_sets(n_genes, n_cells, mean, seed, p_floor=0.02):
    """Independent per-gene expression at a uniform probability, size induced.

    The counterpart to complete_coverage_gene_sets: no anchor, no size law, and no
    coordination between cells. A gene can end up expressed nowhere, and the spread
    of cell sizes is whatever the Binomial gives rather than something imposed.

    Returns (gene_sets, p, realised_mean).
    """
    from src.cells import sample_gene_sets_bernoulli   # keeps this module import-light
    p = bernoulli_p_for_mean(n_genes, mean, p_floor)
    sets = tuple(sample_gene_sets_bernoulli(n_cells, n_genes, [p] * n_genes, seed=seed))
    return sets, p, sum(map(len, sets)) / len(sets)


def receptor_pool(n_genes, k_sub, n_distinct, size, seed, use_interface_model=True):
    """`size` distinct receptors, each built from exactly `n_distinct` genes.

    The knob this exists for: a receptor's binding pocket is formed at the interface
    between adjacent subunits, so a HOMOMER (n_distinct=1) presents k_sub copies of
    one pocket while a fully mixed word presents k_sub different ones. Varying
    n_distinct at FIXED pool size and fixed receptors-per-cell isolates that
    per-receptor expressiveness from every other effect.

    n_distinct=1 can only ever yield n_genes receptors, so size <= n_genes there.
    """
    if not 1 <= n_distinct <= k_sub:
        raise ValueError(f"n_distinct must lie in [1, {k_sub}], got {n_distinct}.")
    if n_distinct == 1:
        if size > n_genes:
            raise ValueError(f"Only {n_genes} homomers exist, asked for {size}.")
        rng = random.Random(seed)
        return tuple(sorted(tuple((g,) * k_sub) for g in rng.sample(range(n_genes), size)))

    rng = random.Random(seed)
    found = set()
    for _ in range(200000):
        if len(found) == size:
            break
        genes = rng.sample(range(n_genes), n_distinct)
        word = tuple(genes) + tuple(rng.choice(genes) for _ in range(k_sub - n_distinct))
        word = tuple(word[i] for i in rng.sample(range(k_sub), k_sub))   # shuffle positions
        if len(set(word)) != n_distinct:
            continue
        found.add(_canonical_rotation(word) if use_interface_model else tuple(sorted(word)))
    if len(found) < size:
        raise ValueError(f"Only found {len(found)} receptors with {n_distinct} distinct "
                         f"genes out of {n_genes}; asked for {size}.")
    return tuple(sorted(found))


def shared_subsets(n_cells, pool_size, per_cell, seed):
    """One sorted tuple of `per_cell` POOL INDICES per cell, drawn without replacement.

    Indices, not receptors, so the same pattern can be applied to several different
    pools. Two arms built from the same subsets have byte-identical cell-type
    structure — same number of distinct repertoires, same multiplicities — and differ
    only in what the pool contains. That is what makes the comparison single-variable.
    """
    if per_cell > pool_size:
        raise ValueError(f"per_cell={per_cell} exceeds pool_size={pool_size}.")
    rng = random.Random(seed)
    return tuple(tuple(sorted(rng.sample(range(pool_size), per_cell)))
                 for _ in range(n_cells))
