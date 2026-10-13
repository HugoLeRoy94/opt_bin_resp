"""Generators for cell gene sets and receptor lists. No config, no CLI, no runner.

This is the ONLY shared code under tasks/cells/. Everything else — every parameter
of every experiment — is written out in the script that runs it. If you are looking
for what a run did, open its script; nothing is hidden here except the combinatorics
of picking which genes land in which cell.

Everything below is pure: same arguments, same output, no global state.
"""
import torch

__all__ = ["complete_coverage_gene_sets", "random_gene_sets", "homomers_only",
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
