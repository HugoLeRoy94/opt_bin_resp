"""The independent expression law: size distribution, gene weighting, and the sweep."""
import math
from collections import Counter

import pytest

from src.cells import (expand_gene_set, gene_expression_probs, sample_gene_sets_by_size,
                       size_pmf_for_mean)


def _mean(pmf):
    return sum((i + 1) * p for i, p in enumerate(pmf))


@pytest.mark.parametrize("family", ["uniform", "exponential"])
@pytest.mark.parametrize("mean", [1.0, 1.5, 2.0, 2.5, 3.0])
def test_size_pmf_hits_the_requested_mean(family, mean):
    pmf = size_pmf_for_mean(5, mean, family)
    assert _mean(pmf) == pytest.approx(mean, abs=1e-9)
    assert sum(pmf) == pytest.approx(1.0)
    assert min(pmf) >= 0


def test_exponential_reaches_means_uniform_cannot():
    for mean in (1.2, 1.73, 2.31, 4.4):
        assert _mean(size_pmf_for_mean(5, mean, "exponential")) == pytest.approx(mean, abs=1e-9)
    with pytest.raises(ValueError, match="cannot reach mean"):
        size_pmf_for_mean(5, 1.2, "uniform")


def test_the_two_size_families_coincide_at_the_flat_mean():
    # lambda = 0 is the flat distribution, so at mean (G+1)/2 the families agree.
    flat = size_pmf_for_mean(5, 3.0, "uniform")
    solved = size_pmf_for_mean(5, 3.0, "exponential")
    assert solved == pytest.approx(flat, abs=1e-9)


def test_size_pmf_rejects_means_outside_the_pool():
    with pytest.raises(ValueError):
        size_pmf_for_mean(5, 0.5, "exponential")
    with pytest.raises(ValueError):
        size_pmf_for_mean(5, 6.0, "exponential")


def test_gene_probs_shape():
    assert gene_expression_probs(5, "uniform") == pytest.approx([0.2] * 5)
    skewed = gene_expression_probs(5, "exponential", ratio=10.0)
    assert sum(skewed) == pytest.approx(1.0)
    assert skewed == sorted(skewed, reverse=True)
    assert skewed[0] / skewed[-1] == pytest.approx(10.0)


def test_sampled_sizes_follow_the_requested_mean():
    pmf = size_pmf_for_mean(5, 2.0, "uniform")
    sets = sample_gene_sets_by_size(20000, 5, pmf, seed=0)
    assert sum(len(s) for s in sets) / len(sets) == pytest.approx(2.0, abs=0.03)
    assert all(1 <= len(s) <= 3 for s in sets)          # flat support is {1,2,3}


def test_gene_weighting_biases_which_genes_appear():
    pmf = size_pmf_for_mean(5, 2.0, "uniform")
    uniform = Counter(g for s in sample_gene_sets_by_size(
        4000, 5, pmf, gene_probs=gene_expression_probs(5, "uniform"), seed=1) for g in s)
    skewed = Counter(g for s in sample_gene_sets_by_size(
        4000, 5, pmf, gene_probs=gene_expression_probs(5, "exponential", 20.0), seed=1)
        for g in s)
    assert max(uniform.values()) / min(uniform.values()) < 1.15
    assert skewed[0] > 3 * skewed[4]


def test_cells_are_drawn_independently():
    """No coordination: a gene may be expressed nowhere, unlike a covering design."""
    pmf = size_pmf_for_mean(8, 1.0, "uniform")          # one gene each, 8 to choose from
    missing = sum(len({g for s in sample_gene_sets_by_size(5, 8, pmf, seed=s)
                       for g in s}) < 8 for s in range(20))
    assert missing == 20        # 5 cells cannot cover 8 genes, and none is forced to


def test_sampler_is_reproducible_and_seed_sensitive():
    pmf = size_pmf_for_mean(5, 2.0, "uniform")
    a = sample_gene_sets_by_size(30, 5, pmf, seed=7)
    assert a == sample_gene_sets_by_size(30, 5, pmf, seed=7)
    assert a != sample_gene_sets_by_size(30, 5, pmf, seed=8)


def test_sampler_rejects_inconsistent_arguments():
    pmf = size_pmf_for_mean(5, 2.0, "uniform")
    with pytest.raises(ValueError, match="gene_probs has length"):
        sample_gene_sets_by_size(5, 5, pmf, gene_probs=[0.5, 0.5])
    with pytest.raises(ValueError, match="only 3 exist"):
        sample_gene_sets_by_size(5, 3, [0.2] * 5)


def test_independent_sampling_defeats_exact_enumeration():
    """Justifies why this task trains with KT and measures by counting.

    The old fixed-size covering design left many identical cells, so the joint
    count alphabet stayed small. Independent sampling does not.
    """
    pmf = size_pmf_for_mean(5, 2.5, "uniform")
    sets = sample_gene_sets_by_size(30, 5, pmf, seed=0)
    states = math.prod(n + 1 for n in Counter(sets).values())
    assert len(set(sets)) > 12          # most of 30 cells are their own type
    assert states > 10 ** 6             # far beyond cell_grouped_max_states


def test_design_builds_and_stays_within_its_pool_cap():
    from tasks.cells.expression_law.scripts.expression_law import design, parse_args
    args = parse_args(["--n_genes", "3", "5", "--means", "1.0", "2.0",
                       "--replicates", "2", "--n_cells", "12"])
    rows = design(args)
    assert len(rows) == 2 * 2 * 2
    for row in rows:
        assert row["receptor_pool"] <= args.max_pool
        assert abs(row["realised_mean_genes"] - row["target_mean_genes"]) < 0.6
        pool = {r for gs in row["cell_gene_sets"] for r in expand_gene_set(gs, 5, True)}
        assert len(pool) == row["receptor_pool"]

    tight = parse_args(["--n_genes", "8", "--means", "3.0", "--max_pool", "10"])
    with pytest.raises(ValueError, match="exceeds --max_pool"):
        design(tight)
