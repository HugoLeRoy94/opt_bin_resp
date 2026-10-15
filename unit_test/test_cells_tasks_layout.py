"""The flattened tasks/cells scripts: single-variable comparisons, matched grids.

These guard the PROPERTIES the folder layout is supposed to deliver, not the
numbers. A script is self-contained, so the only way to check that two of them
differ in one thing is to import both and compare their constants.
"""
import importlib.util
import math
from collections import Counter
from pathlib import Path

import pytest

TASKS = Path(__file__).resolve().parents[1] / "tasks" / "cells"


def load(relative):
    """Import a task script by path, without running main()."""
    path = TASKS / relative
    spec = importlib.util.spec_from_file_location(path.stem + relative.replace("/", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# MU_LIGANDS, CONC_SPAN and CONC_STD are deliberately ABSENT. They are the knobs you
# turn one at a time to probe the environment, so pinning them here would fail on
# every legitimate ablation and quickly be ignored. What matters is the environment
# the two sweeps were actually RUN in, which only the data can tell you: the analysis
# reads it out of each run's config and shouts when selected sweeps disagree.
SHARED = ("N_LIGANDS", "LATENT_DIM", "FAMILY_SPREAD", "FAMILY_DIST", "GEOMETRY",
          "DISTRIBUTION", "NOISE_SIGMA",
          "N_GENES", "N_CELLS", "K_SUB", "GENES_PER_CELL", "CELL_SEED",
          "ENTROPY", "MAX_STATES", "MEASUREMENTS", "EPOCHS", "LR", "BATCH_SIZE",
          "TEST_BATCH_SIZE", "FINAL_BATCH_SIZE", "EVAL_CHUNK_SIZE",
          "TEMPERATURE", "INIT_TEMPERATURE")


@pytest.mark.parametrize("arm", ["heteromers.py", "homomers.py"])
def test_mean_and_threshold_differ_only_in_the_readout(arm):
    """The readout comparison is worthless if anything else drifts between them."""
    mean = load(f"gene_expression_mean/scripts/{arm}")
    threshold = load(f"gene_expression_threshold/scripts/{arm}")
    for name in SHARED:
        assert getattr(mean, name) == getattr(threshold, name), name
    assert mean.READOUT == "mean"
    assert threshold.READOUT == "threshold"
    # and the worlds are NOT shared, so a gap has to survive a change of environment
    assert mean.WORLD_SEED != threshold.WORLD_SEED
    assert mean.CONC_SEED != threshold.CONC_SEED


@pytest.mark.parametrize("task", ["gene_expression_mean", "gene_expression_threshold"])
def test_heteromer_and_homomer_arms_hold_the_same_cells(task):
    """Only what each cell may ASSEMBLE differs; the gene sets must be identical."""
    hetero = load(f"{task}/scripts/heteromers.py")
    homo = load(f"{task}/scripts/homomers.py")
    assert hetero.GENE_SETS == homo.GENE_SETS
    for sets, receptors in zip(homo.GENE_SETS, homo.CELL_RECEPTORS):
        assert len(sets) == len(receptors)
        for genes, cell in zip(sets, receptors):
            assert cell == tuple((g,) * homo.K_SUB for g in genes)
    assert hetero.WORLD_SEED != homo.WORLD_SEED


def test_count_alphabet_cap_is_not_monotonic_in_genes_per_cell():
    """The array's own information ceiling, which is a hump before any training.

    The cap is log2(prod_j (n_j + 1)) over groups of identical cells. Grouping is
    lossless for MI, so this bounds the real channel: it tracks how many DISTINCT
    repertoires exist, C(G, g), which peaks at g = G/2 and collapses to one at g = G.
    """
    mean = load("gene_expression_mean/scripts/heteromers.py")
    caps = [math.log2(math.prod(n + 1 for n in Counter(s).values()))
            for s in mean.GENE_SETS]
    assert len(caps) == 5
    assert caps[1] > caps[0]        # rises to g=2
    assert caps[4] < caps[0]        # and collapses at g=5, every cell identical
    assert caps.index(max(caps)) == 1
    assert max(caps) <= math.log2(mean.MAX_STATES)


def test_both_size_laws_run_the_same_matched_grid():
    """Different support would make the two expression-law curves incomparable."""
    uniform = load("expression_law_mean/scripts/size_law_uniform.py")
    exponential = load("expression_law_mean/scripts/size_law_exponential.py")
    assert uniform.SIZE_FAMILY == "uniform"
    assert exponential.SIZE_FAMILY == "exponential"

    grid = lambda m: sorted((r["n_genes"], r["target_mean"]) for r in m.design())
    assert grid(uniform) == grid(exponential)
    # the flat law cannot reach a mean needing M > G, so those points are absent
    assert (3, 2.5) not in grid(uniform)
    assert (5, 2.5) in grid(uniform)
    # matched mean is NOT matched pool: that is the laws differing, and it must show
    pools = {m: {(r["n_genes"], r["target_mean"]): r["pool"] for r in m.design()}
             for m in (uniform, exponential)}
    assert pools[exponential][(5, 1.5)] > pools[uniform][(5, 1.5)]
    assert uniform.WORLD_SEED != exponential.WORLD_SEED


def test_the_environment_knobs_exist_and_are_wired_to_the_config():
    """Every sweep script must expose the same two environment knobs, by name.

    Their VALUES are free — turning one at a time is the point. What is not free is
    where they live: if a script stops reading MU_LIGANDS / CONC_SPAN, flipping a
    regime silently does nothing.
    """
    for relative in ("gene_expression_mean/scripts/heteromers.py",
                     "gene_expression_mean/scripts/homomers.py",
                     "gene_expression_threshold/scripts/heteromers.py",
                     "gene_expression_threshold/scripts/homomers.py",
                     "expression_law_mean/scripts/size_law_uniform.py",
                     "expression_law_mean/scripts/size_law_exponential.py"):
        module = load(relative)
        assert isinstance(module.MU_LIGANDS, (int, float)), relative
        assert isinstance(module.CONC_SPAN, (int, float)), relative
        source = (TASKS / relative).read_text()
        assert "mu_ligands_per_source  = MU_LIGANDS" in source, relative
        assert "N_LIGANDS, CONC_SPAN, CONC_SEED" in source, relative


# ---------------------------------------------------------------------------
# Mixed-size complete coverage: a continuous axis that still covers every gene
# ---------------------------------------------------------------------------

def test_mixed_coverage_hits_the_mean_and_still_covers():
    """Continuous mean is the whole point; losing coverage would silently change
    the experiment into the independent one."""
    import sys
    sys.path.insert(0, str(TASKS))
    from _lib.repertoires import (complete_coverage_gene_sets,
                                  complete_coverage_gene_sets_mixed)
    for n_cells in (20, 50):
        for mean in (1.0, 1.25, 1.5, 2.75, 4.0, 5.0):
            sets, realised = complete_coverage_gene_sets_mixed(5, n_cells, mean, 0)
            assert len(sets) == n_cells
            assert abs(realised - mean) <= 1.0 / n_cells, (n_cells, mean, realised)
            assert {g for gs in sets for g in gs} == set(range(5)), (n_cells, mean)
            assert {len(gs) for gs in sets} <= {int(mean), int(mean) + 1}
    # at an integer mean it must BE the integer generator, or the two arms of the
    # sweep would silently use different samplers at the grid points they share
    for g in (1, 3, 5):
        assert complete_coverage_gene_sets_mixed(5, 20, float(g), 0)[0] == \
               complete_coverage_gene_sets(5, 20, g, 0)


def test_mixed_coverage_rejects_impossible_means():
    import sys
    sys.path.insert(0, str(TASKS))
    from _lib.repertoires import complete_coverage_gene_sets_mixed
    with pytest.raises(ValueError, match="mean must lie"):
        complete_coverage_gene_sets_mixed(5, 20, 5.5, 0)
    with pytest.raises(ValueError, match="n_cells >= n_genes"):
        complete_coverage_gene_sets_mixed(8, 5, 2.0, 0)


# ---------------------------------------------------------------------------
# The pre-flight that stops a doomed sweep before it starts
# ---------------------------------------------------------------------------

def test_counting_budget_refuses_an_unresolvable_design():
    """The failure this guards against already happened twice: MI went NEGATIVE
    because H(K|X) exceeded log2(B)."""
    import sys
    sys.path.insert(0, str(TASKS))
    from _lib.budget import counting_budget_report, projected_counting_bias
    from _lib.repertoires import complete_coverage_gene_sets
    rows = [{"gene_sets": complete_coverage_gene_sets(5, 50, g, 0)} for g in (1, 2, 3)]
    # bias must fall as the budget rises, and the guard must fire on a small one
    big, _, _ = projected_counting_bias(rows[1]["gene_sets"], 2 ** 26)
    small, _, _ = projected_counting_bias(rows[1]["gene_sets"], 2 ** 16)
    assert small > big
    counting_budget_report(rows, 2 ** 26, limit=0.2)
    with pytest.raises(ValueError, match="exceeds"):
        counting_budget_report(rows, 2 ** 16, limit=0.2)


# ---------------------------------------------------------------------------
# fig3
# ---------------------------------------------------------------------------

FIG3_SHARED = ("N_LIGANDS", "LATENT_DIM", "FAMILY_SPREAD", "FAMILY_DIST", "GEOMETRY",
               "DISTRIBUTION", "NOISE_SIGMA", "N_GENES", "N_CELLS", "GENES_PER_CELL",
               "REPLICATES", "CELL_SEED", "ENTROPY", "MEASUREMENTS", "EPOCHS", "LR",
               "BATCH_SIZE", "TEST_BATCH_SIZE", "FINAL_BATCH_SIZE", "EVAL_CHUNK_SIZE",
               "TEMPERATURE", "INIT_TEMPERATURE", "MU_LIGANDS", "CONC_SPAN", "CONC_STD")


def test_fig3_arms_differ_only_in_k_sub():
    """Three curves on one axis are worthless if anything else drifts between them."""
    arms = {k: load(f"fig3/scripts/k{k}.py") for k in (2, 3, 5)}
    reference = arms[5]
    for k, module in arms.items():
        for name in FIG3_SHARED:
            assert getattr(module, name) == getattr(reference, name), (k, name)
        assert module.K_SUB == k
        assert module.ROWS[0]["gene_sets"] == reference.ROWS[0]["gene_sets"]
    # ... but they must NOT share a world
    seeds = {(m.WORLD_SEED, m.CONC_SEED) for m in arms.values()}
    assert len(seeds) == 3


def test_fig3_pool_saturates_soonest_at_low_k_sub():
    """The mechanism the task tests: at k=2 the pool is exhausted by g=2, so any
    decline past there is pure dilution."""
    import sys
    sys.path.insert(0, str(TASKS.parent.parent))
    from src.cells import expand_gene_set
    pools = {k: [len(expand_gene_set(range(g), k, True)) for g in range(1, 6)]
             for k in (2, 3, 5)}
    assert pools[2] == [1, 3, 6, 10, 15]
    assert pools[5] == [1, 8, 51, 208, 629]
    # union over cells covering every pair saturates at g=2 for k=2
    assert len(expand_gene_set(range(2), 2, True)) == 3
    for k in (2, 3, 5):
        assert pools[k] == sorted(pools[k])
