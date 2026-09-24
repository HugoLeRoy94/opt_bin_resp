# expression_law — independent gene expression, and where the MI peak sits

## Why this task exists

`tasks/cells/gene_expression` found mutual information peaking at exactly **2
genes per cell**, for heteromers, and raising the cell count did not move it.
Two things about that design make the result hard to interpret:

1. Every cell expressed **exactly** `g` genes. `g` was an integer, so the mean
   number of genes per cell could not be moved continuously through the peak.
2. With `--coverage complete`, every gene was forced to appear at least once in
   the array. That requires cells to know what other cells expressed. A
   developing tissue has no such mechanism.

This task replaces both with an independent law and sweeps `G` to test what
actually sets the peak.

## The generating law

Each cell draws its own gene set, knowing nothing about any other cell:

1. **How many** genes it expresses, from a distribution over `{1..G}`
   (`src.cells.size_pmf_for_mean`).
2. **Which** genes, by weighted sampling without replacement
   (`src.cells.gene_expression_probs`, then `src.cells.sample_gene_sets_by_size`).

The two laws are independent, and each is `uniform` or `exponential`:

| law | `uniform` | `exponential` |
|---|---|---|
| size | flat over `{1..M}`; mean `(M+1)/2`, so only 1, 1.5, ... are reachable | `p_i` proportional to `exp(lambda*i)`, `lambda` solved for any requested mean |
| gene | every gene equally likely | geometric decay; `--gene_ratio` = P(first)/P(last) |

At mean `= (G+1)/2` the exponential size law has `lambda = 0` and **is** the
uniform law. The two curves must coincide there, which is a free consistency
check on the sampler.

A gene can end up expressed nowhere. `genes_seen` records how many actually
appeared, and is reported per point rather than assumed.

## The hypothesis being tested

A cell expressing `g` of `G` genes can have one of `C(G, g)` gene sets, so cell
identity carries at most `log2 C(G, g)` bits, maximal at `g = G/2`. In the
existing G=5 data the count of distinct cell types is exactly `C(5, g)` and is
**identical for heteromers and homomers** (5, 10, 10, 5, 1), so it is pure
combinatorics, independent of what the cells assemble.

If that ceiling is what puts the peak at 2, then:

- raising `C` should not move the peak — **already observed**
- raising `G` should move it right, toward `G/2`, and short of it because
  dilution grows with `g` too

`--n_genes 3 5 8` is the test. The second figure in the analysis plots peak
position against `G` with the `g = G/2` line drawn on it.

## What independent sampling costs you

Nearly every cell becomes its own type: at G=5, C=30, mean 2.5 the sweep draws
about **18 distinct gene sets out of 30 cells**, against 10 in the old fixed-size
design. Two consequences, both unavoidable:

- The joint count alphabet `prod_j (n_j + 1)` reaches roughly `10^7`, so **exact
  enumeration is impossible**. Training uses `entropy='grouped_kt_mi'` and
  evaluation uses `grouped_counting`. Both are estimates, not exact values.
- Sampled counting is biased **downward** when the symbol space is undersampled.
  The analysis flags any point with `grouped_counting_unique_fraction > 0.5` or
  MI within one bit of `log2(evaluation inputs)`. Treat those as measurements of
  the budget, not of the array, and re-check them with a larger
  `--final_batch_size`.

The receptor pool is the memory bottleneck and grows fast: G=8 at mean 3 reaches
about 2200 receptors against 629 for the whole of the old G=5 sweep. `--max_pool`
refuses points above a cap rather than failing hours into a run.

## Running it

```bash
# inspect the design, the realised means, the pools and the cell-type counts
python3 tasks/cells/expression_law/scripts/expression_law.py --dry_run

# the two baseline arms, uniform size law and uniform gene law
python3 tasks/cells/expression_law/scripts/expression_law.py --condition heteromers
python3 tasks/cells/expression_law/scripts/expression_law.py --condition homomers

# the shape comparisons, at matched mean genes per cell
python3 tasks/cells/expression_law/scripts/expression_law.py --size_family exponential
python3 tasks/cells/expression_law/scripts/expression_law.py --gene_family exponential --gene_ratio 10
```

Then `bash tasks/cells/expression_law/sync.sh` and edit `SWEEPS` at the top of
`analysis/expression_law.py` to the folders you want on the figure.

## Comparability with gene_expression

The environment block is copied verbatim from
`tasks/cells/gene_expression/scripts/replicates.py` (5 families, 100 ligands,
latent dimension 6, family spread 0.1, `mu_ligands_per_source` 1e-6), as are
`cell_readout='mean'`, `k_sub=5`, the interface model and the multinomial
stoichiometry. What differs is the expression law and the estimator. The
estimator difference alone shifts MI, so compare the two tasks only through the
`grouped_kt_mi` / counting arm of `gene_expression`, never through its exact arm.
