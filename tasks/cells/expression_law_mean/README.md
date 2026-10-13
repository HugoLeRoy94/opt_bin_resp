# expression_law — independent gene expression, and where the MI peak sits

## Why this task exists

`tasks/cells/gene_expression_mean` found mutual information peaking at exactly **2
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
  The analysis plots the **Good-Turing missing mass** `grouped_counting_missing_mass`
  ($f_1/B$, the share of observations that are the only sighting of their symbol)
  beside the MI, and MI against it. There is no pass/fail threshold: the quantity
  is continuous, and a point sitting at a high missing mass is partly a
  measurement of `--final_batch_size` rather than of the array.

### How big does the evaluation batch need to be

`tasks/cells/gene_expression_mean/scripts/estimator_crosscheck.py` measured this bias
against exact enumeration, on 5 expression levels x 3 budgets of the G=5, C=30
arrays. Over those 12 points:

```
bias [bits]  ~=  3.34 * (B / 2^H(K)) ** -0.92
```

What matters is the **ratio of budget to effective alphabet** `2^H(K)`, not the
budget alone. Bias falls below 0.1 bits at a ratio of about 46, and below 0.05 at
about 98.

Two consequences, both printed by `--dry_run`:

**Repeats are not budget.** The final measurement is repeated `--final_repeats`
times. Repeats show evaluation noise, and cannot touch a systematic bias that
depends only on the batch size. Ten repeats of 16k carry exactly the bias of one.
At equal total cost, few repeats of a large batch win: on the G=5, C=30 g=3 arrays,
spending 163,840 inputs as 10 x 16,384 gives 1.82 bits of bias, and as 1 x 163,840
gives 0.51. The defaults here are therefore 2 repeats of 524,288.

**The cell count is the strongest lever, not the budget.** `H(K) = MI + H(K|X)`,
and `H(K|X)` grows with C because every cell contributes its own output noise,
close to half a bit each under the `mean` readout. So `2^H(K)` grows
*exponentially* in C, while the budget you can afford grows linearly. At C=30 the
projection is `H(K)` near 17-20 bits, needing millions to tens of millions of
inputs for 0.1 bits of bias. **Halving `--n_cells` cuts `H(K)` by roughly 7 bits,
worth more than a 100x budget increase.** Since the MI peak is set by G and not by
C (see above), C=15-20 costs little scientifically and makes the measurement
honest.

Run one replicate at one G first, read `grouped_count_entropy_counting_plugin` and
`grouped_counting_missing_mass` out of it, then set the budget for the full sweep.

The receptor pool is the memory bottleneck and grows fast: G=8 at mean 3 reaches
about 2200 receptors against 629 for the whole of the old G=5 sweep. `--max_pool`
refuses points above a cap rather than failing hours into a run.

## The size-law comparison: two scripts, one grid

`scripts/size_law_uniform.py` and `scripts/size_law_exponential.py` are thin
wrappers over `expression_law.py`. Both keep the GENE law uniform and change only
the SIZE law, so the pair isolates the shape of "how many genes a cell expresses".
Anything passed on the command line overrides their defaults.

```bash
python3 tasks/cells/expression_law_mean/scripts/size_law_uniform.py --dry_run
bash tasks/run_remote.sh cells/expression_law_mean size_law_uniform.py 0
bash tasks/run_remote.sh cells/expression_law_mean size_law_exponential.py 1
# the homomer arm of each
bash tasks/run_remote.sh cells/expression_law_mean size_law_uniform.py 0 -- --condition homomers
```

Defaults: `--n_genes 3 5`, `--means 1.0 1.5 2.0 2.5 3.0`, `--n_cells 30`,
`--replicates 3`, `--max_pool 8192`, `--match_size_support`. That is 8 design
points and 24 optimizations per arm, about 1.0 h (uniform) and 1.5 h (exponential)
per condition at the calibrated cluster rate; `--dry_run` reprojects it.

Three things make this a controlled comparison, and one thing does not:

**Matched grid.** `--match_size_support` drops the `(G, mean)` points the flat law
cannot reach. A flat law over `{1..M}` only reaches means `(M+1)/2`, so without
the flag the exponential arm covers means the uniform arm does not and the two
curves have different support. The earlier runs in `data/expression_law` have
exactly that problem: the uniform arm ran `--n_genes 3 5 8`, the exponential arm
`--n_genes 3 5`.

**Matched environment, unmatched world.** Both arms use the same environmental
settings (below). They do NOT use the same seed: `world_seed_for()` hashes the
size family, the gene family and the condition, so latents and per-ligand
concentration means differ between arms. A gap between the curves has to survive a
change of environment to mean anything.

**One thing is NOT matched, and cannot be.** At the same mean the exponential law
spreads mass over the whole range `{1..G}`, so a few cells express every gene and
the receptor pool saturates immediately: 629 receptors at G=5 for every mean above
1, against 53 -> 629 as the flat law's mean rises. Matched MEAN is not matched pool
and not matched cost. That is the laws differing, not a confound, but it is why the
exponential arm is the more expensive one and why `--max_pool` has to be raised
past the 4096 default.

Adding `--n_genes 8` back costs about 6 h (uniform) and 15 h (exponential) per
condition, because the exponential law hits the full G=8 pool of 6560 receptors.

## Running the general script

```bash
# inspect the design, the realised means, the pools and the cell-type counts
python3 tasks/cells/expression_law_mean/scripts/expression_law.py --dry_run

# the two baseline arms, uniform size law and uniform gene law
python3 tasks/cells/expression_law_mean/scripts/expression_law.py --condition heteromers
python3 tasks/cells/expression_law_mean/scripts/expression_law.py --condition homomers

# the gene-law comparison, at matched mean genes per cell
python3 tasks/cells/expression_law_mean/scripts/expression_law.py --gene_family exponential --gene_ratio 10
```

Then `bash tasks/cells/expression_law/sync.sh` and edit `SWEEPS` at the top of
`analysis/expression_law.py` to the folders you want on the figure.

## Comparability with gene_expression

The environment matches `tasks/cells/gene_expression_mean/scripts/gene_expression.py`:
1 family, 100 ligands, latent dimension 6, family spread 0.1, gaussian geometry,
one presence block, `cell_readout='mean'`, `k_sub=5`, the interface model and the
multinomial stoichiometry. Two parts of it are now knobs rather than constants,
and both must stay in step with that script:

- `--mu_ligands` (default 3.0) is the zero-truncated Poisson rate for the number
  of ligands present per sniff. With one presence block it is the only presence
  knob. The old value was `1e-6`, effectively exactly one ligand per sniff; 3.0 is
  a MIXTURE, which is a different problem, not a tweak. Runs in
  `data/expression_law` from before this change are single-ligand and are not
  comparable with new ones.
- `--conc_log10_span` (default 2.0) makes the concentration channel heterogeneous:
  each ligand's log10 concentration mean is drawn uniformly in `[-span, +span]`,
  so abundances differ by up to 4 decades. One independent vector per run, seeded
  from the world seed. The environment initialises its EC50 from the MEAN of that
  vector, so the spread moves which ligands sit near saturation rather than moving
  the whole dynamic range. `--conc_log10_span 0` restores the old flat channel.

What still differs from `gene_expression` is the expression law and the
estimator. `gene_expression` now trains on `kt_mi`; this task trains on
`grouped_kt_mi` because independent sampling leaves nearly every cell its own type
and the exact count alphabet is out of reach. Both are KT bounds, but the grouping
is not a no-op, so compare the two tasks through their KT arms and never against
an exact-enumeration number.
