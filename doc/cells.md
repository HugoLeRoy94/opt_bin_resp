# Cell mode

A cell $c$ is defined by the set of genes $G_c$ it expresses, $g = |G_c|$. It assembles
every receptor those genes allow, unless its repertoire is given explicitly. The
estimator sees $C$ cells, the physics runs over the deduplicated receptor pool
$R_{\rm pool}$. Code: `model/cells.py`. Theory: `../doc/theory/02_biophysics_mwc.md` §2.8,
`04_discrete_information.md` §4, `08_environmental_entropy_limits.md` §7.

## Entering cell mode

Set one of three `Config` fields. Precedence `cell_receptors` > `cell_gene_sets` > `n_cells`
(`model/cells.py::build_cell_array`).

| field | meaning |
|---|---|
| `cell_receptors` | explicit receptor list per cell |
| `cell_gene_sets` | explicit gene set per cell |
| `n_cells` | sample $C$ gene sets with `cell_sampling_strategy` |

`train/config.py::Config.__post_init__` resolves the cells once and stores the result:
`cell_gene_sets` (derived when sampled or explicit), `n_cells`, and `receptor_indices`
= the pool. `n_receptors` and `receptor_sampling_*` are then ignored.
`train/runner.py::Runner.build` rebuilds the same `CellArray` from the stored fields.
The pool order is deterministic, so it matches `receptor_indices`.

## Gene set to repertoire

`model/cells.py::expand_gene_set`: the canonical ring arrangements of all $g^k$ words
over $G_c$, $k$ = `k_sub`. Rotations are identified, reflections stay distinct (the
$\pm$ face asymmetry breaks mirror symmetry), same convention as `model/arrays.py`.
Canonical form = lexicographically smallest rotation (`model/arrays.py::_canonical_rotation`).
By Burnside's lemma, with $\varphi$ Euler's totient:

$$|\text{repertoire}(g)| = \frac1k\sum_{d\mid k}\varphi(d)\,g^{k/d}.$$

`model/arrays.py::pool_size(g, k)` evaluates it.

| $g$ | 1 | 2 | 3 | 4 | 5 | 8 |
|---|---|---|---|---|---|---|
| $k=5$ | 1 | 8 | 51 | 208 | 629 | 6560 |

**Pool.** `CellArray.__init__`: $R_{\rm pool}$ = sorted union of all repertoires. A
receptor shared by several cells is simulated once. $R_{\rm pool} \le$ `pool_size(n_genes, k_sub)`.
Costs: `batching.md`.

## Sampling gene sets

Seeded by `cell_sampling_seed`. Cells are drawn independently, so a gene can be
expressed by no cell.

- `bernoulli`, `sample_gene_sets_bernoulli`: gene $u$ expressed with probability
  $\pi_u$, independently. Through `Config`, $\pi_u = 2/G$ for every gene. Empty cells
  and cells above `cell_max_genes` are rejected and redrawn.
- `size_pmf`, `sample_gene_sets_by_size`: draw $g$ with $P(g = i+1) \propto$ `cell_size_pmf[i]`,
  then $g$ distinct genes. Through `Config` they are uniform. The size law and the
  gene-identity law are independent.

Helpers for building laws: `size_pmf_for_mean(n_genes, mean, family)`. `uniform` is
flat over $\{1..M\}$, reachable means $(M+1)/2$. `exponential` is
$p_i \propto e^{\lambda i}$ with $\lambda$ found by bisection, any mean in $(1, G)$.
`gene_expression_probs` builds per-gene weights, but `Config` has no field for them.

**Explicit repertoires** (`cell_receptors`). Each receptor is canonicalised to its
smallest rotation and duplicates within a cell are dropped. Weights are uniform over
the listed receptors and `cell_stoichiometry` is ignored. `cell_gene_sets` is derived
for reporting only. This path exists because a gene set cannot express a cell holding
a single heteromer: the genes of `[0,0,0,1,1]` also make every other ring over genes
0 and 1.

## Abundances

`repertoire_weights`, `cell_stoichiometry`. $w_{cr}$ is the fraction of cell $c$'s
receptors of type $r$, rows of $W$ $(C, R_{\rm pool})$ sum to 1, $w_{cr} = 0$ when $c$
cannot make $r$.

- `multinomial` (default): subunits made in equal amounts assemble at random, so the
  $g^k$ ordered words are equally likely and
  $w_{cr} = \operatorname{rot}(r)/g^k$, $\operatorname{rot}(r)$ = number of distinct rotations of $r$
  (`_ring_multiplicity`). Homomers have $\operatorname{rot} = 1$. At $g = 5$, $k = 5$ they are
  $5/3125 = 0.16\%$ of the cell.
- `uniform`: $w_{cr} = 1/|\text{repertoire}|$. At $g = 5$, $k = 5$ homomers are $5/629 = 0.8\%$.

## Readout

`CellReadout.forward`, the only readout:

$$A_{bc} = \operatorname{clamp}_{[0,1]}\sum_r w_{cr}\, p_{br},$$

$p_{br}$ the receptor open probability (`pipeline.md` §2), $A_{bc}$ read as the firing
probability of a stochastic binary cell. $W$ is a buffer, never learned, saved as
`readout_state` in `final_model.pt`. With one receptor per cell ($W$ the identity) it is
the receptor model. Justification and alternatives: `02_biophysics_mwc.md` §2.8.

## Chunked evaluation

`model/cells.py::cell_activity`. The sum over $r$ is linear, so the pool is walked in
slices of `cell_pool_chunk` columns of the composition matrix and the $(B, C)$
accumulator is summed. Peak forward tensor $(B, S, \text{chunk})$ instead of
$(B, S, R_{\rm pool})$. The $(B, S, P)$ pocket energies are shared by all chunks. Under
autograd the retained graph is still $O(B\,S\,R_{\rm pool})$ unless `recompute_backward`
is set, which gradient-checkpoints each chunk. `None` runs one pass. Same values either way.

## Grouping identical cells

`estimate/grouping.py::CellGrouping`, built in every cell-mode run by `Runner.build`.
Groups are the distinct rows of $W$ (`torch.unique(W, dim=0)`), never inferred from
similar sampled activities. Group $j$ has $n_j$ cells, $J$ groups, $\sum_j n_j = C$.
Identical rows give identical $A_{bc}$ for every input, with independent output draws,
so the count vector $\mathbf K$, $K_j$ = active cells in group $j$, carries all the
information: $I(X;Y) = I(X;\mathbf K)$. Proof: `04_discrete_information.md` §4 (exact
grouping), `08_environmental_entropy_limits.md` §7.3.

What it changes:

| | ungrouped | grouped |
|---|---|---|
| exact alphabet | $2^C$ | $Q = \prod_j(n_j+1)$ |
| KT pairwise cost | $O(B^2 C)$ | $O(B^2 J)$, same value |
| MI | $I(X;Y)$ | identical |
| entropy | $H(Y)$ | $H(\mathbf K)$, plus $D = \mathbb E\sum_j\log_2\binom{n_j}{K_j}$ to recover $H(Y)$ |

Group probability $p_{bj}$ is the mean of its cells' $A_{bc}$ (`CellGrouping.probabilities`).
The mean leaves values unchanged and spreads gradients evenly over copies. With all
cells distinct, grouping is a no-op. It is used by `grouped=True` objectives
(`pipeline.md` §4) and by the `grouped_information` and `grouped_counting` measurements,
whatever the objective. `Config` rejects all of them outside cell mode.
`cell_grouped_max_states` (default 65536) bounds $Q$ for exact enumeration only.
Independent sampling yields few repeated gene sets, so $Q$ is often far above it. Then
use `mi="kt"`, `grouped=True` and `grouped_counting`.

**Grouped measurement keys.**

| key | quantity |
|---|---|
| `grouped_count_entropy` | $H(\mathbf K)$, exact |
| `grouped_count_conditional_entropy` | $H(\mathbf K\mid X)$, exact |
| `grouped_label_entropy` | $D$ |
| `response_entropy_grouped`, `conditional_entropy_response_grouped` | $H(\mathbf K)+D$, $H(\mathbf K\mid X)+D$ |
| `mutual_information_grouped` | $H(\mathbf K) - H(\mathbf K\mid X)$ |
| `grouped_n_groups`, `grouped_n_states`, `grouped_count_entropy_upper` | $J$, $Q$, $\log_2 Q$ |
| `grouped_count_entropy_counting_{plugin,mm}` | sampled $H(\mathbf K)$, plug-in or MM (Miller-Madow) |
| `grouped_count_conditional_entropy_counting` | analytic $H(\mathbf K\mid X)$, batch mean |
| `grouped_label_entropy_counting` | $D = H(Y\mid X) - H(\mathbf K\mid X)$ |
| `response_entropy_grouped_counting_{plugin,mm}`, `conditional_entropy_response_grouped_counting` | sampled $H(\mathbf K)+D$, analytic $H(Y\mid X)$ |
| `mutual_information_grouped_counting_{plugin,mm}` | sampled $H(\mathbf K)$ minus $H(\mathbf K\mid X)$ |
| `grouped_counting_{K_hat,unique_fraction,missing_mass,samples,log2B}` | distinct $\mathbf K$ seen $U$, $U/n$, GT (Good-Turing) $f_1/n$, $n$, $\log_2 n$ |

Exact keys come from `GroupedCellMutualInformationLoss.metrics_from_statistics`, counting
keys from `estimate/grouping.py::GroupedResponseCounter.metrics`. Binomial supports have
$C + J$ entries (`support_counts`), so counting never allocates the $Q$ states.
