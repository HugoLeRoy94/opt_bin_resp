# Batch sizes and memory

Runs take explicit integers. Nothing is resolved at run time, so `config.json` records
the number actually used. `train/batching.py::suggest` computes a suggestion to paste
into the sweep script:

```bash
python -m train.batching --n_receptors 20 --n_ligands 100 --gpu_gb 80
```

These are **memory models, not measurements**. The safety factors were fitted on an
A100. If a run dies with an out-of-memory error, lower the number by hand.

Symbols. $B$ batch size. $N$ binary outputs the estimator sees: $R$ receptors, or $C$
cells in cell mode. $L$ ligand pool. $S$ sparse ligand width of a batch, $S \le L$. $k$
subunits per receptor (`k_sub`). $P$ distinct pockets of the array. $R_{\rm pool}$
receptor pool in cell mode. $J$ cell groups, $n_j$ their sizes, $Q = \prod_j(n_j+1)$
joint count states. KT = Kolchinsky-Tracey pairwise bound. MI = mutual information.

---

## The three numbers

| name | used for | where it lives |
|---|---|---|
| `batch_size` | one training step | `Config` |
| `test_batch_size` | the per-epoch curve, only when `per_epoch_measure` | `Config` |
| `measure_batch_size` | one measurement pass | a test script's `BATCH_SIZE` / `--batch_size`, not a `Config` field |

Training carries a backward graph, so it is the most memory-hungry per sample. The
per-epoch measurement runs under `no_grad` but often, so `suggest` sets it to
$\min(4B, \text{measure})$ to stop the $O(B^2)$ KT cost dominating every epoch.
A measurement pass runs once and can be as large as memory allows.

Defaults in the test scripts: KT passes use $2^{13}$ with chunk 2048. Hard and sampled
codeword counting use $2^{18}$ (`analytic_baseline`, `large_R`) or $2^{21}$
(`two_families`) with chunk 4096. `grouped_counting` uses $2^{26}$ with chunk 512.

---

## One forward pass

**Sparse ligand axis.** Physics tensors carry $S$ ligand slots, not $L$. $S$ =
`LigandEnvironment.s_upper` starts at 1 and is raised to the largest mixture ever
drawn, never lowered. Estimates using $L$ are conservative. An estimate using a
typical mixture size must allow for the widening on a rare large draw.

**Composition binding.** A pocket's energy depends only on its (plus-face gene,
minus-face gene) pair, so `LigandEnvironment.bind_receptors` lists the $P \le \min(G^2, Rk)$
distinct pockets once. Energies are $(B, S, P)$, and one matmul with the $(P, R)$
composition gives $\ln{\rm EC50}$ $(B, S, R)$. No $(B, S, R, k)$ tensor exists. Example
from the code comment: a pool of 53k receptors has 267k pocket slots but about 644
distinct pockets. Distances use $\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2\langle a,b\rangle$, so no
$(B, S, P, D)$ broadcast exists either.

Live tensors per step: $E$ and its intermediates $(B, S, P)$, then $\ln{\rm EC50}$ and
$\ln c - \ln{\rm EC50}$ $(B, S, R)$, then $(B, R)$ after the log-sum-exp. In cell mode
$R$ is $R_{\rm pool}$, or the chunk width with `cell_pool_chunk` (`cells.md`).

**Pool size in cell mode.** A cell with $g$ genes makes
$\frac1k\sum_{d\mid k}\varphi(d)\,g^{k/d}$ ring receptors ($\varphi$ Euler's totient,
`model/arrays.py::pool_size`). $R_{\rm pool}$ is the deduplicated union, bounded by
`pool_size(n_genes, k_sub)`.

| genes | 1 | 2 | 3 | 4 | 5 | 8 | 10 | 26 |
|---|---|---|---|---|---|---|---|---|
| $k = 2$ | 1 | 3 | 6 | 10 | 15 | 36 | 55 | 351 |
| $k = 3$ | 1 | 4 | 11 | 24 | 45 | 176 | 340 | 5876 |
| $k = 5$ | 1 | 8 | 51 | 208 | 629 | 6560 | 20008 | 2376296 |

The estimator sees $C$, the physics pays for $R_{\rm pool}$. Pass `--n_receptors C` and
`--n_physics_receptors R_pool`, or the physics cost is badly underestimated. Levers on
the physics side: `cell_max_genes` or a capped `cell_size_pmf` (the table is
superlinear in $g$), then `cell_pool_chunk` with `recompute_backward`.

---

## What binds, by estimator

### KT (`mi="kt"`)

The estimator compares every pair of samples: $O(B^2 N)$ work in $(m, m, N)$ tiles.

**Without `recompute_backward`, memory binds.** Autograd retains every tile, so the
retained graph is $4B^2N$ bytes whatever the tile size:

$$B = \sqrt{\text{budget} / (4 N \cdot \text{SAFETY})}, \qquad \text{SAFETY} = 3,$$

`SAFETY` covering the transient `torch.log` output and its backward buffer.

**With `recompute_backward`, compute binds.** Each row block is recomputed in backward
instead of retained, so the graph drops to $O(BN)$ and time becomes the limit:

$$B = \sqrt{\texttt{KT\_COMPUTE\_BUDGET}/N}, \qquad \texttt{KT\_COMPUTE\_BUDGET} = 4 \times 5.3\times10^9 \text{ pair-evaluations}.$$

$5.3\times10^9$ is the memory-bound work on an 80 GiB A100, so the default is about
$2\times$ the batch and $4\times$ the step time. It is a time budget and does not grow
with the GPU.

The training tile $m = \min(B, \texttt{KT\_TRAIN\_TILE} = 4096)$ only sets a transient
$m^2 N$ under the retained $B^2 N$. Larger tiles mean fewer kernel launches. No effect
on the value. With `grouped=True` the width is $J$ instead of $C$ (`--grouped_n_groups`).

### Exact (`mi="shannon"`, `grouped=False`)

The joint table is $(B, 2^N)$ float32, so $B \le \text{budget}/(2^N \cdot 16)$. Usable below
about $N = 15$.

### Exact grouped (`mi="shannon"`, `grouped=True`)

Retains a $(B, Q)$ table: $B \le \text{budget}/(16 Q)$. `cell_grouped_max_states`
(default 65536) refuses the run when $Q$ is larger. Requires `--grouped_n_states`.

---

## Caps that apply to everything

**Statistical saturation.** Beyond these, more samples buy negligible variance:

- `shannon`: $\max(512, 100\cdot 2^N)$,
- KT: $\max(512, 100\cdot2^{\min(N,15)}, 16\cdot2^{N/2})$,
- grouped (either `mi`): $\max(512, 100\,Q)$.

A heuristic, not a convergence guarantee. Whether an estimate is sample-limited is
read from $\log_2 B$ and the `missing_mass` diagnostics.

**Physics.**

$$B \le \text{budget} / (L' \cdot R_{\rm pool} \cdot k \cdot 4 \cdot 16),$$

$L'$ = `--n_ligands`, $R_{\rm pool}$ = `--n_physics_receptors` or `--n_receptors`. It charges
$R_{\rm pool}k$ per ligand slot where the live tensors are $(B, S, P)$ and $(B, S, R)$, and
$P \le Rk$, so it is conservative. Passing $L$ for $L'$ is conservative again since
$S \le L$. The factor 16 leaves room for the retained energy graph next to the KT tiles.
Every cap is floored at $B = 512$ (`B_MIN`), except in the grouped exact branch,
floored at 1.

---

## Evaluation

`Runner._eval_stats` streams chunks of $\min(\texttt{eval\_chunk\_size} \text{ or } B_{\rm train}, B_{\rm eval})$
sniffs under `no_grad`, so $B_{\rm eval}$ is not limited by one forward pass. What it retains:

| measurement | retained across chunks |
|---|---|
| KT bounds | all activities, $(B_{\rm eval}, N)$ float32 on the GPU, $J$ columns when grouped KT is the loss |
| `codeword_entropy` | boolean codes $(B_{\rm eval}, N)$ on the CPU |
| `grouped_counting` | the `SymbolCounter` table |
| `grouped_information` | a $Q$-entry probability sum |

KT evaluation tiles with `EVAL_TILE = 2048`, the binding term being the
$(\text{tile}, B)$ row buffer:

$$B_{\rm eval} \le \text{budget}/(\texttt{EVAL\_TILE}\cdot 16).$$

`suggest` returns $\text{measure} = \max(B, \min(\text{alphabet}, \text{eval cap}))$ with
alphabet $2^{\min(N,62)}$, or $Q$ for grouped KT. For grouped exact it returns the
statistical cap. KT time is $O(B_{\rm eval}^2 N)$ even when memory fits.

---

## Counting memory

`estimate/counting.py::SymbolCounter`, used by `grouped_counting`
(`estimate/grouping.py::GroupedResponseCounter`).

- **Packing.** Given per-column radices, columns are packed into int64 words while the
  radix product stays below $2^{62}$ (`_word_plan`). Binary columns: 61 per word.
  Grouped counts (radices $n_j + 1$) usually fit one word. Packing runs on the
  symbols' device, so $8W$ bytes per observation cross to the CPU for $W$ words.
- **Reduction.** A stable argsort per word, least significant first, then segment
  sums (`_unique_counts`). The code comment measures about 40 bytes per row, against
  about 350 for `torch.unique(dim=0)` on a 2-D table.
- **Buffering.** Rows are reduced when the buffer exceeds both $2^{23}$ rows and the
  current unique table, so the table is re-sorted a logarithmic number of times:
  $O(n\log n)$ in total. Peak memory is the buffer plus the table of distinct symbols.

`torch.unique` is still used in:

- `LigandEnvironment.bind_receptors`: 1-D pair ids, $Rk$ entries, once per bind,
- `CellGrouping.__init__`: rows of $W$, once per run,
- `estimate/measurements.py::miller_madow_entropy`: the $(B_{\rm eval}, N)$ codes of
  `codeword_entropy`, cast to int64. This is the
  memory bound of binary-code counting, not `SymbolCounter`.
