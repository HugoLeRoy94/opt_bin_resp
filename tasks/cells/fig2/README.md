# fig2 — information against genes per cell, by sampling strategy

One graph: MI against genes expressed per cell, with four curves. The curves differ
in how gene sets are drawn and in what each cell may assemble. Everything else —
environment, objective, grid, training — is identical across all four.

| script | gene sets | receptors |
|---|---|---|
| `scripts/complete_heteromers.py` | complete coverage, exactly g genes per cell | every receptor the genes allow |
| `scripts/complete_homomers.py` | complete coverage, exactly g genes per cell | homomers only |
| `scripts/bernoulli_heteromers.py` | each gene independent at probability p, size induced | every receptor the genes allow |
| `scripts/bernoulli_homomers.py` | each gene independent at probability p, size induced | homomers only |

## The two strategies

**Complete coverage** gives every cell one mandatory anchor gene, dealt round-robin,
then fills the rest at random. Every gene is expressed somewhere and every cell holds
exactly `g` genes. That requires cells to know what other cells expressed, which a
developing tissue has no mechanism for — it is a controlled baseline, not a model.

**Bernoulli** expresses each gene independently at the same probability `p`, with no
size law at all: the number of genes per cell is *induced* (Binomial), not chosen,
and a gene can end up expressed nowhere. `p` is solved so the zero-truncated mean
hits 1, 2, ... G, which puts both strategies on one x axis.

One caveat on the left edge: mean 1 needs `p -> 0`, and at `p -> 0` almost every draw
is an empty cell that gets rejected, so the sampler stalls. `bernoulli_p_for_mean`
floors `p` at 0.02, which lands the realised mean a little above 1 (1.02 at G=5).
The analysis plots the **realised** mean, so this shows up as a slightly shifted
first point rather than a silent error.

## The grid

G in {2,3,4,5} x C in {10,20,30,50} x genes-per-cell 1..G = **56 runs per script**,
224 in total. Largest receptor pool is 629 (G=5, g=5, heteromers).

## Environment

One family, 100 ligands, latent dimension 6, log-normal concentrations. Per-ligand
log10 concentration mean drawn uniformly over **2 decades** (`CONC_SPAN = 1.0`), with
**1 decade** of log10 spread per sniff (`CONC_STD = 1.0`), and **3 ligands per sniff**
on average (`MU_LIGANDS = 3.0`). Each of the four arms gets its own `WORLD_SEED` and
`CONC_SEED`, so no two curves share a world.

## Objective and estimators

`entropy="kt_mi"` trains. `grouped_counting` is the **only** measurement, and it
already carries everything the analyses need:

| key | meaning |
|---|---|
| `mutual_information_grouped_counting_mm` | MI |
| `response_entropy_grouped_counting_mm` | H(Y) |
| `conditional_entropy_response_grouped_counting` | H(Y\|X), analytic, not sampled |
| `grouped_counting_K_hat` / `_missing_mass` | alphabet coverage |

One estimator family, so `MI = H(Y) - H(Y|X)` holds exactly in the figures: the label
term cancels between `H(Y) = H(K) + d` and `H(Y|X) = H(K|X) + d`.

The labelled estimator `mutual_information_counting` is deliberately absent. Its
alphabet is `2^C`, which at C=50 is `10^15` symbols, so it is pinned near
`log2(samples)` and measures the budget rather than the array.

The runner builds its own cell grouping for `grouped_counting`, so a grouped loss is
not required.

**No per-epoch evaluation.** `per_epoch_measure=False`, so the only thing logged
during training is `loss` and `train_mutual_information` every 1% of epochs, reused
from the gradient step itself at zero extra cost. `analysis/convergence.py` reads
those columns.

## Running

```bash
bash tasks/run_remote.sh cells/fig2 complete_heteromers.py 0
bash tasks/run_remote.sh cells/fig2 complete_homomers.py 1
bash tasks/run_remote.sh cells/fig2 bernoulli_heteromers.py 2
bash tasks/run_remote.sh cells/fig2 bernoulli_homomers.py 3
bash tasks/run_remote.sh cells/fig2 evaluation_budget.py 0   # after the four finish
bash tasks/cells/fig2/sync.sh
```

`scripts/evaluation_budget.py` retrains nothing. It reloads each saved model and
re-measures the same grouped-counting metrics at 2^14 ... 2^22 input samples, nested
within a repeat so the ladder is monotone by construction, and writes `budgets.json`
next to each run.

## Analysis

| file | what it draws |
|---|---|
| `analysis/fig2.py` | THE figure: MI against genes per cell, four curves |
| `analysis/convergence.py` | training objective against epoch (with the last 20% magnified), and MI against evaluation budget |
| `analysis/entropies.py` | H(Y), H(Y\|X) and their difference, separately |
| `analysis/_load.py` | the shared loader; the only thing the three have in common |

Each analysis fixes `G, C` at the top — the figure is for one array size. Change the
two numbers to redraw for another corner of the grid.
