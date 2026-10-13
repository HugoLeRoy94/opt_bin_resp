# tasks/cells — what each script is

One rule: **every script is self-contained.** Every parameter of an experiment is a
constant at the top of the file that runs it. There is no command line, no config
builder, no shared design layer. If you want to know what a run did, open its script
and read down the page.

The single exception is `_lib/repertoires.py`, which holds the generators for gene
sets and receptor lists — combinatorics, no configuration. Everything else comes
from `src/`, which is the actual library.

## The scripts

| task | script | what it sweeps | readout | objective |
|---|---|---|---|---|
| `gene_expression_mean` | `heteromers.py` | genes/cell 1..5, full repertoires | mean | `grouped_mi`, exact |
| `gene_expression_mean` | `homomers.py` | genes/cell 1..5, homomers only | mean | `grouped_mi`, exact |
| `gene_expression_threshold` | `heteromers.py` | the same, with a firing threshold | threshold | `grouped_mi`, exact |
| `gene_expression_threshold` | `homomers.py` | the same, homomers only | threshold | `grouped_mi`, exact |
| `expression_law_mean` | `size_law_uniform.py` | mean genes/cell, FLAT size law | mean | `grouped_kt_mi`, counting |
| `expression_law_mean` | `size_law_exponential.py` | the same, EXPONENTIAL size law | mean | `grouped_kt_mi`, counting |
| `convergence` | `convergence.py` | nothing — one run against the log2(100) limit | mean | `kt_mi` |
| `equivalence` | `as_cells.py` / `as_receptors.py` | nothing — a numerical identity check | mean | `kt_mi` |

## What is a single-variable comparison against what

- `gene_expression_mean/heteromers.py` vs `gene_expression_mean/homomers.py`
  → the heteromer contribution. Same cells, same world settings; one arm may build
  mixed receptors, the other may not. They coincide at one gene per cell.
- `gene_expression_mean/heteromers.py` vs `gene_expression_threshold/heteromers.py`
  → the readout. **Every other constant in the two files is byte-identical.** Diff
  them if you doubt it.
- `expression_law_mean/size_law_uniform.py` vs `size_law_exponential.py`
  → the shape of the law for how many genes a cell expresses. The gene law is
  uniform in both.

Arms of a comparison never share a world: `WORLD_SEED` and `CONC_SEED` differ
between them on purpose, so a gap has to survive a change of environment. The one
exception is `equivalence`, where both scripts share a seed because it is a
numerical identity check and would be meaningless otherwise.

## Two traps these scripts now print at you

**The grouped count-alphabet ceiling.** `grouped_mi` keeps only how many cells fired
within each group of IDENTICAL cells. That is not an approximation: the count vector
is a sufficient statistic, the label term cancels between `H(Y)` and `H(Y|X)`, and
grouped MI equals the labelled-codeword MI exactly. So `log2(prod_j (n_j + 1))` is a
real ceiling on what the ARRAY can carry. It is not monotonic in genes per cell,
because the number of distinct repertoires available, `C(G, g)`, is not. For G=5,
C=30 complete coverage it runs

```
g=1  5 distinct   14.04 bits
g=2 10 distinct   18.81 bits
g=3  9 distinct   18.62 bits
g=4  5 distinct   13.66 bits
g=5  1 distinct    4.95 bits
```

— a hump peaking at g=2, tracking `C(5, g) = 5, 10, 10, 5, 1`. At g=5 every cell
expresses every gene, so thirty identical cells are thirty noisy copies of ONE
measurement and the array genuinely cannot carry more than log2(31) bits. The four
`gene_expression` scripts print this table before training. A curve sitting on that
line is saturating its combinatorial ceiling; a curve well below it is limited by
receptor chemistry instead. Both are results, but they are different results.

**The mixture regime.** `MU_LIGANDS` is the zero-truncated Poisson rate for ligands
present per sniff. `1e-6` means effectively exactly one. Anything above 1 is a
mixture, which is a different problem, not a tweak — and it silently breaks
`identity_channel` / `concentration_channel`, which estimate their split by grouping
sniffs with the same presence pattern and need those patterns to repeat. Every
script states its value in the header it prints.

## Flipping the environment

`MU_LIGANDS` and `CONC_SPAN` sit together near the top of every script. The default
everywhere is the single-ligand flat-concentration regime (`1e-6`, `0.0`). For the
mixture regime set `MU_LIGANDS = 3.0` and `CONC_SPAN = 2.0`, which gives about three
of 100 ligands per sniff with typical abundances spread over four decades. Change
them in every script you intend to compare, or the comparison is meaningless.

## `_archive/`

The old `_experiments.py` machinery and the scripts built on it, plus the 459-line
`expression_law.py` command line. Moved, not deleted. See `_archive/README.md`.
