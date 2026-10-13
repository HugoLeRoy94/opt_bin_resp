# gene_expression_mean — information against genes expressed per cell, mean readout

Two scripts, both self-contained:

- `scripts/heteromers.py` — each cell assembles EVERY receptor its genes allow,
  homomers and heteromers, weighted by multinomial abundance
- `scripts/homomers.py` — the same cells pinned to the homomers of their genes

The gap between the two curves is the heteromer contribution. At one gene per cell
they coincide by construction.

```bash
bash tasks/run_remote.sh cells/gene_expression_mean heteromers.py 0
bash tasks/run_remote.sh cells/gene_expression_mean homomers.py 1
bash tasks/cells/gene_expression_mean/sync.sh
```

Then open `analysis/gene_expression.py` and run its `# %%` cells.

## The design

G=5 genes, C=30 cells, **complete coverage** (every gene expressed by at least one
cell, via one mandatory anchor gene per cell), genes/cell swept over 1..5. One seed
for all levels, so the arrays are NESTED: the g=2 array is the g=1 array with one
gene added per cell. Environment: 100 ligands, one per sniff, flat concentration
channel. Objective `grouped_mi` with exact enumeration of joint count states.

Complete coverage requires cells to know what other cells expressed, which a
developing tissue has no mechanism for. It is a controlled baseline, not a
biological model. `tasks/cells/expression_law_mean` is the independent version.

## Read the count-alphabet table before believing a bump

Both scripts print this before training:

```
 g  cells  distinct  count states  grouped MI cap  train table
 1     30         5        16,807        14.04 b      0.26 GiB
 2     30        10       460,800        18.81 b      7.03 GiB
 3     30         9       403,200        18.62 b      6.15 GiB
 4     30         5        12,936        13.66 b      0.20 GiB
 5     30         1            31         4.95 b      0.00 GiB
```

`grouped_mi` keeps only how many cells fired within each group of identical cells,
so it cannot exceed log2 of that alphabet. That is not an estimator limitation:
grouping is lossless for MI (the count vector is a sufficient statistic and the
label term cancels), so this is a real ceiling on what the array can carry.

The cap is a hump peaking at g=2 because the number of distinct repertoires
available is `C(5, g) = 5, 10, 10, 5, 1`. At g=5 all thirty cells express all five
genes, so they are thirty noisy copies of one measurement and log2(31) bits is all
there is. Read the measured curve against this line: ON it means the array is
saturating its combinatorial ceiling, BELOW it means receptor chemistry is the
binding constraint.

The 7 GiB column is the fp32 (batch, states) training table, before gradients.

## Relationship to gene_expression_threshold

`gene_expression_threshold/scripts/heteromers.py` is this file with `READOUT =
"threshold"` and the five threshold constants, and nothing else changed. Diff them.
Any difference between the curves is the readout.

## Flipping to the mixture regime

`MU_LIGANDS` (ligands per sniff) and `CONC_SPAN` (per-ligand log10 concentration
mean, drawn in +/- this) sit at the top of each script. The default `1e-6` / `0.0`
is one ligand per sniff at a flat concentration. `3.0` / `2.0` gives about three of
100 ligands per sniff over four decades of abundance. Change both scripts together,
and note that the mixture regime makes `identity_channel` and
`concentration_channel` meaningless — they need presence patterns to repeat.
