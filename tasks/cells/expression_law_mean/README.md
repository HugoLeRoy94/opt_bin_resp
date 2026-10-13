# expression_law_mean — the shape of the expression law, at matched mean

Each cell draws its own gene set knowing nothing about any other cell: first HOW
MANY genes (the size law), then WHICH genes (the gene law). Nothing guarantees a
gene is expressed anywhere, which is the point — contrast the complete-coverage
design in `tasks/cells/gene_expression_mean`, which needs coordination a developing
tissue cannot do.

Two scripts, both self-contained, differing only in the size law:

- `scripts/size_law_uniform.py` — flat over {1..M}, M set so the mean hits the target
- `scripts/size_law_exponential.py` — p_i proportional to exp(lambda*i) over {1..G}

The gene law is uniform in both, so the pair isolates the shape of "how many genes".

```bash
python3 tasks/cells/expression_law_mean/scripts/size_law_uniform.py   # prints the design, then runs
bash tasks/run_remote.sh cells/expression_law_mean size_law_uniform.py 0
bash tasks/run_remote.sh cells/expression_law_mean size_law_exponential.py 1
```

Defaults: G in (3, 5), target means 1.0 to 3.0, 30 cells, 3 replicates, 24 runs per
arm. Both scripts print the full design table and the projected counting bias before
anything trains.

## Matched grid

A flat law over {1..M} has mean (M+1)/2, so it can only reach 1, 1.5, 2, ... and
only while M <= G. `design()` therefore skips any point where `2*mean - 1 > G`, in
BOTH scripts. The exponential law could reach those means, but then the two arms
would have different support and the curves would not be comparable. That skip is
two lines in the script, not a flag.

## Matched mean is not matched pool

The exponential law puts mass on the whole range {1..G} at any mean above 1, so a
few cells express every gene and the receptor pool saturates immediately: 629
receptors at G=5 for every mean above 1, against 53 -> 629 as the flat law's mean
rises. The exponential arm therefore costs more at the same grid point. That is the
laws differing, not a confound, but it cannot be equalized — matching the mean and
matching the pool are incompatible here.

## Why the estimator differs from gene_expression

Independent sampling leaves nearly every cell its own type, so the joint count
alphabet reaches about 10^7 and exact enumeration is out of reach. Training uses
`grouped_kt_mi` and evaluation uses sampled `grouped_counting`. Both are estimates.
Sampled counting is biased DOWNWARD and the bias falls only with
`FINAL_BATCH_SIZE`; the printed projection sizes it, and the Good-Turing missing
mass in the analysis is the real check.

Compare with `gene_expression_mean` only through KT arms, never against an
exact-enumeration number.

## Environment

Identical to `gene_expression_mean`: `MU_LIGANDS = 1e-6`, `CONC_SPAN = 0.0`, 100
ligands. Keep them in step across the two tasks or they are measuring different
problems.
