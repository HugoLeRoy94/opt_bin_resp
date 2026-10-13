# gene_expression_threshold — the same sweep with a firing threshold

Single-variable change against `tasks/cells/gene_expression_mean`. The two tasks'
scripts are identical except for the readout block and the seeds.

```
mean       A_bc = S_bc                          cell reports its drive
threshold  A_bc = sigmoid((S_bc - theta) / T)   cell fires above a current

S_bc    abundance-weighted fraction of cell c's receptors open in sniff b
theta   ONE scalar shared by every cell, pinned to the median drive and re-pinned
        every CELL_RECALIBRATE_EVERY epochs; not fitted
T       cell sharpness, annealed to CELL_TEMPERATURE times the live drive spread
```

Two scripts, both self-contained:

- `scripts/heteromers.py` — full repertoires
- `scripts/homomers.py` — homomers only

```bash
bash tasks/run_remote.sh cells/gene_expression_threshold heteromers.py 0
bash tasks/run_remote.sh cells/gene_expression_threshold homomers.py 1
bash tasks/cells/gene_expression_threshold/sync.sh
```

## Verifying the single variable

```bash
diff tasks/cells/gene_expression_mean/scripts/heteromers.py \
     tasks/cells/gene_expression_threshold/scripts/heteromers.py
```

Everything that comes back should be the docstring, `READOUT`, the five
`CELL_THRESHOLD*` / `CELL_TEMPERATURE` / `CELL_N_MOLECULES` constants, the seeds and
the sweep name. If anything else appears, the comparison is confounded.

## The count-alphabet ceiling applies here too

Same table, same warning as the mean task: the `grouped_mi` ceiling is 14.0, 18.8,
18.6, 13.7, 5.0 bits across g=1..5, a hump peaking at g=2 that contains no biology.
Both scripts print it before training.

Why theta placement matters: with uniform homomer abundances a cell's drive can only
take the values k/g, so theta has to land in a gap between atoms. The analysis prints
where each run's theta sat relative to those atoms; a theta exactly on an atom is the
degenerate case.
