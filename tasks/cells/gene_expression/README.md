# Genes expressed per cell

This task extends `cells/convergence` using `RunConfig`'s existing series API:
`cell_size_pmf` is a **list of tuples**, one fixed size distribution per run.
`SweepRunner(config).execute()` executes the entire series; there is no manual
simulation loop. All list-valued axes in this runner are zipped, not crossed.

The defaults use eight cells, ten available genes, 100 almost-singleton ligands,
interface binding, multinomial receptor abundance, `cell_readout="mean"`, and
`entropy="kt_mi"`. Edit `MEAN_GENES` in `scripts/gene_expression.py` to choose the
series (default `[1, 2, 3, 4, 5]`, integer counts from 1 to `N_GENES`). Each cell
expresses exactly that many genes. As in the other simulation scripts, `main()`
declares `RunConfig` directly and passes it to `SweepRunner`: no config builder or
custom PMF helper. For non-integer expected sizes, edit `cell_size_pmf` directly
with the desired distributions; the analysis reports requested and realized means.

The fixed cell-sampling seed makes repertoires reproducible. Gene choices are
independent across cells; duplicate repertoires remain possible and are counted in
the analysis. The torch seed initializes the whole sweep, not each run separately:
each run starts a fresh world and optimizer (`warm_start=False`), but worlds are
not matched across expression levels. These are exploratory comparisons, not
replicate-averaged estimates of an expression effect. Test repeats in a run average
measurement noise, not optimization or repertoire variability.

Larger expressed gene sets generate rapidly growing receptor pools. The default
stops at five genes per cell; raising it increases memory and computation costs.

From `opt_bin_resp/` (after making the new task available on the cluster):

```bash
bash tasks/run_remote.sh cells/gene_expression gene_expression.py 0
bash tasks/cells/gene_expression/sync.sh
python3 tasks/cells/gene_expression/analysis/gene_expression.py
```

Alternatively run the simulation directly in the container:

```bash
python3 /app/tasks/cells/gene_expression/scripts/gene_expression.py
```

Analysis selects the newest sweep, prints skipped incomplete runs, and saves
`summary.csv`, `information_vs_genes.png`, and `training_curves.png` under
`figures/<sweep-name>/`. It plots identity MI, KT MI bounds, counting MI, conditional
response entropy, hard-code entropy, and receptor-pool size. Hard codes are a
diagnostic, not a convergence requirement for the stochastic mean readout.

```bash
python3 tasks/cells/gene_expression/analysis/gene_expression.py \
  --sweep data/gene_expression/cell_gene_expression_YYYYMMDD_HHMMSS --no-show
```

Analysis reads saved configs/results directly and does not require torch or an
up-to-date database. Curation and synchronization use the shared `gene_expression`
data goal through `manage_data.py`.
