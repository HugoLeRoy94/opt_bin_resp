# Genes expressed per cell

This task extends `cells/convergence` using `RunConfig`'s existing series API:
`cell_size_pmf` is a **list of tuples**, one fixed size distribution per run.
`SweepRunner(config).execute()` executes the entire series; there is no manual
simulation loop. All list-valued axes in this runner are zipped, not crossed.

The defaults use ten cells, three available genes, 100 almost-singleton ligands,
interface binding, multinomial receptor abundance, `cell_readout="mean"`, and
`entropy="grouped_mi"`. Edit `MEAN_GENES` in `scripts/gene_expression.py` to choose the
series (default `[1, 2, 3]`, integer counts from 1 to `N_GENES`). Each cell
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
stops at three genes per cell; raising it increases memory and computation costs.

From `opt_bin_resp/` (after making the new task available on the cluster):

```bash
bash tasks/run_remote.sh cells/gene_expression gene_expression.py 0
bash tasks/run_remote.sh cells/gene_expression gene_expression_homomers.py 0
bash tasks/cells/gene_expression/sync.sh
```

`gene_expression_homomers.py` repeats the same expression-level sweep but passes
explicit `cell_receptors`: a cell expressing genes `{u, v}` contains only homomers
`[u,u,u,u,u]` and `[v,v,v,v,v]`. Thus no heteromers are assembled. Explicit
repertoires use the uniform weights defined by `CellArray`; the list of complete
repertoire configurations is consumed directly as a zipped `RunConfig` sweep axis.
Run folders use `receptors_per_cell_1`, `receptors_per_cell_2`, etc.; exact receptor
identities are saved in the JSON configs, so no separate naming parameter is needed.

Alternatively run the simulation directly in the container:

```bash
python3 /app/tasks/cells/gene_expression/scripts/gene_expression.py
python3 /app/tasks/cells/gene_expression/scripts/gene_expression_homomers.py
```

Open `analysis/gene_expression.py` in the IDE and execute its `# %%` cells, like
the convergence analysis. It selects the latest sweep, prints the final metrics,
then displays the across-sweep comparison and training trajectories. The figure
save lines are present but commented out. Hard codes are a diagnostic, not a
convergence requirement for the stochastic mean readout.

Curation and synchronization use the shared `gene_expression` data goal through
`manage_data.py`.


Both scripts now optimize exact empirical-input MI through binomial counts of
identical cells. `grouped_information` reports count entropy and its conditional
entropy, plus reconstructed full labeled-response entropies and MI.
`full_array_entropy` retains its meaning: it is full response entropy, not count
entropy. Receptor-mode calculations are unchanged. The default alphabet guard
`cell_grouped_max_states=65536` rejects excessively large enumerations; increase
it only with an appropriate input batch budget, or use `kt_mi` for larger alphabets.

For the current three expression levels, the count alphabets have 72, 80, and 11
states. Grouping requires exactly identical abundance rows, not merely similar
responses. Output enumeration is exact; environmental averages still require
independent input samples and simulation repetitions. No new simulations or
changes to saved results are implied by changing these scripts.

The analysis accepts both new grouped-MI results and older KT runs. KT bounds are
plotted only when present, and training curves identify their estimator. It prints
count entropy and its alphabet ceiling separately from MI. Final stochastic
counting remains an independent diagnostic; its finite-sample bias is unchanged.

See `doc/theory/08_environmental_entropy_limits.md` §7 for the cell-specific bounds
and §09.12 for the full metric names. These theory files live at the project root,
one directory above this repository.
