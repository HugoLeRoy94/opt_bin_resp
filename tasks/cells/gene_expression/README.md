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

## Repeated comparisons and robustness experiments

The new scripts keep `cell_readout="mean"`, fresh environmental worlds, and
independent optimization at every point. They do not introduce a second threshold
or a frozen-parameter comparison. Each declares `RunConfig` directly and calls one
`SweepRunner`; metadata axes are expanded explicitly because lists are **zipped**.
The original two scripts remain available separately.

| Script in `scripts/` | Default design | Matching IDE analysis in `analysis/` |
|---|---|---|
| `replicates.py` | 5 optimizations per expression level; array size set by `--n_genes` / `--n_cells` | `replicates.py` (self-contained) |
| `scaling.py` | 3, 4, 5, 6 genes with 3 cells per gene; complete gene coverage; fixed original environment | `scaling.py` |
| `environment.py` | 6 genes, 18 cells; changes to ligand count, latent dimension, and mixture complexity | `environment.py` |
| `evaluation_budget.py` | Re-measure saved models at 4,096, 16,384, and 65,536 inputs, with 3 evaluation repeats | `evaluation_budget.py` |

The first three take `--condition heteromers` (default) or `--condition homomers`.
Corresponding strategies use the same expressed gene sets, but condition-specific
world seeds. Full repertoires use multinomial assembly weights; homomer-only
repertoires use uniform explicit-pool weights. All runs keep one chemical family
and zero observation noise.

### Coverage controls

`--coverage random` samples each cell's subset uniformly using vectorized random
priorities. Sets are nested as genes/cell increases. Realizations differ from the
legacy Python sampler, but preserve its uniform subset distribution at each level.

`--coverage complete` assigns one mandatory anchor gene per cell, distributed as
evenly as possible across genes. Other genes are chosen by random priorities.
Every gene is represented at every expression level; at one gene/cell,
multiplicities are balanced or differ by at most one. Above the baseline,
**coverage is guaranteed but expression counts need not be balanced**. This is a
controlled expression design, not random sampling conditioned on full coverage.
Every cell expresses exactly the requested number of genes. The scripts enforce
`N_cells > N_genes`.

Use `--coverage complete` on the replicate script to compare against its random
default. Use `--coverage random` on scaling/environment to measure the effect of
missing genes. Analyses report represented-gene counts and label designs separately.

### Environment profiles

| Profile | Ligands | Dimension | Family spread | `mu_ligands_per_source` |
|---|---:|---:|---:|---:|
| `base` | 100 | 6 | 0.1 | 1e-6 |
| `ligands` | 300 | 6 | 0.1 | 1e-6 |
| `dimension` | 100 | 12 | 0.1 × sqrt(6/12) | 1e-6 |
| `mixtures` | 100 | 6 | 0.1 | 3 |
| `combined` | 300 | 12 | 0.1 × sqrt(6/12) | 3 |

Increasing dimension holds `family_spread * sqrt(latent_dim)` fixed. Concentration
parameters remain mean=0, standard deviation=1. Presence uses the existing
positive truncated count sampler: `mu=3` is its rate, not exactly three ligands
per sniff. `--profiles` selects a subset. `--baseline_only` screens just the
one-gene baselines before running full expression sweeps. Increased baseline MI
is a hypothesis to test, not an assumed consequence of complexity.

### Launching and budgets

Inspect the plan without writing files or training, from `opt_bin_resp/`:

```bash
python3 tasks/cells/gene_expression/scripts/replicates.py --dry_run
python3 tasks/cells/gene_expression/scripts/scaling.py --dry_run
python3 tasks/cells/gene_expression/scripts/environment.py --baseline_only --dry_run
```

Defaults contain **15, 90, and 150 optimizations per condition**, respectively.
Environment baseline screening reduces the last count to 25. A pilot can use
fewer replicas, gene counts, or profiles. For example:

```bash
bash tasks/run_remote.sh cells/gene_expression replicates.py 0 -- --condition heteromers --replicates 5
bash tasks/run_remote.sh cells/gene_expression replicates.py 1 -- --condition homomers --replicates 5
```

Other examples inside the container (use the same remote wrapper on the cluster):

```bash
python3 /app/tasks/cells/gene_expression/scripts/scaling.py --condition heteromers --n_genes 3 4 --replicates 3
python3 /app/tasks/cells/gene_expression/scripts/scaling.py --condition homomers --n_genes 3 4 --replicates 3
python3 /app/tasks/cells/gene_expression/scripts/environment.py --condition heteromers --profiles base mixtures --baseline_only
python3 /app/tasks/cells/gene_expression/scripts/environment.py --condition homomers --profiles base mixtures --baseline_only
```

Shared controls: `--epochs` (5,000), `--batch_size` (4,096), `--test_batch_size`
(4,096), `--final_batch_size` (16,384), `--eval_chunk_size` (512), `--seed` (0),
and `--replicate_start` (0). `--n_genes` accepts one value for replicates, a list
for scaling/environment. Replicates offers `--n_cells`; the others offer
`--cells_per_gene` (3). Training logs its free objective with
`per_epoch_measure=False`; final measurements retain ten test repeats per model.

Training defaults to `--entropy grouped_mi`; `--entropy kt_mi` retains KT training
with the **same final grouped-MI measurement**. Compare biological strategies with
the same training objective. When sweeps with different protocols are loaded
together the analysis warns and names every setting that differs, marking which
ones it can keep apart and which it will pool; it does not refuse to load them.
Training objective and evaluation estimator are always kept apart, as the
`training_entropy` and `mi_estimator` columns, so two methods never collapse into
one mean. `full_array_entropy` retains its native meaning for the chosen loss;
the shared analysis uses `mutual_information_grouped` for comparable final values.

These larger designs explicitly set `--max_states 262144`, above the core
65,536-state default. Preflight checks the actual count alphabet from repertoire
multiplicities and prints the largest single `(batch_size, count_states)` float32
table. At the default training batch, the largest permitted table is 4 GiB,
before gradients and other tensors. Physics is chunked in pools of 128 with
gradient checkpointing. Distinct repertoires still make enumeration expensive;
there is no automatic estimator change. KT training does not remove the guard
needed by final exact grouped evaluation.

### Independent replicates and analysis

`experiment.json` records the full plan, expression sets, replicate IDs, world
seed, and comparison protocol. `cell_sampling_seed` labels each replicate's
repertoire realization; `cell_max_genes` records exact expression size and gives
distinct directory names even for very short runs. World seeds depend on the
experiment, condition, coverage design, root seed, and starting replicate.
One seeded sequence draws fresh worlds across the sweep; worlds are not matched.

To extend a campaign, use `--replicate_start 5 --replicates 5`, or a new `--seed`.
Exact reruns with identical seeds are not new independent replicates.

`analysis/replicates.py` is self-contained and names its sweeps explicitly in a
`SWEEPS` list at the top, with no "latest sweep" rule; it prints every sweep on
disk with its training objective, evaluation estimator and array size so the
choice is made with the alternatives in view. One curve is drawn per
(condition, array, method), so putting two estimators or two array sizes on one
figure never merges their means. Colour is the biological strategy, dashing the
estimator, marker the array size.

`scaling.py`, `environment.py` and `estimator_comparison.py` still share
`analysis/_shared.py`, which defaults to the latest sweep per condition and
coverage design and warns, naming each differing setting, when the selected
sweeps do not share a protocol.

Each optimization contributes **one mean over its test repeats**. Error bars
are SEM across independent optimizations, and printed `n` counts those runs.
Incomplete runs are excluded and incomplete coverage reported. Missing or
nonpositive baselines yield undefined retention (`NaN`); one replicate has no
across-run uncertainty estimate.

Analyses report total MI, response noise, hard-code entropy, represented genes,
pool size, and count-alphabet size. Retention is the ratio of mean MI to each
strategy's own mean one-gene MI in the same environment and array size; no error
bars are propagated for this ratio. Heteromer advantage is also reported as the
raw difference of environmental means, with unpaired SEM. Coverage diagnostics
expose gains due to recruiting previously missing genes. Figure saves and summary
CSV exports are commented out, like the original analysis.

Scaling curves use **genes expressed per cell** on the horizontal axis. Receptor
pool size counts distinct types in the **whole array**: with complete coverage,
homomers already have all `G` types at the one-gene baseline, even though types
per cell increase from 1 to `g`. Count-alphabet size is a separate computational
diagnostic: `S = product(n_j + 1)` for identical-cell group sizes `n_j`, with the
largest `S` across replicates plotted. `log2(S)` bounds count entropy; it does not
predict MI, which also subtracts conditional count entropy.

The environment baseline figure compares categorical profiles at one gene per
cell. At this point both strategies contain only homomers; differences come from
independent worlds and optimizations. A baseline-only campaign has no retention
curve across expression levels, and the final analysis cell explains this. Omit
`--baseline_only` for both conditions to obtain that curve.

### Grouped KT and counting for larger cell arrays

For side-by-side plots of saved exact and KT/counting campaigns, execute the
`# %%` cells in `analysis/estimator_comparison.py`. It automatically selects the
latest compatible replicates design with multiple methods, keeping array size,
environment and budgets fixed while allowing training/evaluation methods and
the enumeration guard to differ. It loads each method separately through the
existing validation and replicate accounting. `SWEEPS` can select explicit folders.

The first figure compares MI with independent-run SEM and individual run points;
the second compares full response entropy and conditional response entropy.
Axes and strategy colors match across panels. Missing strategies and incomplete
replicate counts are shown. Training and evaluation methods appear in panel
titles: their combined difference does not isolate optimization effects from
counting bias. Printed summaries include the counting unique-vector fraction.
Figures are saved beside the script as `estimator_comparison_mi.png` and
`estimator_comparison_entropies.png`; set `SAVE_FIGURES=False` to only display them.

Grouping is shared across exact enumeration, KT and sampled counting. The existing
defaults remain `--entropy grouped_mi --evaluation exact`. To avoid enumerating
the joint count alphabet during **both training and evaluation**, use:

```bash
bash tasks/run_remote.sh cells/gene_expression replicates.py 0 -- \
  --condition heteromers --coverage complete --n_genes 5 --n_cells 30 \
  --replicates 5 --entropy grouped_kt_mi --evaluation counting

bash tasks/run_remote.sh cells/gene_expression replicates.py 1 -- \
  --condition homomers --coverage complete --n_genes 5 --n_cells 30 \
  --replicates 5 --entropy grouped_kt_mi --evaluation counting
```

The same options work with `scaling.py` and `environment.py`. Grouped KT reuses
the ordinary KT kernels, with multiplicity weights: its values and gradients
match binary KT, with pairwise work proportional to the number of groups instead
of cells. It optimizes a lower bound, while exact grouped training optimizes the
empirical MI itself. Native entropy retains the labeled-response meaning.

`--evaluation counting` samples binomial group counts and stores frequencies of
observed integer vectors on CPU. It reports plug-in and Miller–Madow MI, analytical
conditional entropy, reconstructed full response entropy, and unique-count
coverage. It does not sample or store the irrelevant identities of firing cells.
The analysis uses **plug-in counting MI**, explicitly labels the estimator, and
does not pool it with exact results. Count entropy and labeled-response entropy
remain separate metrics; no exact results are overwritten or relabeled.

`--max_states` is enforced only if training or evaluation requests exact
enumeration. Thus grouped KT plus counting needs no increase to that guard.
Grouped KT plus the default exact evaluation still enforces it. Counting-only
measurement requests omit `full_array_entropy`, which would otherwise calculate
the training loss's native entropy (a KT bound for grouped KT).

Counting needs its own convergence test; convergence of exact evaluation only
checks input sampling. On saved models, select:

```bash
python3 tasks/cells/gene_expression/scripts/evaluation_budget.py \
  --run_dirs /path/to/saved/run --estimator counting \
  --budgets 4096 16384 65536 --repeats 3 --chunk_size 512
```

Use `--estimator exact` (the default) for an exact reference where enumeration
fits. The analysis retains the latest report **per model and estimator**, so
both can be plotted together. Counting errors include output sampling; small SEM
does not exclude systematic unseen-state bias, and negative MI estimates are
retained. The frequency table itself grows with the number of observed distinct
vectors. Neither KT nor counting removes receptor-physics costs.

### Input-budget convergence on saved models

Pass saved run folders, locally or on the cluster:

```bash
python3 tasks/cells/gene_expression/scripts/evaluation_budget.py \
  --run_dirs /path/to/run_YYYYMMDD_HHMMSS \
  --budgets 4096 16384 65536 --repeats 3 --chunk_size 512
```

This restores the saved receptor parameters and cell readout, without retraining.
Input budgets are nested within each repeat; repeats use separate seeds. It writes
a timestamped `grouped_evaluation_budget_*.json` alongside the original results,
without replacing them. `--device cpu` is supported; CUDA is used when available.
`--max_states` can override the saved config's guard.

The matching analysis selects the latest report per model and estimator, plots MI against
budget and `log2(B)`, and labels its errors as **evaluation-sampling** uncertainty.
It does not measure optimized-world variability or cure a training objective
saturated at its own `log2(batch_size)` ceiling. Main analyses flag final MI within
one bit of `log2(B)` as a reason to inspect this convergence. To test the training
budget, rerun a selected pilot with larger `--batch_size`, keeping its analysis
separate from the original protocol.

A focused convergence pilot for the six-gene, eighteen-cell sweep is:

1. Select saved models at **1, 2, 3, and 6 genes per cell**, for both strategies:
   baseline, observed peak, initial decline, and the identical-cell endpoint.
   Start with the first replicate at every point rather than choosing the best;
   extend to all independent replicates to check the environmental mean.
2. Evaluate at **4,096, 16,384, and 65,536 inputs**, with **3 repeats** and
   `--chunk_size 512`. Add **262,144** if MI has not stabilized. An operational
   target is a change below **0.02–0.05 bits** at the last doubling/quadrupling,
   small repeat scatter, and stable differences between expression levels.
   This tolerance is a practical choice, not a statistical guarantee.
3. Separately test optimization with training `--batch_size 4096`, `8192`, and
   `16384`, using the same selected design and **65,536 final evaluation inputs**.
   Keep the epoch count fixed to compare gradient quality at equal update count;
   larger batches also consume more total inputs and compute. Analyze each
   training budget separately: the loader warns about a mixed protocol and pools
   runs that differ only in batch size, because batch size has no column.

Evaluation chunks are aggregated before taking entropy: a chunk of 512 does not
impose a 9-bit ceiling when the total evaluation budget is larger. Increasing the
receptor pool does not itself require proportionally more input samples: all
receptor contributions are computed, and grouped conditional probabilities are
enumerated rather than estimated by sampling binary output patterns. Input
convergence depends on the stimulus-dependent response distribution. Being well
below `log2(B)` is useful but does not establish convergence.
