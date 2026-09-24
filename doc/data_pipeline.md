# From disk to plot: how simulation data is stored, indexed and read

This document answers one question: **when a figure appears, where did its numbers
come from?** It walks the whole path, from the files a simulation writes, through
synchronization with the cluster, to the exact arithmetic behind one plotted point.

Every number and file listing below was read off the real `data/` tree. Each
section ends with a command you can run to re-check it yourself, because this
document will drift and the code will not.

Related documents:
- `doc/curation_and_sync.md` — the two user-facing commands, `curate` and `sync`.
- `doc/theory/07_optimization_pipeline.md` — what happens *inside* a run.

---

## 1. The three levels on disk

```
data/                                      LEVEL 0
│
├── gene_expression/                       LEVEL 1: a GOAL
│   ├── replicates_homomers_complete_20260923_112624/    LEVEL 2: a SWEEP
│   │   ├── sweep_config.json                   written by the framework
│   │   ├── experiment.json                     written by the task script
│   │   ├── .state                              running|complete|failed|interrupted
│   │   │
│   │   └── cell_max_genes_5/receptors_per_cell_5/cell_sampling_seed_2/
│   │       family_spread_0.1/latent_dim_6/mu_ligands_per_source_1e-06/
│   │       n_cells_30/n_genes_5/n_ligands_100/
│   │       └── run_20260923_132052/            LEVEL 3: one RUN
│   │           ├── config.json
│   │           ├── stats.csv
│   │           ├── test_results.json
│   │           ├── best_model.pt
│   │           └── checkpoints/
│   │
│   └── replicates_heteromers_complete_20260923_112537/   another SWEEP
│
├── fig1/                                  another GOAL
└── convergence/                           another GOAL
```

**Goal** — a folder you choose, one per research question. Set by
`RunConfig.base_folder`. There are currently 14.

**Sweep** — one launch of one task script. The folder is named
`<sweep_name>_<launch timestamp>`. Everything that launch produced lives under it.

**Run** — one optimization: one environment, one cell array, trained, then measured.
A sweep holds many runs because it loops over parameter combinations.

### Why the path is so deep

The nesting is not decoration. `src/IO.py::_run_rel_path` builds one directory
level per swept parameter, named `<parameter>_<value>`, sorted alphabetically, with
a timestamped leaf so two identical parameter sets never collide.

So `cell_max_genes_5/.../n_cells_30/n_genes_5/run_20260923_132052` reads as:
5 genes expressed per cell, 30 cells, 5 genes in the pool. **The path is the
parameter record.** `ls` and `find` are enough to know what a run was.

```bash
# re-check: count runs per goal
for g in data/*/; do printf "%-36s %s\n" "$g" \
  "$(find "$g" -name config.json | wc -l)"; done
```

---

## 2. The five file types

### `sweep_config.json` — top of a sweep
Written by `SweepLogger._save_sweep_config` when the sweep starts. It is the
`RunConfig` you built in your script, as JSON. Swept fields are **lists**:

```
sweep_name         = replicates_homomers_complete
entropy            = grouped_mi
epochs             = 5000
n_genes            = [5, 5, 5, 5, 5, ...]          one entry per planned run
cell_max_genes     = [1, 2, 3, 4, 5, 1, 2, ...]
cell_sampling_seed = [0, 0, 0, 0, 0, 1, 1, ...]
curation_state     = review
```

The recipe for the whole sweep.

### `config.json` — every run directory
Same fields, but every value is now a **scalar**, because this is one point:

```
entropy            = grouped_mi
n_genes            = 5
cell_max_genes     = 5
cell_sampling_seed = 2
cell_gene_sets     = [[0,1,2,3,4], [0,1,2,3,4], ...]   the 30 cells
receptor_indices   = [[0,0,0,0,0], [1,1,1,1,1], ...]   the receptor pool
```

**This is ground truth.** Every other artifact in the pipeline is derived from
`config.json` plus `test_results.json` and can be regenerated from them.

Load it with `src.IO.SingleRunLoader(run_dir).load_config()`, which returns a real
`SingleRunConfig` and applies the backward-compatibility fixes for older runs
(renamed and removed fields). Parsing the JSON by hand skips those fixes.

### `stats.csv` — every run directory
The training history: `loss, train_mutual_information, lr, epoch`. The loop logs
about 100 rows regardless of the epoch count, so with `epochs=5000` each row is
roughly every 50 epochs. Used for convergence diagnostics. The gene_expression
figures do not read it.

### `test_results.json` — every run directory
**The source of every plotted number.** Written by `SimulationRunner.run` after
training, by re-measuring the trained model. Every value is a **list of 10**:

```
mutual_information_grouped   : [2.2408, 2.2410, 2.2436, 2.2435, 2.2383,
                                2.2419, 2.2396, 2.2436, 2.2418, 2.2447]
conditional_entropy_response : [13.664, 13.791, 13.598, ...]
full_array_entropy           : [15.922, 16.049, 15.858, ...]
response_evaluation_samples  : [16384, 16384, 16384, ...]
```

Ten, because `SimulationRunner._test` takes `test_epochs=10`: it draws a fresh
batch of inputs and measures, ten separate times, on **the same trained model**.

> **The distinction that matters most.** Those 10 numbers measure *evaluation
> sampling noise* on one model. They are NOT 10 replicates. Variation between
> independently optimized arrays is far larger and is captured only by running the
> same design point with different `cell_sampling_seed` values. Never quote the
> spread of the 10 as an uncertainty on a scientific claim.

### `best_model.pt` and `checkpoints/`
Trained weights. Needed only to re-measure a model later, for example by
`tasks/cells/gene_expression/scripts/evaluation_budget.py`. Analyses that only
read final numbers use `best_model.pt` purely as evidence the run finished.

---

## 3. `experiment.json`: the plan, written before execution

This file is **not** part of the framework. The task script writes it, at
`tasks/cells/gene_expression/_experiments.py::launch`:

```python
runner = SweepRunner(config)
(Path(runner.master_logger.sweep_root) / "experiment.json").write_text(json.dumps(manifest, indent=2))
runner.execute()
```

Build the sweep, record the plan at the top of its folder, then run. Contents:

| key | meaning |
|---|---|
| `experiment` | which task script produced this (`replicates`, `scaling`, ...) |
| `condition` | `heteromers` or `homomers` |
| `coverage` | the gene-set design (`complete` or `random`) |
| `world_seed` | seed for the whole sweep, derived from the arguments |
| `arguments` | the exact command-line flags used |
| `protocol.points` | the planned design points as tuples |
| `protocol.settings` | the 64 `RunConfig` values that define the protocol |
| `protocol_id` | 16-character hash of `protocol` |
| `rows` | every planned RUN, with its exact `cell_gene_sets` |

**Why it exists.** A sweep folder cannot say what was *supposed* to happen. If 3 of
25 runs crashed, the folder holds 22 runs and looks complete. `experiment.json`
lets the loader report "22/25 completed" and warn that some means rest on fewer
samples. It also lets the loader verify that the gene sets on disk match the ones
that were planned, catching a silently re-seeded design.

`protocol_id` is the comparison key. Two sweeps with the same hash ran the same
protocol. A different hash means something differed, and
`analysis/_shared.py::_warn_if_incompatible` prints exactly which settings, marking
each as kept separate or pooled.

---

## 4. There is no index. There used to be, and why it went

Analyses read the run directories. `src.IO.index_goal` crawls one goal folder and
returns a DataFrame with one row per run:

| block | columns |
|---|---|
| bookkeeping | `path`, `sweep_folder`, `sweep_name`, `sweep_date`, `run_timestamp`, `status`, `receptor_type`, `curation_state`, `curation_label`, `run_mtime` |
| config | every scalar field of the saved `SingleRunConfig`, under its own name |
| results | `<metric>_mean` per list-valued key of `test_results.json` |

List-valued config fields (`conc_mean`, `cell_gene_sets`, `receptor_indices`, ...)
do not fit one table cell and are omitted. Read them per run with
`SingleRunLoader`, as `plot_concentration_vs_family_spread.py::attach_cfg` does.

`plotlib.load_runs(goal, **filters)` wraps it with the filtering and the derived
`R` column. Nothing else is needed.

### Why indexing is fast enough to need no cache

`index_goal` uses `SingleRunLoader.load_config_dict`, which applies the
backward-compatibility fixes but does NOT construct `SingleRunConfig`. That matters:
`SingleRunConfig.__post_init__` calls `build_cell_array`, expanding every gene set
into its full receptor repertoire. Correct for running a simulation, ruinous when
repeated once per run while crawling. Indexing `gene_expression` takes 0.26 s this
way and took 8.3 s through the dataclass.

| goal | runs | index time |
|---|---|---|
| gene_expression | 312 | 0.28 s |
| fig1_2 | 650 | 0.25 s |
| all 10 goals | 1221 | well under 1 s each |

### The retired `runs.db`

Until recently each goal held a SQLite file, `data/<goal>/runs.db`, with the same
table, maintained incrementally by a per-run hook and rebuilt by `manage_data.py
sync`. It was removed, along with `src/db.py` (656 lines), because:

- **It saved 0.4 seconds.** Reading the index took 0.007 s against a 0.42 s crawl.
  A 60-fold ratio on a quantity nobody waits for.
- **No SQL feature was used.** `plotlib.load_runs` issued `SELECT * FROM runs` and
  did every filter in pandas. No `WHERE`, no index, no join.
- **It was a second source of truth that drifted.** Two writers disagreed about
  `sweep_name` for months: the per-run hook derived it from the run's own inner
  timestamp and wrote `"run"`, while the rebuild derived it from the sweep folder.
  Whichever ran last won, and no plot revealed it.
- **`git_hash` was actively misleading.** It recorded repository HEAD at *indexing*
  time, not at run time, so one rebuild stamped all 650 `fig1_2` runs with the same
  current commit. That column is gone rather than reinstated.

Equivalence was checked before deleting anything: for all 10 non-empty goals,
`load_runs` now returns the same row count and identical values on every column the
database had, up to float rounding (the old rebuild used `statistics.fmean`, the
crawl uses `numpy.mean`) and two columns the old schema stored with TEXT affinity
(`batch_size`, `test_batch_size`, now native ints). It returns MORE config columns
than before, because the old schema hand-listed 31 fields while the crawl keeps
every scalar one.

> **If crawling ever becomes slow**, cache the frame with
> `df.to_parquet(f"data/{goal}/index.parquet")` and delete that file whenever it
> looks stale. Do not reintroduce an incrementally-updated index. The cost of this
> pipeline was never speed, it was two descriptions of one fact.

## 5. Cluster synchronization

The cluster is authoritative for raw data. `curation.csv`, tracked in git, is
authoritative for the human keep/delete decision.

### `python manage_data.py curate [goal]`

Walks local sweeps and shows each one:

```
gene_expression/replicates_homomers_complete_20260923_112624
  execution: complete (25/25 runs complete)
  size:      412.3 MiB
  curation:  review
  [k]eep  [d]elete  [s]kip  [q]uit:
```

Your answer is appended to `curation.csv`:

```csv
path,state,label
fig1/ng10_20260724_132647,keep,Figure 1 KT — 10 genes
```

This command edits only that CSV. It never touches data.

### `python manage_data.py sync [goal] [--yes]`

1. List every sweep labelled `delete` in scope and require the word `delete`.
2. Delete them on the cluster first, then locally.
3. Mirror the cluster goal down with `rsync --delete`. Mirroring also removes local sweeps the cluster does not have, so a dry run
   lists those and requires the word `mirror` first.

`src.curation.read_curation` is the single parser of `curation.csv`. `manage_data` reads
it strictly before rewriting it; the indexer reads it leniently, so one malformed
row cannot stop a run from being indexed.

---

## 6. From disk to a plotted point

Worked example: `tasks/cells/gene_expression/analysis/replicates.py`.

`replicates.py` is deliberately SELF-CONTAINED: it imports no shared analysis
helper, so changing what it plots cannot change another figure. Duplication
between analysis scripts is accepted for that reason. `scaling.py`,
`environment.py` and `_method_comparison.py` still share `analysis/_shared.py`.

### Step 1 — select sweeps

A literal list of folder names at the top of the file:

```python
SWEEPS = [
    "replicates_heteromers_complete_20260923_112537",   # grouped_mi   / exact
    "replicates_homomers_complete_20260923_112624",     # grouped_mi   / exact
    "replicates_heteromers_complete_20260923_135950",   # grouped_kt_mi / counting
]
```

There is no "pick the latest" rule. Automatic selection is what once put a
KT-trained sweep on the same axis as three exactly-trained ones unnoticed. The
script first prints `available()`: every sweep on disk for this experiment, its
condition, coverage, training objective, evaluation estimator, array size and
planned run count, with the selected ones marked.

### Step 2 — read every finished run

Via `src.IO.SweepLoader.iter_run_dirs`, for each run holding `test_results.json`:

- load `config.json` as a `SingleRunConfig`
- check the run appears in `experiment.json`'s plan, keyed by
  (`cell_sampling_seed`, genes per cell)
- check its gene sets on disk match the planned ones
- pick the metric key for how it was measured: `mutual_information_grouped` for
  exact enumeration, `mutual_information_grouped_counting_plugin` for sampled
  counting
- **take the mean of the 10 test repeats**

Result: one row per run.

| column | meaning |
|---|---|
| `mi` | mean of the 10 `mutual_information_grouped` values |
| `response_noise` | mean of `conditional_entropy_response_grouped`, H(response given input) |
| `count_entropy` | mean of `grouped_count_entropy` |
| `full_entropy` | mean of `response_entropy_grouped`, H(response) |
| `genes_per_cell` | genes expressed per cell |
| `cell_sampling_seed` | which replicate |
| `genes_represented` | distinct genes appearing anywhere in the array |
| `training_entropy` | which objective was optimized |
| `mi_estimator` | which estimator produced the final number |

Verified on real data:

```
test_results.json['mutual_information_grouped'] =
  [2.2408, 2.2410, 2.2436, 2.2435, 2.2383, 2.2419, 2.2396, 2.2436, 2.2418, 2.2447]
mean of those 10          = 2.241878
column `mi`               = 2.241878      MATCH
```

### Step 3 — collapse replicates

Groups by `(condition, array, method, genes_per_cell)` and averages, where
`array` is `"G=5, C=30"` and `method` is `"grouped_mi / exact"`. Method is part of
the identity, so two estimators can sit on one figure without their means ever
being merged. Verified:

```
the 5 runs at homomers / 5 genes / 5 genes-per-cell:
  seed 0 -> 2.241878
  seed 1 -> 2.259692
  seed 2 -> 2.259334
  seed 3 -> 2.260969
  seed 4 -> 2.261860

mi_mean = mean of those 5          = 2.256747
mi_sd   = standard deviation       = 0.008373
mi_sem  = mi_sd / sqrt(5)          = 0.003745
```

**There are two averaging stages.** Ten test repeats collapse to one number per
run, removing evaluation noise. Then five independently optimized runs collapse to
one point. **The error bar on the figure is the second one**, the spread across
independent optimizations, which is the honest uncertainty.

`retained` is this point's `mi_mean` divided by the `mi_mean` at
`genes_per_cell == 1` **of the same curve**, so a method or array offset cannot
leak into it. 1.0 means as good as one gene per cell.

### Step 4 — the panels

Before drawing anything the script prints every curve: its x values, its y values,
its error bars and its run count per point. Nothing reaches the figure without
being printed first. It then draws a 2 by 2 figure, x axis always `genes_per_cell`:

| panel | y | meaning |
|---|---|---|
| top left | `mi_mean` with `mi_sem` bars | bits carried by the array |
| top right | `retained` | the same as a fraction of the one-gene baseline |
| bottom left | `noise_mean` | H(response given input), the array's response noise |
| bottom right | `represented_mean` | distinct genes present in the array |

Then a scatter of every individual run, unaveraged, so the replicate-to-replicate
spread the error bars summarise is directly visible.

Curve style encodes the distinction that matters: **colour is the biological
strategy** (heteromers against homomers, what the project actually asks about),
**dashing is the estimator method** (an artefact of measurement, not biology), and
**marker is the array size**. A method difference can then never be mistaken for a
biological one at a glance.

### Step 5 — read the line labels before believing the figure

A real example from the current data, with four sweeps auto-selected:

```
LINE 1: heteromers, complete, grouped counting (plug-in), grouped_kt_mi
        x: [1,2,3,4,5]  y: [5.413, 6.011, 3.896, 3.667, 2.353]   array G=5, C=30
LINE 2: heteromers, random,   exact grouped, grouped_mi
        x: [1,2,3]      y: [3.162, 3.097, 1.749]                 array G=3, C=10
LINE 3: homomers,   complete, exact grouped, grouped_mi
        x: [1,2,3,4,5]  y: [5.428, 5.044, 4.287, 3.293, 2.257]   array G=5, C=30
LINE 4: homomers,   random,   exact grouped, grouped_mi
        x: [1,2,3]      y: [3.176, 2.583, 1.732]                 array G=3, C=10
```

Only **line 2 against line 4** is a valid heteromer-versus-homomer comparison:
same coverage, same array, same objective, same estimator.

**Line 1 against line 3 is not.** Same array size, but line 1 trained with
`grouped_kt_mi` and measured by sampled counting, line 3 trained with `grouped_mi`
and measured by exact enumeration. Two things changed at once, so a gap between
them cannot be attributed to biology.

`estimator_comparison.py` exists for exactly that case: it pairs sweeps that differ
only in method and puts them in separate panels.

---

## 7. Summary

- `config.json` and `test_results.json` are ground truth. Everything else is
  derived and regenerable.
- There is no stored index. `src.IO.index_goal` crawls the directories in
  well under a second per goal. See section 4 for why the old `runs.db` went.
- `experiment.json` is written by the task script and records the plan, making
  incomplete sweeps detectable.
- Numbers are averaged twice: 10 test repeats per run, then N runs per point. The
  plotted error bar is the second.
- Automatic sweep selection can silently place incomparable sweeps on one axis.
  Check the line labels, or name the sweep folders explicitly.
