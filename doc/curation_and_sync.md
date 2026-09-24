# Data curation and synchronization

The data workflow has two user-facing commands, both run from `opt_bin_resp/`:

```bash
python manage_data.py curate [goal]
python manage_data.py sync [goal]
```

Analyses read the run directories directly, so there is nothing to index and no
database to maintain. See `doc/data_pipeline.md` for how a stored run becomes a
plotted number.

## States

Execution and curation are independent:

- Execution is machine-owned: `running`, `complete`, `failed`, or `interrupted`.
  New sweeps store this single word in `.state`. Legacy sweeps are summarized by
  their number of `test_results.json` files.
- Curation is user-owned: implicit `review`, explicit `keep`, or explicit
  `delete`. Decisions apply to a complete timestamped sweep, not individual
  parameter points.

New `RunConfig` objects default to `curation_state="review"`. A script may set a
known decision in advance:

```python
RunConfig(
    # simulation fields ...
    curation_state="keep",
    curation_label="Figure 1 KT — 5 genes",
)
```

Usually it is easier to decide afterward with `curate`. Post-hoc decisions are
stored in the Git-tracked `curation.csv` and override the config default. A label
is required only for kept data.

## Review

```bash
python manage_data.py curate
python manage_data.py curate convergence
python manage_data.py curate fig1 --all
```

The prompt shows the execution result, completed/expected run count, size, and
current decision. It only edits `curation.csv`; it never deletes data.

## Synchronize

```bash
python manage_data.py sync
python manage_data.py sync fig1
```

The cluster (`leroy@10.187.172.7:/storage/leroy/data`) is authoritative for raw
data. `sync` performs these operations in order:

1. Show every `delete`-labelled sweep in scope and require the word `delete`.
2. Delete those exact sweep directories on the cluster first, then locally.
3. Mirror the cluster goal locally with `rsync --delete`. Mirroring also removes
   local sweeps the cluster does not have, so a dry run lists them first and
   requires the word `mirror` before anything is deleted.

After a deletion succeeds its temporary `delete` row is removed from
`curation.csv`; durable `keep` labels remain. This prevents later syncs from
reconfirming data that are already absent.

For unattended use, `--yes` supplies the deletion confirmation. Use it only after
reviewing `curation.csv`:

```bash
python manage_data.py sync fig1 --yes
```

Environment variables can override the endpoints for tests or another host:
`OCTOPUS_DATA_ROOT`, `OCTOPUS_CURATION_FILE`, `OCTOPUS_DATA_SERVER`, and
`OCTOPUS_REMOTE_DATA_ROOT`.

## Where curation shows up afterwards

`src/curation.py` is the only reader of `curation.csv` and of the `curation_state`
/ `curation_label` fields saved in a sweep's `sweep_config.json`. The CSV wins over
the config default. `manage_data.py` parses it strictly before rewriting it; the
run indexer parses it leniently, so one malformed row cannot stop a run from being
read.

`src.IO.index_goal`, and therefore `plotlib.load_runs`, carries two columns per
run so you can filter an analysis by your own decision:

- `curation_state`: `review`, `keep`, or `delete`, inherited from its sweep.
- `curation_label`: the short name you gave a kept sweep.

```python
from src.plotlib import load_runs
df = load_runs("fig1", curation_state="keep")
```
