# Data curation and synchronization

The data workflow has two user-facing commands, both run from `opt_bin_resp/`:

```bash
python manage_data.py curate [goal]
python manage_data.py sync [goal]
```

`runs.db` is only a derived analysis cache. It is rebuilt automatically by
`sync`; normal work does not require database-management commands.

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
3. Mirror the cluster goal locally with `rsync --delete` (database files excluded).
4. Fully rebuild the local `runs.db`, including curation columns.

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

## Database columns

Analysis code continues to read `runs.db`. In addition to the existing `status`
(`complete` or `partial` per run), each row now has:

- `curation_state`: `review`, `keep`, or `delete`, inherited from its sweep.
- `curation_label`: the short name assigned to a kept sweep.

The low-level `python -m src.db ...` interface remains available for debugging,
but it is not part of the normal workflow.
