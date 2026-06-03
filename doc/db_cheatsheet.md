# runs.db cheat sheet

`runs.db` lives in your data root (e.g. `/app/data/fig1/runs.db`).  
All paths stored in the DB are **relative to that directory**.

---

## Setup

```bash
# Create the DB for the first time (also creates the directory if needed)
python -m src.db init /mnt/hcleroy/Postdoc2/octopus_smelling/opt_bin_resp/data/[..]/runs.db

# Index everything already on disk
python -m src.db backfill /app/data/fig1/runs.db
```

Once `init` has been run, sweeps auto-index each run as it finishes.

---

## Keeping the index up to date

| Goal | Command |
|---|---|
| Full rebuild from disk | `backfill runs.db` |
| Only touch new/changed runs | `sync runs.db` |
| Flag rows whose folder is gone | `reconcile runs.db [--dry-run]` |

---

## Cluster → local workflow

```bash
# 1. Copy a subset of run folders (never copy runs.db itself)
rsync -av --exclude='runs.db' user@cluster:/app/data/fig1/ /local/data/fig1/

# 2. Build a fresh local index over exactly what landed
python -m src.db init     /local/data/fig1/runs.db
python -m src.db backfill /local/data/fig1/runs.db
```

---

## Query

```bash
# All complete runs with 5 genes
python -m src.db query runs.db --where "n_genes=5 AND status='complete'"

# Specific columns, limit output
python -m src.db query runs.db \
  --where "entropy='renyi' AND n_receptors>10" \
  --cols "path,n_genes,n_receptors,full_array_entropy_mean" \
  --limit 20

# Sort (raw SQL in --where is fine)
python -m src.db query runs.db \
  --where "1=1 ORDER BY full_array_entropy_mean DESC" \
  --cols "path,full_array_entropy_mean" \
  --limit 10
```

---

## Manage rows

```bash
# Index one specific run
python -m src.db add-run runs.db homomers_20260603_143022/n_genes_5/run_20260603_143100

# Remove a row (does not delete files)
python -m src.db delete runs.db homomers_20260603_143022/n_genes_5/run_20260603_143100
python -m src.db delete runs.db some/path --dry-run   # preview first

# Update path after renaming a folder
python -m src.db move runs.db old/relative/path new/relative/path
```

---

## Schema changes

```bash
# Add a custom column
python -m src.db alter runs.db add-col my_note TEXT

# Remove a column (rebuilds table)
python -m src.db alter runs.db remove-col my_note --dry-run
python -m src.db alter runs.db remove-col my_note
```

Metric columns (`full_array_entropy_mean`, etc.) are added automatically
when new metrics appear in `test_results.json` — no manual `alter` needed.

---

## Quick reference

| Command | Effect |
|---|---|
| `init` | Create table (no-op if exists) |
| `backfill` | Upsert all runs found on disk |
| `sync` | Backfill only changed/new runs |
| `add-run` | Upsert one run directory |
| `reconcile` | Flag missing paths as `status='missing'` |
| `delete` | Remove a row from the index |
| `move` | Rename a path in the index |
| `alter add-col` | Add a column |
| `alter remove-col` | Rebuild table without a column |
| `query` | Print matching rows via pandas |
