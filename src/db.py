"""
db.py — SQLite lookup-table index over sweep run directories.

`runs.db` is a *derived* index — `config.json` is ground truth.
Delete and rebuild any time with `backfill`.  The table holds one row per run
directory; scalar config fields are columns; list-valued fields (conc_mean,
conc_mean, kernel_params, etc.) are skipped; metric means from test_results.json are added
as dynamic columns when first seen.

WAL mode + exponential-backoff retry make concurrent writes from parallel
sweeps safe.

CLI (all commands take the db path as the first positional argument):

    python -m src.db init       runs.db
    python -m src.db backfill   runs.db
    python -m src.db add-run    runs.db  path/to/run_dir
    python -m src.db sync       runs.db
    python -m src.db reconcile  runs.db  [--dry-run]
    python -m src.db delete     runs.db  relative/path  [--dry-run]
    python -m src.db move       runs.db  old/path  new/path
    python -m src.db alter      runs.db  add-col    col_name  TYPE
    python -m src.db alter      runs.db  remove-col col_name  [--dry-run]
    python -m src.db query      runs.db  [--where EXPR] [--cols c1,c2] [--limit N]

Examples:
    python -m src.db init      /app/data/fig1/runs.db
    python -m src.db backfill  /app/data/fig1/runs.db
    python -m src.db query     /app/data/fig1/runs.db --where "n_genes=5 AND full_array_entropy_mean > 4" --cols "path,n_genes,full_array_entropy_mean"
    python -m src.db reconcile /app/data/fig1/runs.db --dry-run
"""
import argparse
import json
import os
import re
import sqlite3
import subprocess
import time
import random
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import _TUPLE_FIELDS
from src.IO import SingleRunLoader

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_TS_RE = re.compile(r'(\d{8}_\d{6})')

# Config fields that are list-valued → skip in schema
_SKIP_CFG = _TUPLE_FIELDS | frozenset({"receptor_indices"})

# Fixed metadata columns; path is the PRIMARY KEY
_META_COLS: list[tuple[str, str]] = [
    ("path",          "TEXT PRIMARY KEY"),
    ("sweep_name",    "TEXT"),
    ("sweep_date",    "TEXT"),
    ("sweep_folder",  "TEXT"),
    ("receptor_type", "TEXT"),   # "homomer" | "heteromer"
    ("status",        "TEXT"),
    ("run_mtime",     "REAL"),
    ("git_hash",      "TEXT"),
    ("created",       "TEXT"),
    ("modified",      "TEXT"),
]

# Scalar SingleRunConfig fields → schema columns
_CFG_COLS: list[tuple[str, str]] = [
    ("n_families",                 "INTEGER"),
    ("n_ligands",                  "INTEGER"),
    ("latent_dim",                 "INTEGER"),
    ("family_spread",              "REAL"),
    ("average_family_distance",    "REAL"),
    ("environment_geometry",       "TEXT"),
    ("distribution_type",          "TEXT"),
    ("observation_noise_sigma",    "REAL"),
    # Schema change: rho_block removed, mu_sources / mu_ligands_per_source added.
    # Existing runs.db files require a fresh init or manual migration.
    ("n_presence_blocks",          "INTEGER"),
    ("mu_sources",                 "REAL"),
    ("mu_ligands_per_source",      "REAL"),
    ("block_shared_conc_mean",     "INTEGER"),
    ("conc_model_type",            "TEXT"),
    ("n_genes",                    "INTEGER"),
    ("k_sub",                      "INTEGER"),
    ("temperature",                "REAL"),
    ("affinity_kernel",            "TEXT"),
    ("batch_size",                 "TEXT"),
    ("entropy",                    "TEXT"),
    ("cov_weight",                 "REAL"),
    ("penalty_type",               "TEXT"),
    ("n_c_bins",                   "INTEGER"),
    ("epochs",                     "INTEGER"),
    ("lr",                         "REAL"),
    ("use_scheduler",              "INTEGER"),
    ("test_batch_size",            "TEXT"),
    ("eval_chunk_size",            "INTEGER"),
    ("n_receptors",                "INTEGER"),
    ("receptor_sampling_strategy", "TEXT"),
    ("receptor_sampling_seed",     "INTEGER"),
    ("use_interface_model",        "INTEGER"),
    ("recompute_backward",         "INTEGER"),
]

_ALL_BASE_COLS = _META_COLS + _CFG_COLS


# ──────────────────────────────────────────────────────────────────────────────
# SQLite helpers
# ──────────────────────────────────────────────────────────────────────────────

def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def _retry(fn, max_tries: int = 8, base_delay: float = 0.05) -> Any:
    """Retry fn() with exponential backoff on SQLite lock errors."""
    for attempt in range(max_tries):
        try:
            return fn()
        except sqlite3.OperationalError as e:
            if attempt < max_tries - 1 and "locked" in str(e).lower():
                time.sleep(base_delay * (2 ** attempt) + random.random() * 0.05)
            else:
                raise


def _existing_cols(conn: sqlite3.Connection) -> set[str]:
    return {r["name"] for r in conn.execute("PRAGMA table_info(runs)").fetchall()}


def _ensure_metric_cols(conn: sqlite3.Connection, metric_keys: list[str]) -> None:
    """ALTER TABLE to add any missing {metric}_mean REAL columns."""
    existing = _existing_cols(conn)
    for key in metric_keys:
        col = f"{key}_mean"
        if col not in existing:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} REAL")


def _ensure_schema_cols(conn: sqlite3.Connection) -> None:
    """Add any base schema columns missing from an older DB (e.g. receptor_type)."""
    existing = _existing_cols(conn)
    for name, typ in _ALL_BASE_COLS:
        bare_typ = typ.split()[0]   # strip PRIMARY KEY / NOT NULL etc.
        if name not in existing:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {name} {bare_typ}")


# ──────────────────────────────────────────────────────────────────────────────
# Metadata helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sweep_info(run_dir: str) -> tuple[Optional[str], Optional[str]]:
    """Walk ancestors upward; return (sweep_name, sweep_date) from the first
    directory component whose name matches the sweep timestamp pattern."""
    parts = os.path.abspath(run_dir).split(os.sep)
    for i in range(len(parts) - 1, 0, -1):
        m = _TS_RE.search(parts[i])
        if m:
            sweep_date = m.group(1)
            sweep_name = parts[i][:m.start()].rstrip("_")
            return sweep_name, sweep_date
    return None, None


def _git_hash() -> Optional[str]:
    """Short HEAD hash of the repo containing db.py, or None if not in a repo."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        r = subprocess.run(
            ["git", "-C", here, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _cfg_values(cfg) -> dict[str, Any]:
    """Extract scalar SingleRunConfig values; coerce bools → int for SQLite."""
    row: dict[str, Any] = {}
    for col, _ in _CFG_COLS:
        val = getattr(cfg, col, None)
        if isinstance(val, bool):
            val = int(val)
        elif isinstance(val, list):
            val = None
        row[col] = val
    return row


# ──────────────────────────────────────────────────────────────────────────────
# Row builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_row(run_dir: str, db_path: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Load run_dir and assemble a DB row dict.  Returns (row, None) or (None, reason)."""
    try:
        cfg = SingleRunLoader(run_dir).load_config()
    except Exception as e:
        return None, str(e)

    data_root            = os.path.dirname(os.path.abspath(db_path))
    rel_path             = os.path.relpath(os.path.abspath(run_dir), data_root)
    sweep_name, sweep_date = _sweep_info(run_dir)
    test_json            = os.path.join(run_dir, "test_results.json")
    status               = "complete" if os.path.exists(test_json) else "partial"
    now_iso              = datetime.now(timezone.utc).isoformat(timespec="seconds")

    row: dict[str, Any] = {
        "path":          rel_path,
        "sweep_name":    sweep_name,
        "sweep_date":    sweep_date,
        "sweep_folder":  rel_path.split("/")[0],
        "receptor_type": "homomer" if cfg.n_receptors is None else "heteromer",
        "status":        status,
        "run_mtime":     os.path.getmtime(run_dir),
        "git_hash":      _git_hash(),
        "created":       now_iso,
        "modified":      now_iso,
    }
    row.update(_cfg_values(cfg))

    # Metric means — one column per metric key, added dynamically
    if os.path.exists(test_json):
        with open(test_json) as f:
            data = json.load(f)
        for k, v in data.items():
            if isinstance(v, list) and v:
                row[f"{k}_mean"] = float(np.mean(v))

    return row, None


def _upsert(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    """INSERT OR REPLACE, preserving the original `created` timestamp."""
    existing = conn.execute(
        "SELECT created FROM runs WHERE path = ?", (row["path"],)
    ).fetchone()
    if existing:
        row["created"] = existing["created"]

    cols         = ", ".join(row.keys())
    placeholders = ", ".join("?" for _ in row)
    conn.execute(
        f"INSERT OR REPLACE INTO runs ({cols}) VALUES ({placeholders})",
        list(row.values()),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def init(db_path: str) -> None:
    """Create runs.db with the base schema.  No-op if the table already exists."""
    parent = os.path.dirname(os.path.abspath(db_path))
    if not os.path.isdir(parent):
        raise FileNotFoundError(f"Directory does not exist: {parent}")
    col_defs = ",\n    ".join(f"{name} {typ}" for name, typ in _ALL_BASE_COLS)
    ddl = f"CREATE TABLE IF NOT EXISTS runs (\n    {col_defs}\n)"
    def _do():
        with _connect(db_path) as conn:
            conn.execute(ddl)
    _retry(_do)
    print(f"Initialised: {db_path}")


def add_run(run_dir: str, db_path: str) -> None:
    """Upsert a single run directory into runs.db.

    Silent no-op when db_path doesn't exist — sweeps run unchanged on
    environments that haven't called `init` yet.
    """
    if not os.path.exists(db_path):
        return
    row, err = _build_row(run_dir, db_path)
    if row is None:
        return
    metric_keys = [k[:-5] for k in row if k.endswith("_mean")]
    def _do():
        with _connect(db_path) as conn:
            _ensure_schema_cols(conn)
            _ensure_metric_cols(conn, metric_keys)
            _upsert(conn, row)
    _retry(_do)


def backfill(db_path: str) -> None:
    """Crawl the directory containing runs.db and upsert every discovered run."""
    data_root = os.path.dirname(os.path.abspath(db_path))
    print(f"Scanning: {data_root}")
    count = skipped = 0
    for root, _dirs, files in os.walk(data_root):
        if "config.json" not in files:
            continue
        rel = os.path.relpath(root, data_root)
        if not _TS_RE.search(rel):
            print(f"  [skip] no timestamp in path: {rel}")
            skipped += 1
            continue
        row, err = _build_row(root, db_path)
        if row is None:
            print(f"  [skip] parse error in {rel}: {err}")
            skipped += 1
            continue
        metric_keys = [k[:-5] for k in row if k.endswith("_mean")]
        def _do(r=row, mk=metric_keys):
            with _connect(db_path) as conn:
                _ensure_schema_cols(conn)
                _ensure_metric_cols(conn, mk)
                _upsert(conn, r)
        _retry(_do)
        count += 1
    # Prune rows whose directories were deleted since the last backfill
    with _connect(db_path) as conn:
        all_paths = [r["path"] for r in conn.execute("SELECT path FROM runs").fetchall()]
    stale = [p for p in all_paths if not os.path.isdir(os.path.join(data_root, p))]
    if stale:
        def _prune(paths=stale):
            with _connect(db_path) as conn:
                conn.executemany("DELETE FROM runs WHERE path = ?", [(p,) for p in paths])
        _retry(_prune)
        print(f"Pruned {len(stale)} deleted run(s) from index.")
    print(f"Backfilled {count} run(s), skipped {skipped} → {db_path}")


def sync(db_path: str) -> None:
    """Like backfill but skips runs whose on-disk mtime matches the stored value."""
    with _connect(db_path) as conn:
        known = {
            r["path"]: r["run_mtime"]
            for r in conn.execute("SELECT path, run_mtime FROM runs").fetchall()
        }
    data_root = os.path.dirname(os.path.abspath(db_path))
    count = 0
    for root, _dirs, files in os.walk(data_root):
        if "config.json" not in files:
            continue
        rel = os.path.relpath(root, data_root)
        if not _TS_RE.search(rel):
            continue
        if known.get(rel) == os.path.getmtime(root):
            continue
        row, err = _build_row(root, db_path)
        if row is None:
            print(f"  [skip] parse error in {rel}: {err}")
            continue
        metric_keys = [k[:-5] for k in row if k.endswith("_mean")]
        def _do(r=row, mk=metric_keys):
            with _connect(db_path) as conn:
                _ensure_schema_cols(conn)
                _ensure_metric_cols(conn, mk)
                _upsert(conn, r)
        _retry(_do)
        count += 1
    print(f"Synced {count} run(s) → {db_path}")


def reconcile(db_path: str, dry_run: bool = False) -> None:
    """Flag rows whose run_dir no longer exists on disk as status='missing'."""
    data_root = os.path.dirname(os.path.abspath(db_path))
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT path FROM runs WHERE status != 'missing'"
        ).fetchall()
    missing = [
        r["path"] for r in rows
        if not os.path.isdir(os.path.join(data_root, r["path"]))
    ]
    if not missing:
        print("No missing runs found.")
        return
    tag = "[DRY RUN] " if dry_run else ""
    print(f"{tag}Missing ({len(missing)}):")
    for p in missing:
        print(f"  {p}")
    if not dry_run:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        def _do():
            with _connect(db_path) as conn:
                conn.executemany(
                    "UPDATE runs SET status='missing', modified=? WHERE path=?",
                    [(now, p) for p in missing],
                )
        _retry(_do)
        print(f"Marked {len(missing)} run(s) as missing.")


def delete_run(rel_path: str, db_path: str, dry_run: bool = False) -> None:
    """Remove a row from the index by relative path."""
    with _connect(db_path) as conn:
        found = conn.execute(
            "SELECT 1 FROM runs WHERE path = ?", (rel_path,)
        ).fetchone()
    if found is None:
        print(f"Not found: {rel_path}")
        return
    if dry_run:
        print(f"[DRY RUN] Would delete: {rel_path}")
        return
    def _do():
        with _connect(db_path) as conn:
            conn.execute("DELETE FROM runs WHERE path = ?", (rel_path,))
    _retry(_do)
    print(f"Deleted: {rel_path}")


def move_run(old_path: str, new_path: str, db_path: str) -> None:
    """Update the path column for a renamed/moved run directory."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    def _do():
        with _connect(db_path) as conn:
            n = conn.execute(
                "UPDATE runs SET path=?, modified=? WHERE path=?",
                (new_path, now, old_path),
            ).rowcount
        return n
    n = _retry(_do)
    if n == 0:
        print(f"Not found: {old_path}")
    else:
        print(f"Moved: {old_path!r} → {new_path!r}")


def alter_add(col: str, col_type: str, db_path: str) -> None:
    """Add a column to the runs table (idempotent)."""
    def _do():
        with _connect(db_path) as conn:
            if col in _existing_cols(conn):
                print(f"Column already exists: {col}")
                return
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {col_type}")
            print(f"Added column: {col} {col_type}")
    _retry(_do)


def alter_remove(col: str, db_path: str, dry_run: bool = False) -> None:
    """Rebuild the runs table without the given column."""
    with _connect(db_path) as conn:
        all_cols = [r["name"] for r in conn.execute("PRAGMA table_info(runs)").fetchall()]
    if col not in all_cols:
        print(f"Column not found: {col}")
        return
    keep = [c for c in all_cols if c != col]
    if dry_run:
        print(f"[DRY RUN] Would remove column '{col}' (keeping {len(keep)} columns)")
        return
    col_list = ", ".join(keep)
    conn = _connect(db_path)
    try:
        conn.execute(f"CREATE TABLE runs_new AS SELECT {col_list} FROM runs")
        conn.execute("DROP TABLE runs")
        conn.execute("ALTER TABLE runs_new RENAME TO runs")
        conn.commit()
    finally:
        conn.close()
    print(f"Removed column: {col}")


def query(
    db_path: str,
    where: Optional[str] = None,
    cols: Optional[str] = None,
    limit: Optional[int] = None,
) -> None:
    """Print matching rows as a pandas DataFrame."""
    col_str = cols if cols else "*"
    sql = f"SELECT {col_str} FROM runs"
    if where:
        sql += f" WHERE {where}"
    if limit:
        sql += f" LIMIT {limit}"
    with _connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn)
    if df.empty:
        print("(no results)")
        return
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 40)
    print(df.to_string(index=False))
    print(f"\n{len(df)} row(s)")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.db",
        description="Manage the runs.db SQLite index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, hlp in [
        ("init",      "Create runs.db schema"),
        ("backfill",  "Index all runs under db directory"),
        ("sync",      "Backfill only changed/new runs"),
    ]:
        s = sub.add_parser(name, help=hlp)
        s.add_argument("db", help="Path to runs.db")

    s = sub.add_parser("add-run", help="Index a single run directory")
    s.add_argument("db",      help="Path to runs.db")
    s.add_argument("run_dir", help="Absolute or relative path to the run directory")

    for name, hlp in [("reconcile", "Flag rows with missing disk paths")]:
        s = sub.add_parser(name, help=hlp)
        s.add_argument("db", help="Path to runs.db")
        s.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("delete", help="Remove a row by relative path")
    s.add_argument("db",   help="Path to runs.db")
    s.add_argument("path", help="Relative path as stored in DB")
    s.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("move", help="Update path for a renamed run directory")
    s.add_argument("db",       help="Path to runs.db")
    s.add_argument("old_path", help="Current path in DB")
    s.add_argument("new_path", help="New path")

    s = sub.add_parser("alter", help="Modify the table schema")
    s.add_argument("db", help="Path to runs.db")
    sub_a = s.add_subparsers(dest="alter_cmd", required=True)

    sa = sub_a.add_parser("add-col", help="Add a column (idempotent)")
    sa.add_argument("col");  sa.add_argument("col_type", metavar="TYPE")

    sa = sub_a.add_parser("remove-col", help="Rebuild table without a column")
    sa.add_argument("col")
    sa.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("query", help="Print rows matching criteria")
    s.add_argument("db",      help="Path to runs.db")
    s.add_argument("--where", default=None, metavar="EXPR",
                   help="SQL WHERE expression, e.g. \"n_genes=5 AND status='complete'\"")
    s.add_argument("--cols",  default=None,
                   help="Comma-separated column names (default: all)")
    s.add_argument("--limit", default=None, type=int)

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if   args.cmd == "init":      init(args.db)
    elif args.cmd == "backfill":  backfill(args.db)
    elif args.cmd == "add-run":   add_run(args.run_dir, args.db)
    elif args.cmd == "sync":      sync(args.db)
    elif args.cmd == "reconcile": reconcile(args.db, dry_run=args.dry_run)
    elif args.cmd == "delete":    delete_run(args.path, args.db, dry_run=args.dry_run)
    elif args.cmd == "move":      move_run(args.old_path, args.new_path, args.db)
    elif args.cmd == "alter":
        if   args.alter_cmd == "add-col":    alter_add(args.col, args.col_type, args.db)
        elif args.alter_cmd == "remove-col": alter_remove(args.col, args.db,
                                                dry_run=getattr(args, "dry_run", False))
    elif args.cmd == "query":
        query(args.db, where=args.where, cols=args.cols, limit=args.limit)


if __name__ == "__main__":
    main()
