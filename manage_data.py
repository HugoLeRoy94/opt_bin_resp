#!/usr/bin/env python3
"""Two-command sweep curation and cluster/local synchronization.

User workflow
-------------
    python manage_data.py curate [goal]
    python manage_data.py sync [goal] [--yes]

The cluster is authoritative for raw data.  ``curation.csv`` is authoritative
for the human keep/delete decision.  ``runs.db`` is rebuilt automatically.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sqlite3
from statistics import fmean
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parent
LOCAL_DATA = Path(os.environ.get("OCTOPUS_DATA_ROOT", ROOT / "data")).resolve()
CURATION_FILE = Path(os.environ.get("OCTOPUS_CURATION_FILE", ROOT / "curation.csv")).resolve()
SERVER = os.environ.get("OCTOPUS_DATA_SERVER", "leroy@10.187.172.7")
REMOTE_DATA = os.environ.get("OCTOPUS_REMOTE_DATA_ROOT", "/storage/leroy/data").rstrip("/")

STATES = {"keep", "delete"}
_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_TIMESTAMP_RE = re.compile(r"\d{8}_\d{6}")
SSH_OPTIONS = [
    "-o", "BatchMode=yes",
    "-o", "ControlMaster=auto",
    "-o", "ControlPersist=60",
    "-o", "ControlPath=/tmp/octopus-data-ssh-%C",
]


def _validate_component(value: str, what: str) -> str:
    if not _COMPONENT_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"Unsafe {what}: {value!r}")
    return value


def _validate_sweep_path(value: str) -> str:
    value = value.strip().strip("/")
    parts = value.split("/")
    if len(parts) != 2:
        raise ValueError(f"Sweep path must be GOAL/SWEEP: {value!r}")
    _validate_component(parts[0], "goal")
    _validate_component(parts[1], "sweep")
    if not _TIMESTAMP_RE.search(parts[1]):
        raise ValueError(f"Sweep path has no timestamp: {value!r}")
    return value


def load_curation() -> dict[str, tuple[str, str]]:
    decisions: dict[str, tuple[str, str]] = {}
    if not CURATION_FILE.exists():
        return decisions
    with CURATION_FILE.open(newline="") as f:
        for row in csv.DictReader(f):
            path = _validate_sweep_path(row.get("path", ""))
            state = row.get("state", "").strip()
            label = row.get("label", "").strip()
            if state not in STATES:
                raise ValueError(f"Invalid curation state {state!r} for {path}")
            if state == "keep" and not label:
                raise ValueError(f"A kept sweep needs a label: {path}")
            decisions[path] = (state, label)
    return decisions


def save_curation(decisions: dict[str, tuple[str, str]]) -> None:
    CURATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CURATION_FILE.with_suffix(CURATION_FILE.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=("path", "state", "label"), lineterminator="\n"
        )
        writer.writeheader()
        for path in sorted(decisions):
            state, label = decisions[path]
            writer.writerow({"path": path, "state": state, "label": label})
    os.replace(tmp, CURATION_FILE)


def _all_goals(decisions: dict[str, tuple[str, str]]) -> list[str]:
    goals = {
        p.name for p in LOCAL_DATA.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    } if LOCAL_DATA.exists() else set()
    goals.update(path.split("/", 1)[0] for path in decisions)
    return sorted(goals)


def _selected_goals(raw: Iterable[str], decisions: dict[str, tuple[str, str]]) -> list[str]:
    raw = list(raw)
    if not raw or raw == ["all"]:
        return _all_goals(decisions)
    if "all" in raw:
        raise ValueError("Use either 'all' or explicit goal names, not both.")
    return sorted({_validate_component(goal, "goal") for goal in raw})


def _sweeps(goals: Iterable[str]) -> list[Path]:
    found: list[Path] = []
    for goal in goals:
        root = LOCAL_DATA / goal
        if not root.is_dir():
            continue
        found.extend(
            child for child in root.iterdir()
            if child.is_dir() and _TIMESTAMP_RE.search(child.name)
        )
    return sorted(found)


def _tree_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _human_size(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _execution_summary(sweep: Path) -> tuple[str, int, int]:
    configs = list(sweep.rglob("config.json"))
    complete = sum((path.parent / "test_results.json").is_file() for path in configs)
    marker = sweep / ".state"
    if marker.is_file():
        state = marker.read_text().strip()
    elif configs and complete == len(configs):
        state = "complete (legacy)"
    elif configs:
        state = "incomplete (legacy)"
    else:
        state = "artifacts only"
    return state, complete, len(configs)


def _configured_curation(sweep: Path) -> tuple[str, str]:
    config_path = sweep / "sweep_config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError, TypeError):
        return "review", ""
    state = config.get("curation_state", "review")
    label = str(config.get("curation_label", "")).strip()
    if state == "keep" and label:
        return state, label
    if state == "delete":
        return state, label
    return "review", ""


def _effective_decisions(
    goals: Iterable[str], overrides: dict[str, tuple[str, str]]
) -> dict[str, tuple[str, str]]:
    """Merge config defaults with the tracked post-hoc overrides."""
    effective: dict[str, tuple[str, str]] = {}
    for sweep in _sweeps(goals):
        key = f"{sweep.parent.name}/{sweep.name}"
        configured = _configured_curation(sweep)
        if configured[0] in STATES:
            effective[key] = configured
    effective.update(overrides)
    return effective


def curate(goals: list[str], show_all: bool) -> None:
    decisions = load_curation()
    selected = _selected_goals(goals, decisions)
    effective = _effective_decisions(selected, decisions)
    sweeps = _sweeps(selected)
    if not show_all:
        sweeps = [
            sweep for sweep in sweeps
            if f"{sweep.parent.name}/{sweep.name}" not in effective
        ]
    if not sweeps:
        print("No sweeps need review.")
        return

    for sweep in sweeps:
        key = f"{sweep.parent.name}/{sweep.name}"
        state, complete, total = _execution_summary(sweep)
        current, label = effective.get(key, ("review", ""))
        print(f"\n{key}")
        print(f"  execution: {state} ({complete}/{total} runs complete)")
        print(f"  size:      {_human_size(_tree_size(sweep))}")
        print(f"  curation:  {current}" + (f" — {label}" if label else ""))
        choice = input("  [k]eep  [d]elete  [s]kip  [q]uit: ").strip().lower()
        if choice == "q":
            break
        if choice == "s" or not choice:
            continue
        if choice == "k":
            label = input("  short label: ").strip()
            if not label:
                print("  skipped: a kept sweep needs a label")
                continue
            decisions[key] = ("keep", label)
        elif choice == "d":
            decisions[key] = ("delete", "")
        else:
            print("  unknown choice; skipped")
            continue
        save_curation(decisions)
        print(f"  saved: {decisions[key][0]}")


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=check)


def _remote_exists(path: str) -> bool:
    result = _run(
        ["ssh", *SSH_OPTIONS, SERVER, "test", "-e", path],
        check=False,
    )
    return result.returncode == 0


def _remote_goals() -> list[str]:
    command = (
        f"find {shlex.quote(REMOTE_DATA)} -mindepth 1 -maxdepth 1 "
        "-type d -printf '%f\\n'"
    )
    result = subprocess.run(
        ["ssh", *SSH_OPTIONS, SERVER, command],
        check=True, capture_output=True, text=True,
    )
    return sorted(
        _validate_component(line.strip(), "remote goal")
        for line in result.stdout.splitlines() if line.strip()
    )


def _delete_decisions(
    decisions: dict[str, tuple[str, str]], goals: list[str], assume_yes: bool
) -> list[str]:
    targets = sorted(
        path for path, (state, _label) in decisions.items()
        if state == "delete" and path.split("/", 1)[0] in goals
    )
    if not targets:
        return []

    print("\nSweeps labelled delete (cluster and local):")
    for path in targets:
        print(f"  {path}")
    if not assume_yes:
        if not sys.stdin.isatty():
            raise RuntimeError("Deletion needs an interactive confirmation or --yes.")
        answer = input("Type 'delete' to continue: ").strip()
        if answer != "delete":
            raise RuntimeError("Synchronization cancelled; no data were deleted.")

    # Delete remotely first in one validated call. If SSH fails, local data remain
    # available and sync stops. Missing remote paths are harmless and idempotent.
    validated = [_validate_sweep_path(rel) for rel in targets]
    remote_targets = [f"{REMOTE_DATA}/{rel}" for rel in validated]
    _run(["ssh", *SSH_OPTIONS, SERVER, "rm", "-rf", "--", *remote_targets])
    for rel in validated:
        print(f"  deleted/absent remote: {rel}")

    for rel in targets:
        local = LOCAL_DATA / rel
        if local.is_dir():
            shutil.rmtree(local)
            print(f"  deleted local:  {rel}")
    return targets


def _sync_goal(goal: str) -> None:
    remote = f"{REMOTE_DATA}/{goal}"
    if not _remote_exists(remote):
        print(f"skip {goal}: no cluster directory")
        return
    local = LOCAL_DATA / goal
    local.mkdir(parents=True, exist_ok=True)
    print(f"\nsync {goal}")
    _run([
        "rsync", "-az", "--delete", "--exclude=/runs.db",
        "--exclude=/runs.db-wal", "--exclude=/runs.db-shm",
        "-e", "ssh " + " ".join(SSH_OPTIONS),
        f"{SERVER}:{remote}/", f"{local}/",
    ])

def _rebuild_goal(goal: str) -> None:
    local = LOCAL_DATA / goal
    local.mkdir(parents=True, exist_ok=True)
    decisions = _effective_decisions([goal], load_curation())
    rows: list[dict[str, object]] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    git = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip() or None

    for config_path in local.rglob("config.json"):
        try:
            config = json.loads(config_path.read_text())
        except (OSError, ValueError, TypeError) as exc:
            print(f"  skip invalid config {config_path}: {exc}")
            continue
        run_dir = config_path.parent
        rel = run_dir.relative_to(local).as_posix()
        sweep_folder = rel.split("/", 1)[0]
        match = _TIMESTAMP_RE.search(sweep_folder)
        test_path = run_dir / "test_results.json"
        curation_state, curation_label = decisions.get(
            f"{goal}/{sweep_folder}", ("review", "")
        )
        row: dict[str, object] = {
            "path": rel,
            "sweep_name": sweep_folder[:match.start()].rstrip("_") if match else sweep_folder,
            "sweep_date": match.group(0) if match else None,
            "sweep_folder": sweep_folder,
            "receptor_type": "homomer" if config.get("n_receptors") is None else "heteromer",
            "status": "complete" if test_path.is_file() else "partial",
            "curation_state": curation_state,
            "curation_label": curation_label,
            "run_mtime": run_dir.stat().st_mtime,
            "git_hash": git,
            "created": now,
            "modified": now,
        }
        for key, value in config.items():
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) and not isinstance(value, (list, dict)):
                row.setdefault(key, int(value) if isinstance(value, bool) else value)
        if test_path.is_file():
            try:
                results = json.loads(test_path.read_text())
                for key, values in results.items():
                    if (re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                            and isinstance(values, list) and values):
                        row[f"{key}_mean"] = fmean(float(value) for value in values)
            except (OSError, ValueError, TypeError) as exc:
                print(f"  metrics unavailable for {rel}: {exc}")
        rows.append(row)

    base_types = {
        "path": "TEXT PRIMARY KEY", "sweep_name": "TEXT", "sweep_date": "TEXT",
        "sweep_folder": "TEXT", "receptor_type": "TEXT", "status": "TEXT",
        "curation_state": "TEXT", "curation_label": "TEXT", "run_mtime": "REAL",
        "git_hash": "TEXT", "created": "TEXT", "modified": "TEXT",
    }
    keys = list(base_types)
    extras = sorted({key for row in rows for key in row if key not in base_types})
    keys.extend(extras)

    def sql_type(key: str) -> str:
        if key in base_types:
            return base_types[key]
        values = [row[key] for row in rows if row.get(key) is not None]
        if any(isinstance(value, str) for value in values):
            return "TEXT"
        if any(isinstance(value, float) for value in values):
            return "REAL"
        return "INTEGER"

    db_path = local / "runs.db"
    tmp = local / "runs.db.tmp"
    if tmp.exists():
        tmp.unlink()
    quoted = lambda key: '"' + key.replace('"', '""') + '"'
    with sqlite3.connect(tmp) as connection:
        definitions = ", ".join(f"{quoted(key)} {sql_type(key)}" for key in keys)
        connection.execute(f"CREATE TABLE runs ({definitions})")
        if rows:
            columns = ", ".join(quoted(key) for key in keys)
            placeholders = ", ".join("?" for _ in keys)
            connection.executemany(
                f"INSERT INTO runs ({columns}) VALUES ({placeholders})",
                [[row.get(key) for key in keys] for row in rows],
            )
    for suffix in ("", "-wal", "-shm"):
        old = Path(str(db_path) + suffix)
        if old.exists():
            old.unlink()
    os.replace(tmp, db_path)
    print(f"Indexed {len(rows)} run(s) -> {db_path}")


def sync(goals: list[str], assume_yes: bool) -> None:
    overrides = load_curation()
    selected = _selected_goals(goals, overrides)
    if not goals or goals == ["all"]:
        selected = sorted(set(selected) | set(_remote_goals()))
    before = _effective_decisions(selected, overrides)
    deleted = _delete_decisions(before, selected, assume_yes)
    # A successful deletion needs no tombstone: remove post-hoc delete rows so
    # future syncs stay quiet. Kept labels remain the durable scientific catalog.
    pruned = False
    for path in deleted:
        if overrides.get(path, (None, ""))[0] == "delete":
            del overrides[path]
            pruned = True
    if pruned:
        save_curation(overrides)
    for goal in selected:
        _sync_goal(goal)

    # A brand-new remotely produced sweep can carry curation_state="delete" in
    # sweep_config.json. It becomes visible only after the pull, so enforce it now.
    after = _effective_decisions(selected, overrides)
    new_deletes = {
        path: decision for path, decision in after.items()
        if decision[0] == "delete" and path not in before
    }
    _delete_decisions(new_deletes, selected, assume_yes)
    for goal in selected:
        _rebuild_goal(goal)
    remaining = [
        sweep for sweep in _sweeps(selected)
        if f"{sweep.parent.name}/{sweep.name}" not in after
    ]
    print(f"\nSync complete. {len(remaining)} sweep(s) still need review.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Curate sweeps and mirror cluster data locally."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    curate_parser = commands.add_parser("curate", help="interactively label sweeps")
    curate_parser.add_argument("goals", nargs="*", metavar="GOAL")
    curate_parser.add_argument("--all", action="store_true", help="also revisit labelled sweeps")

    sync_parser = commands.add_parser("sync", help="apply deletions, mirror, and rebuild indexes")
    sync_parser.add_argument("goals", nargs="*", metavar="GOAL")
    sync_parser.add_argument("--yes", action="store_true", help="confirm labelled deletions")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "curate":
            curate(args.goals, args.all)
        else:
            sync(args.goals, args.yes)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
