#!/usr/bin/env python3
"""Two-command sweep curation and cluster/local synchronization.

User workflow
-------------
    python manage_data.py curate [goal]
    python manage_data.py sync [goal] [--yes]

The cluster is authoritative for raw data.  ``curation.csv`` is authoritative
for the human keep/delete decision.  Analyses read the run directories
directly, so there is no index to rebuild.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Iterable

from src import curation


ROOT = Path(__file__).resolve().parent
LOCAL_DATA = Path(os.environ.get("OCTOPUS_DATA_ROOT", ROOT / "data")).resolve()
# src.curation owns the registry path so this tool and anything that reads a
# sweep's curation state cannot end up looking at two different files.
CURATION_FILE = Path(curation.CURATION_PATH).resolve()
SERVER = os.environ.get("OCTOPUS_DATA_SERVER", "leroy@10.187.172.7")
REMOTE_DATA = os.environ.get("OCTOPUS_REMOTE_DATA_ROOT", "/storage/leroy/data").rstrip("/")

STATES = curation.CURATION_STATES
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
    """Strictly parsed registry, with every key checked as a safe GOAL/SWEEP path.

    These keys are interpolated into a remote ``rm -rf``, so the shape check stays
    here even though src.curation does the parsing.
    """
    return {_validate_sweep_path(path): decision
            for path, decision in curation.read_curation(CURATION_FILE, strict=True).items()}


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


def _effective_decisions(
    goals: Iterable[str], overrides: dict[str, tuple[str, str]]
) -> dict[str, tuple[str, str]]:
    """Merge each sweep's own default with the tracked post-hoc overrides.

    Same precedence as src.curation.sweep_curation, which the run indexer applies.
    """
    effective: dict[str, tuple[str, str]] = {}
    for sweep in _sweeps(goals):
        configured = curation.configured_curation(str(sweep))
        if configured[0] in STATES:
            effective[f"{sweep.parent.name}/{sweep.name}"] = configured
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


RSYNC_FLAGS = ["-az", "--delete"]


def _sync_goal(goal: str, assume_yes: bool) -> None:
    """Mirror one goal from the cluster, which is authoritative for raw data.

    --delete makes the local tree a true mirror, so a sweep that exists only
    locally (produced by a local run, or already removed upstream) disappears.
    That is silent data loss for anything never pushed, so a dry run reports the
    local-only sweeps first and asks before any deletion.
    """
    remote = f"{REMOTE_DATA}/{goal}"
    if not _remote_exists(remote):
        print(f"skip {goal}: no cluster directory")
        return
    local = LOCAL_DATA / goal
    local.mkdir(parents=True, exist_ok=True)
    print(f"\nsync {goal}")
    transfer = ["-e", "ssh " + " ".join(SSH_OPTIONS), f"{SERVER}:{remote}/", f"{local}/"]
    preview = subprocess.run(
        ["rsync", *RSYNC_FLAGS, "--dry-run", "--out-format=%o %n", *transfer],
        check=True, capture_output=True, text=True,
    )
    doomed = sorted({line.split(" ", 1)[1].strip("/").split("/")[0]
                     for line in preview.stdout.splitlines()
                     if line.startswith("del. ") and line.count("/") >= 1})
    if doomed:
        print(f"  {len(doomed)} local sweep(s) absent from the cluster will be removed:")
        for name in doomed:
            print(f"    {goal}/{name}")
        if not assume_yes:
            if not sys.stdin.isatty():
                raise RuntimeError(f"Mirroring {goal} would delete local-only data; "
                                   "rerun interactively or pass --yes.")
            if input("  Type 'mirror' to continue: ").strip() != "mirror":
                raise RuntimeError("Synchronization cancelled; no data were deleted.")
    _run(["rsync", *RSYNC_FLAGS, *transfer])

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
        _sync_goal(goal, assume_yes)

    # A brand-new remotely produced sweep can carry curation_state="delete" in
    # sweep_config.json. It becomes visible only after the pull, so enforce it now.
    after = _effective_decisions(selected, overrides)
    new_deletes = {
        path: decision for path, decision in after.items()
        if decision[0] == "delete" and path not in before
    }
    _delete_decisions(new_deletes, selected, assume_yes)
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

    sync_parser = commands.add_parser("sync", help="apply labelled deletions, then mirror the cluster")
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
