"""Which sweeps are worth keeping, and why.

`curation.csv` at the repository root is the human keep/delete decision, tracked in
git so it survives any local data being deleted or re-synced.  A sweep may also
carry its own default in the `curation_state` / `curation_label` fields of its
saved `sweep_config.json`; the CSV overrides it.

This is the ONLY reader of those two sources.  `manage_data.py` writes the CSV and
must agree with anything that reads it.
"""
import csv
import json
import os
from typing import Optional


CURATION_PATH = os.environ.get(
    "OCTOPUS_CURATION_FILE",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "curation.csv"),
)
CURATION_STATES = frozenset({"keep", "delete"})
_CURATION_CACHE: tuple[Optional[tuple[str, int]], dict[str, tuple[str, str]]] = (None, {})


def read_curation(path: Optional[str] = None,
                  strict: bool = False) -> dict[str, tuple[str, str]]:
    """Parse curation.csv into {"goal/sweep": (state, label)}.

    The single reader for the human keep/delete registry: `manage_data.py` and the
    index must never disagree about what a row means.  strict=True raises on a
    malformed row (used when about to rewrite the file); the default skips it, so
    indexing a run never fails because of an unrelated typo.  Results are cached
    per file until its mtime changes.
    """
    global _CURATION_CACHE
    path = str(path or CURATION_PATH)
    try:
        stamp = (path, os.stat(path).st_mtime_ns)
    except FileNotFoundError:
        stamp = None
    if not strict and stamp is not None and _CURATION_CACHE[0] == stamp:
        return dict(_CURATION_CACHE[1])

    catalog: dict[str, tuple[str, str]] = {}
    if stamp is not None:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                path = row.get("path", "").strip().strip("/")
                state = row.get("state", "").strip()
                label = row.get("label", "").strip()
                if state not in CURATION_STATES or not path:
                    if strict:
                        raise ValueError(f"Invalid curation state {state!r} for {path!r}")
                    continue
                if state == "keep" and not label:
                    if strict:
                        raise ValueError(f"A kept sweep needs a label: {path}")
                    continue
                catalog[path] = (state, label)
    if not strict:
        _CURATION_CACHE = (stamp, dict(catalog))
    return catalog


def configured_curation(sweep_dir: str) -> tuple[str, str]:
    """The sweep's own default, from the RunConfig fields saved in sweep_config.json.

    A "keep" without a label is not a decision, so it degrades to "review".
    """
    try:
        with open(os.path.join(sweep_dir, "sweep_config.json")) as f:
            config = json.load(f)
        state = config.get("curation_state", "review")
        label = str(config.get("curation_label", "")).strip()
    except (OSError, ValueError, TypeError, AttributeError):
        return "review", ""
    if state == "delete" or (state == "keep" and label):
        return state, label
    return "review", ""


def sweep_curation(data_root: str, sweep_folder: str) -> tuple[str, str]:
    """Return registry override, then config default, then implicit review."""
    goal = os.path.basename(os.path.abspath(data_root))
    decision = read_curation().get(f"{goal}/{sweep_folder}")
    if decision is not None:
        return decision
    return configured_curation(os.path.join(data_root, sweep_folder))
