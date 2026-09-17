#!/usr/bin/env bash
# Sync all standalone single-run roots through the common workflow.
set -e

SERVER="leroy@10.187.172.7"
REMOTE="/storage/leroy/data"
OPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

for dir in $(ssh "${SERVER}" "ls -d ${REMOTE}/single_run_* 2>/dev/null"); do
    name="$(basename "$dir")"
    python3 "$OPT_ROOT/manage_data.py" sync "$name"
done
