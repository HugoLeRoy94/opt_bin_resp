#!/usr/bin/env bash
# Syncs all single_run_* directories from the server (no runs.db — standalone dirs).
set -e

SERVER="leroy@10.187.172.7"
REMOTE="/storage/leroy/data"
OPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="$OPT_ROOT/data"

for dir in $(ssh "${SERVER}" "ls -d ${REMOTE}/single_run_* 2>/dev/null"); do
    name="$(basename "$dir")"
    rsync -avz --progress "${SERVER}:${dir}/" "${LOCAL}/${name}/"
done
