#!/usr/bin/env bash
# Sync cell-convergence runs and rebuild their local run index.
set -e

SERVER="leroy@10.187.172.7"
REMOTE="/storage/leroy/data"
OPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOCAL="$OPT_ROOT/data"

rsync -avz --progress "${SERVER}:${REMOTE}/convergence/" "${LOCAL}/convergence/"

cd "$OPT_ROOT"
python3 -m src.db backfill "${LOCAL}/convergence/runs.db"
