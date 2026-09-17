#!/usr/bin/env bash
# Sync cell/receptor-equivalence runs and rebuild their local run index.
set -e

SERVER="leroy@10.187.172.7"
REMOTE="/storage/leroy/data"
OPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOCAL="$OPT_ROOT/data"

rsync -avz --progress "${SERVER}:${REMOTE}/equivalence/" "${LOCAL}/equivalence/"

cd "$OPT_ROOT"
python3 -m src.db backfill "${LOCAL}/equivalence/runs.db"
