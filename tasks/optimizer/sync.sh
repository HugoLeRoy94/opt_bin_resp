#!/usr/bin/env bash
set -e

SERVER="leroy@10.187.172.7"
REMOTE="/storage/leroy/data"
OPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="$OPT_ROOT/data"

rsync -avz --progress "${SERVER}:${REMOTE}/optimizer/" "${LOCAL}/optimizer/"

cd "$OPT_ROOT"
python -m src.db backfill "${LOCAL}/optimizer/runs.db"
